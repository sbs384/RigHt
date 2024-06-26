

import copy
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Any, Union, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from bi_encoder.arguments import ModelArguments, \
    RetrieverTrainingArguments as TrainingArguments, DataArguments
from torch import nn, Tensor
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig, BertModel, BertForMaskedLM
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)
kl_losses=[]
clw_losses=[]

@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class DensePooler(nn.Module):

    def __init__(self, input_dim: int = 768, output_dim: int = 768, tied=True):
        super(DensePooler, self).__init__()
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)
        self._config = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'tied': tied
        }

    def load(self, model_dir: str):
        pooler_path = os.path.join(model_dir, 'pooler.pt')
        if pooler_path is not None:
            if os.path.exists(pooler_path):
                logger.info(f'Loading Pooler from {pooler_path}')
                state_dict = torch.load(pooler_path, map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)

    def forward(self, q: Tensor = None, p: Tensor = None, **kwargs):
        if q is not None:
            return self.linear_q(q)
        elif p is not None:
            return self.linear_p(p)
        else:
            raise ValueError


class BiEncoderModel(PreTrainedModel):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 pooler: nn.Module = None,
                 untie_encoder: bool = False,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_x_device: bool = False,
                 temperature: float = 1.0,
                 contrastive_loss_weight: float = 1.0,
                 kl_loss_weight: float = 1.0,
                 model_args: ModelArguments = None):
        super().__init__(PretrainedConfig())
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.untie_encoder = untie_encoder
        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.negatives_x_device = negatives_x_device
        self.contrastive_loss_weight = contrastive_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.model_args = model_args
        if self.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError(
                    'Distributed training has not been initialized for representation all gather.'
                )
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self._keys_to_ignore_on_save = None
        self.softmax = nn.Softmax(dim=1)
        if self.sentence_pooling_method == 'att':
            self.att_layer = nn.Sequential(nn.Linear(768, 300, bias=False),
                                           nn.Tanh(),
                                           nn.Linear(300, 1, bias=False))
    def embeddings(self, lm: PreTrainedModel, input_ids):
        if isinstance(lm, BertModel):
            embeddings = lm.embeddings.word_embeddings
        elif isinstance(lm, BertForMaskedLM):
            embeddings = lm.bert.embeddings.word_embeddings
        else:
            raise RuntimeError(
                "unknown how to get word embeddings for given model", type(lm))
        return embeddings(input_ids)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == 'att':
            att_logits = self.att_layer(hidden_state) - 1e9 * (
                1 - mask.float().cuda()).unsqueeze(2)
            att_scores = self.softmax(att_logits).transpose(2, 1)
            pooler_output = torch.bmm(att_scores, hidden_state).squeeze(1)
            return pooler_output
        
    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        p_reps = self.sentence_embedding(p_hidden, psg['attention_mask'])
        if self.pooler is not None:
            p_reps = self.pooler(p=p_reps)  # D * d
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        q_reps = self.sentence_embedding(q_hidden, qry['attention_mask'])
        if self.pooler is not None:
            q_reps = self.pooler(q=q_reps)
        if self.normlized:
            q_reps = torch.nn.functional.normalize(q_reps, dim=-1)
        return q_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = DensePooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @staticmethod
    def build_pooler(model_args):
        pooler = DensePooler(model_args.projection_in_dim,
                             model_args.projection_out_dim,
                             tied=not model_args.untie_encoder)
        pooler.load(model_args.model_name_or_path)
        return pooler

    def adv_training(self,
                     query: Dict[str, Tensor] = None,
                     passage: Dict[str, Tensor] = None,
                     teacher_score: Tensor = None):
        qry_inputs_embeds = self.embeddings(
            self.lm_q, query.pop('input_ids')).clone().detach()
        psg_inputs_embeds = self.embeddings(
            self.lm_p, passage.pop('input_ids')).clone().detach()
        qry_inputs_embeds.requires_grad = True
        psg_inputs_embeds.requires_grad = True
        query['inputs_embeds'] = qry_inputs_embeds
        passage['inputs_embeds'] = psg_inputs_embeds
        output: EncoderOutput = self(query,
                                     passage,
                                     teacher_score,
                                     adv_training=False)
        loss = output.loss
        qry_grad, psg_grad = torch.autograd.grad(
            loss, [qry_inputs_embeds, psg_inputs_embeds],
            retain_graph=False,
            create_graph=False)

        qry_inputs_embeds = qry_inputs_embeds + self.model_args.adv_norm * qry_grad.sign(
        )
        psg_inputs_embeds = psg_inputs_embeds + self.model_args.adv_norm * psg_grad.sign(
        )
        query['inputs_embeds'] = qry_inputs_embeds
        passage['inputs_embeds'] = psg_inputs_embeds
        return self(query, passage, teacher_score, adv_training=False)

    def forward(self,
                query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None,
                teacher_score: Tensor = None,
                adv_training=True):
        if self.training and adv_training and self.model_args.adv_training:
            return self.adv_training(query, passage, teacher_score)
        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps,
                                 p_reps=p_reps,
                                 loss=None,
                                 scores=None)

        if self.training:
            kl_loss = 0.0
            if teacher_score is not None:
                student_p = p_reps.view(q_reps.size(0), -1,
                                        q_reps.size(-1))  # B N D
                student_q = q_reps.view(q_reps.size(0), 1,
                                        q_reps.size(-1))  # B 1 D
                student_score = self.compute_similarity(
                    student_q, student_p).squeeze(1)  # B N

                input = F.log_softmax(student_score / self.temperature, dim=-1)
                target = F.softmax(teacher_score, dim=-1)
                kl_loss = self.kl_loss_weight * self.kl(input, target)
                kl_losses.append(kl_loss.item())
                # scores=student_score
                # target=torch.zeros(scores.size(0),
                #                   device=scores.device,
                #                   dtype=torch.long)
            # else:
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores / self.temperature
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0),
                                device=scores.device,
                                dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))
            

            # if self.negatives_x_device:
            #     q_reps = self._dist_gather_tensor(q_reps)
            #     p_reps = self._dist_gather_tensor(p_reps)

            

            loss = self.compute_loss(scores, target)
            clw_losses.append(loss.item())
            
            if teacher_score is not None:
                loss = kl_loss + self.contrastive_loss_weight * loss

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores / self.temperature
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        train_args: TrainingArguments,
        **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:

                _qry_model_path = os.path.join(model_args.model_name_or_path,
                                               'query_model')
                _psg_model_path = os.path.join(model_args.model_name_or_path,
                                               'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(
                    f'loading query model weight from {_qry_model_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    _qry_model_path, **hf_kwargs)
                logger.info(
                    f'loading passage model weight from {_psg_model_path}')
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(
                    _psg_model_path, **hf_kwargs)
            else:
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(lm_q=lm_q,
                    lm_p=lm_p,
                    pooler=pooler,
                    negatives_x_device=train_args.negatives_x_device,
                    untie_encoder=model_args.untie_encoder,
                    normlized=model_args.normlized,
                    sentence_pooling_method=model_args.sentence_pooling_method,
                    temperature=train_args.temperature,
                    contrastive_loss_weight=train_args.contrastive_loss_weight,
                    kl_loss_weight=train_args.kl_loss_weight,
                    model_args=model_args)
        return model

    @classmethod
    def load(
        cls,
        model_name_or_path,
        normlized,
        sentence_pooling_method,
        model_args=None,
        **hf_kwargs,
    ):
        # load local
        untie_encoder = True
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            if os.path.exists(_qry_model_path):
                logger.info(
                    f'found separate weight for query/passage encoders')
                logger.info(
                    f'loading query model weight from {_qry_model_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    _qry_model_path, **hf_kwargs)
                logger.info(
                    f'loading passage model weight from {_psg_model_path}')
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(
                    _psg_model_path, **hf_kwargs)
                untie_encoder = False
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path,
                                                       **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.load_pooler(model_name_or_path, **pooler_config_dict)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            untie_encoder=untie_encoder,
            normlized=normlized,
            sentence_pooling_method=sentence_pooling_method,
        )
        return model

    # def load_state_dict(self,
    #                     state_dict: Mapping[str, Any],
    #                     strict: bool = True):
    #     # 把pooler忽视了，没有用到反正
    #     if self.untie_encoder:
    #         return super().load_state_dict(state_dict, strict)
    #     return self.lm_q.load_state_dict(state_dict)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "10GB",
        **kwargs,
    ):
        if os.path.isfile(save_directory):
            logger.error(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
            return
        output_dir = save_directory
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir,
                                                   'passage_model'))
        else:
            self.lm_q.save_pretrained(output_dir)
        if self.pooler:
            self.pooler.save_pooler(output_dir)

    def load_model(self, ckpt_dir):
        if self.untie_encoder:
            self.lm_q = self.lm_q.from_pretrained(
                os.path.join(ckpt_dir, 'query_model'))
            self.lm_p = self.lm_p.from_pretrained(
                os.path.join(ckpt_dir, 'passage_model'))
        else:
            self.lm_q = self.lm_p = self.lm_q.save_pretrained(ckpt_dir)
        if self.pooler:
            self.pooler.load(ckpt_dir)