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
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig, BertModel, BertForMaskedLM, AutoModelForMaskedLM
from transformers.file_utils import ModelOutput
from bi_encoder.utils import aggregate, cal_remove_dim

logger = logging.getLogger(__name__)
kl_losses = []
clw_losses = []


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class DensePooler(nn.Module):

    def __init__(self,
                 input_dim: int = 768,
                 output_dim: int = 768,
                 tied=True,
                 name='pooler'):
        super(DensePooler, self).__init__()
        self.name = name
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

    def forward(self, q: Tensor = None, p: Tensor = None):
        if q is not None:
            return self.linear_q(q)
        elif p is not None:
            return self.linear_p(p)
        else:
            raise ValueError

    def load(self, ckpt_dir: str):
        if ckpt_dir is not None:
            _pooler_path = os.path.join(ckpt_dir, '{}.pt'.format(self.name))
            if os.path.exists(_pooler_path):
                logger.info(f'Loading Pooler from {ckpt_dir}')
                state_dict = torch.load(os.path.join(ckpt_dir, '{}.pt'.format(
                    self.name)),
                                        map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training {} from scratch".format(self.name))
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(),
                   os.path.join(save_path, '{}.pt'.format(self.name)))
        with open(
                os.path.join(save_path, '{}_config.json').format(self.name),
                'w') as f:
            json.dump(self._config, f)


class BiEncoderModel(PreTrainedModel):
    TRANSFORMER_CLS = AutoModel
    POOLER = DensePooler

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
                 term_weight_trans: nn.Module = None,
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
        self.term_weight_trans = term_weight_trans
        self.softmax = nn.Softmax(dim=-1)
        if self.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError(
                    'Distributed training has not been initialized for representation all gather.'
                )
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self._keys_to_ignore_on_save = None

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

    def encode_passage(self, psg, skip_mlm):
        if psg is None:
            return None, None

        psg_out = self.lm_p(**psg, output_hidden_states=True, return_dict=True)
        p_seq_hidden = psg_out.hidden_states[-1]
        p_cls_hidden = p_seq_hidden[:, 0]  # get [CLS] embeddings
        p_term_weights = self.term_weight_trans(
            p_seq_hidden[:, 1:])  # batch, seq, 1

        if not skip_mlm:
            p_logits = psg_out.logits[:, 1:]  # batch, seq, vocab
            p_logits = self.softmax(p_logits)
            attention_mask = psg['attention_mask'][:, 1:].unsqueeze(-1)
            p_lexical_reps = torch.max(
                (p_logits * p_term_weights) * attention_mask, dim=-2).values
        else:
            ## w/o MLM
            ## p_term_weights = torch.relu(p_term_weights)
            p_lexical_reps = torch.zeros(
                p_seq_hidden.shape[0],
                p_seq_hidden.shape[1],
                30522,
                dtype=p_seq_hidden.dtype,
                device=p_seq_hidden.device)  # (batch, seq, vocab)
            p_lexical_reps = torch.scatter(p_lexical_reps,
                                           dim=-1,
                                           index=psg.input_ids[:, 1:, None],
                                           src=p_term_weights)
            p_lexical_reps = p_lexical_reps.max(-2).values

        if self.pooler is not None:
            p_semantic_reps = self.pooler(p=p_cls_hidden)  # D * d
        else:
            p_semantic_reps = None

        return p_lexical_reps, p_semantic_reps

    def encode_query(self, qry, skip_mlm):
        if qry is None:
            return None, None
        qry_out = self.lm_q(**qry, output_hidden_states=True, return_dict=True)
        q_seq_hidden = qry_out.hidden_states[-1]
        q_cls_hidden = q_seq_hidden[:, 0]  # get [CLS] embeddings

        q_term_weights = self.term_weight_trans(
            q_seq_hidden[:, 1:])  # batch, seq, 1

        if not skip_mlm:
            q_logits = qry_out.logits[:, 1:]  # batch, seq-1, vocab
            q_logits = self.softmax(q_logits)
            attention_mask = qry['attention_mask'][:, 1:].unsqueeze(-1)
            q_lexical_reps = torch.max(
                (q_logits * q_term_weights) * attention_mask, dim=-2).values
        else:
            # w/o MLM
            # q_term_weights = torch.relu(q_term_weights)
            q_lexical_reps = torch.zeros(
                q_seq_hidden.shape[0],
                q_seq_hidden.shape[1],
                30522,
                dtype=q_seq_hidden.dtype,
                device=q_seq_hidden.device)  # (batch, len, vocab)
            q_lexical_reps = torch.scatter(q_lexical_reps,
                                           dim=-1,
                                           index=qry.input_ids[:, 1:, None],
                                           src=q_term_weights)
            q_lexical_reps = q_lexical_reps.max(-2).values

        if self.pooler is not None:
            q_semantic_reps = self.pooler(q=q_cls_hidden)
        else:
            q_semantic_reps = None

        return q_lexical_reps, q_semantic_reps

    @staticmethod
    def merge_reps(lexical_reps, semantic_reps):
        dim = lexical_reps.shape[1] + semantic_reps.shape[1]
        merged_reps = torch.zeros(lexical_reps.shape[0],
                                  dim,
                                  dtype=lexical_reps.dtype,
                                  device=lexical_reps.device)
        merged_reps[:, :lexical_reps.shape[1]] = lexical_reps
        merged_reps[:, lexical_reps.shape[1]:] = semantic_reps
        return merged_reps

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

    @staticmethod
    def build_term_weight_transform(model_args):
        term_weight_trans = DensePooler(model_args.projection_in_dim,
                                        1,
                                        tied=not model_args.untie_encoder,
                                        name="TermWeightTrans")
        term_weight_trans.load(model_args.model_name_or_path)
        return term_weight_trans

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

    def embed_query(self, qry, skip_mlm):
        # encode, aggregate & merge
        lexical_reps, semantic_reps = self.encode_query(qry, skip_mlm)
        lexical_reps = aggregate(
            lexical_reps,
            self.model_args.agg_dim,
            full=not self.model_args.semi_aggregate)
        if semantic_reps is not None:
            reps = self.merge_reps(lexical_reps, semantic_reps)
        else:
            reps = lexical_reps
        return reps

    def forward(self,
                query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None,
                teacher_score: Tensor = None,
                adv_training=True):

        if self.training and adv_training and self.model_args.adv_training:
            return self.adv_training(query, passage, teacher_score)
        q_lexical_reps, q_semantic_reps = self.encode_query(
            query, self.model_args.skip_mlm)
        p_lexical_reps, p_semantic_reps = self.encode_passage(
            passage, self.model_args.skip_mlm)
        q_reps, p_reps = None, None
        # for inference
        if q_lexical_reps is None or p_lexical_reps is None:

            if query is not None:
                q_lexical_reps = aggregate(
                    q_lexical_reps,
                    self.model_args.agg_dim,
                    full=not self.model_args.semi_aggregate)
                if q_semantic_reps is not None:
                    q_reps = self.merge_reps(q_lexical_reps, q_semantic_reps)
                else:
                    q_reps = q_lexical_reps

            if passage is not None:
                p_lexical_reps = aggregate(
                    p_lexical_reps,
                    self.model_args.agg_dim,
                    full=not self.model_args.semi_aggregate)
                if p_semantic_reps is not None:
                    p_reps = self.merge_reps(p_lexical_reps, p_semantic_reps)
                else:
                    p_reps = p_lexical_reps

            return EncoderOutput(q_reps=q_reps,
                                 p_reps=p_reps,
                                 loss=None,
                                 scores=None)

        if self.training:
            q_tok_reps = aggregate(q_lexical_reps,
                                   self.model_args.agg_dim,
                                   full=not self.model_args.semi_aggregate)
            p_tok_reps = aggregate(p_lexical_reps,
                                   self.model_args.agg_dim,
                                   full=not self.model_args.semi_aggregate)

            kl_loss = 0.0
            if teacher_score is not None:
                n_q, n_d = q_tok_reps.shape
                # student_p = p_reps.view(q_reps.size(0), -1,
                #                         q_reps.size(-1))  # B N D
                # student_q = q_reps.view(q_reps.size(0), 1,
                #                         q_reps.size(-1))  # B 1 D
                student_p_topk = p_tok_reps.view(n_q, -1, n_d)
                student_q_topk = q_tok_reps.view(n_q, -1, n_d)
                lexical_scores = self.compute_similarity(
                    student_q_topk, student_p_topk).squeeze(1)
                if q_semantic_reps is not None:
                    n_q, n_d = q_semantic_reps.shape
                    student_p_semantic = p_semantic_reps.view(n_q, -1, n_d)
                    student_q_semantic = q_semantic_reps.view(n_q, -1, n_d)
                    semantic_scores = self.compute_similarity(
                        student_q_semantic, student_p_semantic).squeeze(1)
                else:
                    semantic_scores = 0
                # student_score = self.compute_similarity(
                #     student_q, student_p).squeeze(1)  # B N
                student_score = lexical_scores + semantic_scores

                input = F.log_softmax(student_score / self.temperature, dim=-1)
                lexical_input = F.log_softmax(lexical_scores /
                                              self.temperature,
                                              dim=-1)
                semantic_input = F.log_softmax(semantic_scores /
                                               self.temperature,
                                               dim=-1)
                target = F.softmax(teacher_score, dim=-1)
                kl_loss = self.kl(input, target)
                if q_semantic_reps is not None:
                    kl_loss = kl_loss + 0.5 * self.kl(
                        lexical_input, target) + 0.5 * self.kl(
                            semantic_input, target)
                kl_loss *= self.kl_loss_weight
                kl_losses.append(kl_loss.item())

                # lexical matching

            lexical_scores = self.compute_similarity(q_tok_reps, p_tok_reps)

            # semantic matching
            if q_semantic_reps is not None:
                semantic_scores = self.compute_similarity(
                    q_semantic_reps, p_semantic_reps)
            else:
                semantic_scores = 0
            # fusion
            scores = lexical_scores + semantic_scores

            # scores = self.compute_similarity(q_reps, p_reps)
            # if self.temperature < 1:
            scores = scores / self.temperature
            # scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0),
                                  device=scores.device,
                                  dtype=torch.long)
            target = target * (p_semantic_reps.size(0) //
                               q_semantic_reps.size(0))

            # if self.negatives_x_device:
            #     q_reps = self._dist_gather_tensor(q_reps)
            #     p_reps = self._dist_gather_tensor(p_reps)
            target = torch.nn.functional.one_hot(
                target, num_classes=lexical_scores.size(1)).float()
            loss = self.kl(F.log_softmax(scores, dim=-1), target)
            # loss = self.compute_loss(scores, target)
            if q_semantic_reps is not None:
                loss = loss + 0.5 * self.kl(
                    F.log_softmax(lexical_scores, dim=-1),
                    target) + 0.5 * self.kl(
                        F.log_softmax(semantic_scores, dim=-1), target)
                # loss = loss + 0.5 * self.compute_loss(
                #     lexical_scores, target) + 0.5 * self.compute_loss(
                #         semantic_scores, target)
            clw_losses.append(loss.item())

            if teacher_score is not None:
                loss = kl_loss + self.contrastive_loss_weight * loss

        else:
            # scores = self.compute_similarity(q_reps, p_reps)
            # scores = scores / self.temperature
            # lexical matching
            if self.model_args.agg_dim is not None:
                q_tok_reps = aggregate(q_lexical_reps,
                                       self.model_args.agg_dim,
                                       full=not self.model_args.semi_aggregate)
                p_tok_reps = aggregate(p_lexical_reps,
                                       self.model_args.agg_dim,
                                       full=not self.model_args.semi_aggregate)
                lexical_scores = self.compute_similarity(
                    q_tok_reps, p_tok_reps)
            else:
                lexical_scores = self.compute_similarity(
                    q_tok_reps, p_tok_reps)

            # semantic matching
            if q_semantic_reps is not None:
                semantic_scores = self.compute_similarity(
                    q_semantic_reps, p_semantic_reps)
            else:
                semantic_scores = 0

            # score fusion
            scores = scores = lexical_scores + semantic_scores
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
        if model_args.skip_mlm:
            cls.TRANSFORMER_CLS = AutoModel
        else:
            cls.TRANSFORMER_CLS = AutoModelForMaskedLM
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

        term_weight_trans = cls.build_term_weight_transform(model_args)

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
                    term_weight_trans=term_weight_trans,
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

        TermWeightTrans_weights = os.path.join(model_name_or_path,
                                               'TermWeightTrans.pt')
        TermWeightTrans_config = os.path.join(model_name_or_path,
                                              'TermWeightTrans_config.json')
        if os.path.exists(TermWeightTrans_weights) and os.path.exists(
                TermWeightTrans_config):
            logger.info(f'found TermWeightTrans weight and configuration')
            with open(TermWeightTrans_config) as f:
                TermWeightTrans_config_dict = json.load(f)
            # Todo: add name to config
            TermWeightTrans_config_dict['name'] = 'TermWeightTrans'
            term_weight_trans = cls.POOLER(**TermWeightTrans_config_dict)
            term_weight_trans.load(model_name_or_path)
        else:
            term_weight_trans = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            untie_encoder=untie_encoder,
            normlized=normlized,
            sentence_pooling_method=sentence_pooling_method,
            term_weight_trans=term_weight_trans,
            model_args=model_args,
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
        if self.term_weight_trans:
            self.term_weight_trans.save_pooler(output_dir)

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
        if self.term_weight_trans:
            self.term_weight_trans.load(ckpt_dir)


# def build_cross_encoder(model_args:ModelArguments, data_args:DataArguments, training_args:TrainingArguments):
#     config = AutoConfig.from_pretrained(
#         model_args.teacher_model_name_or_path,
#         num_labels=1,
#         cache_dir=model_args.cache_dir,
#     )
#     _model_class = CrossEncoder

#     model = _model_class.from_pretrained(
#         model_args, data_args, training_args,
#         model_args.teacher_model_name_or_path,
#         from_tf=bool(".ckpt" in model_args.model_name_or_path),
#         config=config,
#         cache_dir=model_args.cache_dir,
#     )

#     return model
