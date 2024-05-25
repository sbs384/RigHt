import logging
from typing import Mapping, Optional,Any

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .arguments import ModelArguments, DataArguments, \
    CETrainingArguments as TrainingArguments

logger = logging.getLogger(__name__)


class CrossEncoder(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, model_args: ModelArguments, data_args: DataArguments,
                 train_args: TrainingArguments):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )

        if train_args.local_rank >= 0:
            self.world_size = dist.get_world_size()
        self._keys_to_ignore_on_save = None

    def forward(self, batch):
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        logits = ranker_out.logits

        if self.train_args.temperature is not None:
            logits = logits / self.train_args.temperature
        if self.training:
            scores = logits.view(
                -1, # self.train_args.per_device_train_batch_size,
                self.data_args.train_group_size
            )
            ranker_out.logits = scores
            target_label = scores.new_zeros(scores.shape[0], dtype=torch.long)
            loss = self.cross_entropy(scores, target_label)
            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out
            )
        else:
            # 验证集计算指标
            if self.data_args.dev_corpus_file != None:
                ranker_out.logits = logits.reshape(-1,self.data_args.train_group_size).argmax(-1)
            return ranker_out

    def load_state_dict(self,
                        state_dict: Mapping[str, Any],
                        strict: bool = True):
        return self.hf_model.load_state_dict(state_dict,strict)

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        reranker = cls(hf_model, model_args, data_args, train_args)
        return reranker

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
