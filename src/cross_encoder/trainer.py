import logging
import os
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from transformers.trainer import Trainer, nested_detach
from transformers.trainer_utils import PredictionOutput, EvalPrediction, EvalLoopOutput
from datasets import load_metric
import evaluate

from .modeling import CrossEncoder

logger = logging.getLogger(__name__)
# metric = evaluate.load("accuracy", "f1")


def compute_metrics(eval_pred):
    train_group_size = 8
    logits: np.ndarray = eval_pred.predictions
    # print(logits.shape)
    # logits = logits.reshape(-1, train_group_size)
    # predictions = np.argmax(logits, axis=-1)
    predictions = torch.tensor(logits)
    labels = torch.zeros_like(predictions)
    acc = (predictions == labels).float().mean().item()
    return {
        'accuracy': acc,
    }


class CETrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model: CrossEncoder, inputs):
        return model(inputs)['loss']

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Tuple[Dict[str, Union[torch.Tensor, Any]]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor],
               Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config,
                                      "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.args.fp16:
                with autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            loss = None
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items()
                               if k not in ignore_keys)
            else:
                logits = outputs

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        labels = None

        return (loss, logits, labels)

    def prediction_loop(self, *args, **kwargs) -> PredictionOutput:
        pred_outs = super().prediction_loop(*args, **kwargs)
        preds, label_ids, metrics = pred_outs.predictions, pred_outs.label_ids, pred_outs.metrics
        preds = preds.squeeze()
        if self.compute_metrics is not None:
            metrics_no_label = self.compute_metrics(
                EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics_no_label = {}

        for key in list(metrics_no_label.keys()):
            if not key.startswith("eval_"):
                metrics_no_label[f"eval_{key}"] = metrics_no_label.pop(key)

        return PredictionOutput(predictions=preds,
                                label_ids=label_ids,
                                metrics={
                                    **metrics,
                                    **metrics_no_label
                                })

    def evaluation_loop(self, *args, **kwargs) -> EvalLoopOutput:
        pred_outs = super().evaluation_loop(*args, **kwargs)
        preds, label_ids, metrics = pred_outs.predictions, pred_outs.label_ids, pred_outs.metrics
        preds = preds.squeeze()
        if self.compute_metrics is not None:
            # preds = preds[1]
            metrics_no_label = self.compute_metrics(
                EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics_no_label = {}

        for key in list(metrics_no_label.keys()):
            if not key.startswith("eval_"):
                metrics_no_label[f"eval_{key}"] = metrics_no_label.pop(key)

        return EvalLoopOutput(predictions=preds,
                              label_ids=label_ids,
                              metrics={
                                  **metrics,
                                  **metrics_no_label
                              },
                              num_samples=pred_outs.num_samples)
