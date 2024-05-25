from torch.cuda.amp import autocast
from transformers.trainer import *
import evaluate
import math
import torch.nn.functional as F
from bi_encoder.test import eval_test
from bi_encoder.arguments import DataArguments
from copy import deepcopy

# metric = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits: torch.Tensor = eval_pred.predictions[2]
#     # print(logits.shape)
#     # logits = logits.reshape(-1, train_group_size)
#     # print('compute_metrics', logits.shape)
#     n_samples, batch_size = logits.shape
#     labels = list(range(batch_size)) * (math.ceil(n_samples / batch_size))
#     labels = labels[:n_samples]
#     predictions = np.argmax(logits, axis=-1)
#     metrics = metric.compute(predictions=predictions, references=labels)
#     metrics['loss'] = F.cross_entropy(torch.tensor(logits,dtype=torch.float32),
#                                      torch.tensor(labels,dtype=torch.long),
#                                      reduction='mean').item()
#     return metrics


def wrapped_compute_metrics(model, training_args: TrainingArguments,
                            data_args: DataArguments, tokenizer):
    args = deepcopy(training_args)
    # training_args.disable_tqdm = True
    eval_trainer = BiTrainer(model=model, args=args)

    def compute_metrics(_):
        return eval_test(eval_trainer, data_args, tokenizer, split='dev')

    return compute_metrics


class BiTrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(f'MODEL {self.model.__class__.__name__} '
                                      f'does not support save interface')
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

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
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

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
        # preds, label_ids, metrics = None, None, {}
        # len(preds)=3
        # q_resp, r_reps, scores
        if self.compute_metrics is not None:
            # preds = preds[1]
            metrics_no_label = self.compute_metrics(
                EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics_no_label = {}

        for key in list(metrics_no_label.keys()):
            if not key.startswith("eval_"):
                metrics_no_label[f"eval_{key}"] = metrics_no_label.pop(key)
        dataloader = args[0]
        return EvalLoopOutput(predictions=preds,
                              label_ids=label_ids,
                              metrics={
                                  **metrics,
                                  **metrics_no_label
                              },
                              num_samples=len(dataloader))
    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = self.state.best_model_checkpoint
        if os.path.exists(best_model_path):
            if not hasattr(self.model, 'load_model'):
                raise NotImplementedError(f'MODEL {self.model.__class__.__name__} '
                                        f'does not support load interface')
            self.model.load_model(best_model_path)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )