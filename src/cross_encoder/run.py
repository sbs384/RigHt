import logging
import os
from pathlib import Path
import sys
sys.path.append('./src')

from cross_encoder.trainer import CETrainer, compute_metrics
from cross_encoder.modeling import CrossEncoder
from cross_encoder.arguments import ModelArguments, DataArguments, \
    CETrainingArguments as TrainingArguments
from cross_encoder.data import TrainDatasetForCE, DevDatasetForCE, PredictionDatasetForCE, GroupCollator
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers.trainer import TrainerCallback, TrainerState, TrainerControl

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    _model_class = CrossEncoder

    model = _model_class.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    if training_args.do_train:
        train_dataset = TrainDatasetForCE(
            data_args, tokenizer=tokenizer, train_args=training_args
        )
    else:
        train_dataset = None

    dev_dataset = None
    if training_args.do_eval:
        dev_dataset = DevDatasetForCE(
            data_args, tokenizer=tokenizer, train_args=training_args
        )

    # Initialize our Trainer
    _trainer_class = CETrainer
    trainer = _trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=GroupCollator(tokenizer),
        compute_metrics=compute_metrics
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    if training_args.do_train:
        if data_args.neg_dir!='':
            callback = EpochCallback(train_dataset)
            trainer.add_callback(callback)
        trainer.train()
        # 加了 load_best_model_at_end 后，会把best model保存到 output_dir 顶层，因此可以直接从那里load
        trainer.save_model()

        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_predict:
        trainer.compute_metrics = None
        logging.info("*** Prediction ***")
        *d, file = data_args.prediction_save_path.split('/')
        Path('/'.join(d)).mkdir(parents=True, exist_ok=True)
        if os.path.exists(data_args.prediction_save_path) and not data_args.overwrite_prediction:
            raise FileExistsError(f"Existing: {data_args.prediction_save_path}. Please save to other paths")

        test_dataset = PredictionDatasetForCE(
            data_args, tokenizer=tokenizer,
            max_len=data_args.max_len,
        )

        pred_scores = trainer.predict(test_dataset=test_dataset).predictions

        if trainer.is_world_process_zero():
            assert len(test_dataset) == len(pred_scores)
            with open(data_args.prediction_save_path, "w") as writer:
                for pair, score in zip(test_dataset.test_data, pred_scores):
                    writer.write(f'{pair[0]}\t{pair[1]}\t{score}\n')

class EpochCallback(TrainerCallback):
    def __init__(self, train_dataset: TrainDatasetForCE) -> None:
        super().__init__()
        self.train_dataset = train_dataset

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.train_dataset.update_negs()
        return super().on_epoch_begin(args, state, control, **kwargs)
    
if __name__ == "__main__":
    main()
