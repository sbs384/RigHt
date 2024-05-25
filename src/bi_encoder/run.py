import sys

sys.path.append('./src')

import torch
from bi_encoder.test import eval_test, read_id
import logging
import os
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from bi_encoder.trainer import BiTrainer, wrapped_compute_metrics
from bi_encoder.arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from bi_encoder.data import DevDatasetForBiE, TrainDatasetForBiE, PredictionDataset, BiCollator, PredictionCollator
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from bi_encoder.hard import HardMiner, EpochCallback
from copy import deepcopy
from transformers.trainer import TrainerCallback, TrainerState, TrainerControl, Trainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if model_args.agg_dim > 0:
        from bi_encoder.agg_model import BiEncoderModel,clw_losses,kl_losses
        assert not model_args.skip_mlm
    else:
        from bi_encoder.bi_model import BiEncoderModel,clw_losses,kl_losses
        assert model_args.skip_mlm

    if (os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir) and training_args.do_train
            and not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    # logger.info('Config: %s', config)

    if training_args.do_train:
        model = BiEncoderModel.build(
            model_args,
            training_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )

    else:
        model = BiEncoderModel.load(
            model_args.model_name_or_path,
            normlized=model_args.normlized,
            model_args=model_args,
            sentence_pooling_method=model_args.sentence_pooling_method)

    # Get datasets
    if training_args.do_train:
        train_dataset = TrainDatasetForBiE(args=data_args, tokenizer=tokenizer)
    else:
        train_dataset = None
    training_args.log_level = "NOTSET"
    training_args.report_to = []
    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=BiCollator(tokenizer,
                                 query_max_len=data_args.query_max_len,
                                 passage_max_len=data_args.passage_max_len),
    )
    evalCallback = EpochEvalCallback(trainer, data_args, tokenizer)
    if training_args.do_eval:
        trainer.add_callback(evalCallback)
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    if training_args.do_train:
        if data_args.neg_source == 'ance':
            hard_miner = None
            if data_args.import_neg_dir == '':
                hard_miner = HardMiner(model, training_args, data_args,
                                       tokenizer)
            epoch_callback = EpochCallback(train_dataset, hard_miner)
            trainer.add_callback(epoch_callback)
        trainer.train()
        if not training_args.do_eval:
            trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_predict:

        model.eval()
        if training_args.do_eval:
            logger.info(
                f"*** Loading best from {evalCallback.best_ckpt} (score: {evalCallback.best_metric}) ***"
            )
            # load from best ckpt saved on top dir
            model_args.model_name_or_path = training_args.output_dir
            model = BiEncoderModel.load(
                training_args.output_dir,
                normlized=model_args.normlized,
                model_args=model_args,
                sentence_pooling_method=model_args.sentence_pooling_method)
            trainer = BiTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=BiCollator(
                    tokenizer,
                    query_max_len=data_args.query_max_len,
                    passage_max_len=data_args.passage_max_len),
            )
            metrics = eval_test(trainer, data_args, tokenizer)
            for k, v in metrics.items():
                metrics[k] = round(v * 100, 4)

            result_file = os.path.join(data_args.result_dir,
                                       data_args.result_file + f'_best.json')
            Path(data_args.result_dir).mkdir(parents=True, exist_ok=True)
            with open(result_file, 'w') as f:
                model_args, data_args, training_args
                saved_dict = {
                    'test': metrics,
                    'args': {
                        'training_args': training_args.to_dict(),
                        'model_args': asdict(model_args),
                        'data_args': asdict(data_args),
                    },
                    'train': {
                        'clw_loss': np.mean(clw_losses),
                        'kl_loss': np.mean(kl_losses) if len(kl_losses) > 0 else 'Nan',
                    },
                    'dev': evalCallback.best_metrics
                }
                json.dump(saved_dict, f, indent=4)
            print('=' * 30)
            for x, y in metrics.items():
                print('{:5}: {:.2f} %'.format(x, y))
            print('=' * 30)
        else:
            ckpts = [model_args.model_name_or_path]
            if model_args.model_dir != '':
                all_ckpts = []
                for file in os.listdir(model_args.model_dir):
                    if file == 'runs':
                        continue
                    pth = os.path.join(model_args.model_dir, file)
                    if os.path.isdir(pth):
                        all_ckpts.append(pth)
                if all_ckpts:
                    ckpts = all_ckpts

            for ckpt in ckpts:
                logger.info(f"*** Prediction {ckpt} ***")
                model_args.model_name_or_path = ckpt
                model = BiEncoderModel.load(
                    ckpt,
                    normlized=model_args.normlized,
                    model_args=model_args,
                    sentence_pooling_method=model_args.sentence_pooling_method)
                step = ckpt.split('-')[-1]
                trainer = BiTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    data_collator=BiCollator(
                        tokenizer,
                        query_max_len=data_args.query_max_len,
                        passage_max_len=data_args.passage_max_len),
                )
                metrics = eval_test(trainer, data_args, tokenizer)
                for k, v in metrics.items():
                    metrics[k] = round(v * 100, 4)

                result_file = os.path.join(
                    data_args.result_dir,
                    data_args.result_file + f'_{step}.json')
                Path(data_args.result_dir).mkdir(parents=True, exist_ok=True)
                with open(result_file, 'w') as f:
                    model_args, data_args, training_args
                    saved_dict = {
                        'test': metrics,
                        'args': {
                            'training_args': training_args.to_dict(),
                            'model_args': asdict(model_args),
                            'data_args': asdict(data_args),
                        }
                    }
                    json.dump(saved_dict, f, indent=4)
                print('=' * 30)
                for x, y in metrics.items():
                    print('{:5}: {:.2f} %'.format(x, y))
                print('=' * 30)


class EpochEvalCallback(TrainerCallback):

    def __init__(self, trainer: Trainer, data_args: DataArguments,
                 tokenizer) -> None:
        super().__init__()
        self.trainer = trainer
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.best_metric = 0
        self.best_ckpt = ''
        self.best_metrics = {}

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        self.trainer.model.eval()
        with torch.no_grad():
            metrics = eval_test(self.trainer,
                                self.data_args,
                                self.tokenizer,
                                split='dev')
        logging.info(f"{metrics}")
        p1 = metrics['P@1']
        if p1 >= self.best_metric:
            self.best_metric = p1
            self.best_ckpt = state.global_step
            self.best_metrics = metrics
            # 直接保存到顶层output_dir
            self.trainer.save_model()

        return super().on_epoch_end(args, state, control, **kwargs)


if __name__ == "__main__":
    main()
