import numpy as np
import torch
import sys

from collections import Counter
import logging
import os
from pathlib import Path
from bi_encoder.arguments import DataArguments, RetrieverTrainingArguments as TrainingArguments
from bi_encoder.data import PredictionCollator, PredictionDataset, TrainDatasetForBiE
from bi_encoder.trainer import BiTrainer
from collections import defaultdict
from transformers.trainer import TrainerCallback, TrainerState, TrainerControl


class HardMiner:
    def __init__(self, model, training_args: TrainingArguments, data_args: DataArguments, tokenizer, batch_size=128) -> None:
        self.model = model
        self.query_collator = PredictionCollator(
            tokenizer=tokenizer, is_query=True)
        self.corpus_collator = PredictionCollator(
            tokenizer=tokenizer, is_query=False)

        self.query_dataset = PredictionDataset(
            data_path=data_args.train_query_file, tokenizer=tokenizer,
            max_len=data_args.passage_max_len,
        )
        self.corpus_dataset = PredictionDataset(
            data_path=data_args.corpus_file, tokenizer=tokenizer,
            max_len=data_args.passage_max_len,
        )

        self.trainer = BiTrainer(
            model=model,
            args=training_args,
        )

        self.train_qrels = read_train_file(data_args.train_qrels)
        self.corpus_id = read_mapping_id(data_args.corpus_id_file)
        self.query_id = read_mapping_id(data_args.train_query_id_file)

        self.batch_size = batch_size

    def encode(self):
        self.trainer.data_collator = self.query_collator
        query_embeddings = self.trainer.predict(
            test_dataset=self.query_dataset).predictions

        self.trainer.data_collator = self.corpus_collator
        corpus_embeddings = self.trainer.predict(
            test_dataset=self.corpus_dataset).predictions

        query_embeddings = torch.tensor(query_embeddings, device='cuda')
        corpus_embeddings = torch.tensor(corpus_embeddings, device='cuda')
        return query_embeddings, corpus_embeddings

    def gen_hard(self):
        self.model.eval()
        with torch.no_grad():
            query_embeddings, corpus_embeddings = self.encode()
            hard_indices = []
            hard_negs = defaultdict(list)
            for st in range(0, len(query_embeddings), self.batch_size):
                ed = st+self.batch_size
                query = query_embeddings[st:ed]
                logits = self.model.compute_similarity(query, corpus_embeddings)
                score, indices = logits.topk(k=256, dim=-1)
                hard_indices += indices.cpu().numpy().tolist()
            for offset, ind in enumerate(hard_indices):
                qid = self.query_id[offset]
                pos = self.train_qrels[qid]
                negs = [self.corpus_id[i]
                        for i in ind if self.corpus_id[i] not in pos]
                hard_negs[qid].extend(negs)
        return hard_negs


class EpochCallback(TrainerCallback):
    def __init__(self, train_dataset: TrainDatasetForBiE, hard_miner: HardMiner) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.hard_miner = hard_miner

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.hard_miner is None:
            negs = None
        else:
            negs = self.hard_miner.gen_hard()
        self.train_dataset.update_negs(negs)
        return super().on_epoch_begin(args, state, control, **kwargs)


def read_mapping_id(id_file):
    ids = []
    for line in open(id_file, encoding='utf-8'):
        id, offset = line.strip().split('\t')
        ids.append(id)
    return ids


def read_train_file(train_file):
    q2p_map = defaultdict(set)
    for line in open(train_file, encoding='utf-8'):
        line = line.strip('\n').split('\t')
        qid = line[0]
        pos = line[1].split(',')
        for p in pos:
            q2p_map[qid].add(p)
    return q2p_map
