import collections
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import datasets
import torch
from cross_encoder.arguments import DataArguments, CETrainingArguments
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding
from copy import deepcopy


def read_mapping_id(id_file):
    id_dict = {}
    for line in open(id_file, encoding='utf-8'):
        id, offset = line.strip().split('\t')
        id_dict[id] = int(offset)
    return id_dict


def read_train_file(train_file):
    train_data = []
    for line in open(train_file, encoding='utf-8'):
        line = line.strip('\n').split('\t')
        qid = line[0]
        pos = line[1].split(',')
        train_data.append((qid, pos))
    return train_data


def read_neg_file(neg_file):
    neg_data = collections.defaultdict(list)
    if neg_file == '' or neg_file is None:
        return neg_data
    for line in open(neg_file, encoding='utf-8'):
        line = line.strip('\n').split('\t')
        qid = line[0]
        neg = line[1].split(',')
        neg_data[qid].extend(neg)
    return neg_data


def read_test_file(test_file, min_topk, prediction_topk):
    test_data = []
    for line in open(test_file, encoding='utf-8'):
        line = line.strip('\n').split('\t')
        if len(line) >= 3:
            rank = int(line[2])
            if min_topk <= rank <= prediction_topk:
                test_data.append((line[0], line[1]))
        else:
            test_data.append((line[0], line[1]))
    return test_data


def read_neg_from_ranking_file():
    pass


class TrainDatasetForCE(Dataset):

    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer,
        train_args: CETrainingArguments = None,
    ):
        self.corpus_dataset = datasets.Dataset.load_from_disk(args.corpus_file)
        self.query_dataset = datasets.Dataset.load_from_disk(
            args.train_query_file)
        self.train_qrels = read_train_file(args.train_qrels)
        self.train_negative = read_neg_file(args.neg_file)
        self.corpus_id = read_mapping_id(args.corpus_id_file)
        self.query_id = read_mapping_id(args.train_query_id_file)

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.train_qrels)
        self.train_args = train_args
        self.epoch = 0

    def update_negs(self):
        neg_file = os.path.join(self.args.neg_dir, f'epoch_{self.epoch}.txt')
        if os.path.exists(neg_file):
            print('loading neg from', neg_file)
            self.train_negative = read_neg_file(neg_file)
        self.epoch += 1

    def create_one_example(self, qry_encoding: List[int],
                           doc_encoding: List[int]):
        item = self.tokenizer.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation='only_second',
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        group = self.train_qrels[item]
        examples = []
        qry = self.query_dataset[self.query_id[group[0]]]
        pos = self.corpus_dataset[self.corpus_id[random.choice(group[1])]]
        examples.append([qry['input_ids'], pos['input_ids']])
        query_negs = self.train_negative[
            group[0]][:self.args.sample_neg_from_topk]

        if len(query_negs) < self.args.train_group_size - 1:
            negs = random.sample(self.corpus_id.keys(),
                                 k=self.args.train_group_size - 1 -
                                 len(query_negs))
            negs.extend(query_negs)
        else:
            negs = random.sample(query_negs, k=self.args.train_group_size - 1)

        for neg_entry in negs:
            neg_psg = self.corpus_dataset[self.corpus_id[neg_entry]]
            examples.append((qry['input_ids'], neg_psg['input_ids']))

        batch_data = []
        for e in examples:
            batch_data.append(self.create_one_example(*e))
        return batch_data


class DevDatasetForCE(TrainDatasetForCE):

    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer,
        train_args: CETrainingArguments = None,
    ):
        args = deepcopy(args)
        args.corpus_file = args.dev_corpus_file
        args.corpus_id_file = args.dev_orpus_id_file
        args.train_query_file = args.dev_query_file
        args.train_query_id_file = args.dev_query_id_file
        args.neg_file = args.dev_neg_file
        args.train_qrels = args.dev_qrels
        super().__init__(args, tokenizer, train_args)
        # 设置固定的负例，避免每次验证时负例不同
        if args.neg_file is None:
            corpus = set(self.corpus_id.keys())
            qrels = collections.defaultdict(set)
            for q, p in self.train_qrels:
                qrels[q]|=set(p)
            for q in self.query_id:
                self.train_negative[q] = random.sample(
                    corpus - qrels[q], k=args.train_group_size - 1)
        else:
            for q, negs in self.train_negative.items():
                self.train_negative[q] = negs[:args.train_group_size - 1]


class PredictionDatasetForCE(Dataset):

    def __init__(self,
                 args: DataArguments,
                 tokenizer: PreTrainedTokenizer,
                 max_len=128):
        self.corpus_dataset = datasets.Dataset.load_from_disk(args.corpus_file)
        self.query_dataset = datasets.Dataset.load_from_disk(
            args.test_query_file)
        self.corpus_id = read_mapping_id(args.corpus_id_file)
        self.query_id = read_mapping_id(args.test_query_id_file)

        self.test_data = read_test_file(args.test_file,
                                        args.prediction_topk_min,
                                        args.prediction_topk)

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.test_data)
        self.max_len = max_len

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, item):
        qid, pid = self.test_data[item]
        qry = self.query_dataset[self.query_id[qid]]['input_ids']
        psg = self.corpus_dataset[self.corpus_id[pid]]['input_ids']

        case = self.tokenizer.encode_plus(
            qry,
            psg,
            truncation='only_second',
            max_length=self.max_len,
            padding=False,
        )
        return case


@dataclass
class GroupCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(
            self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(features[0], list):
            features = sum(features, [])
        return super().__call__(features)
