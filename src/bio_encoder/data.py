import collections
from copy import deepcopy
import logging
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional

import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding
from pathlib import Path
from .arguments import DataArguments
import sys


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
    for line in open(neg_file, encoding='utf-8'):
        line = line.strip('\n').split('\t')
        qid = line[0]
        neg = line[1].split(',')
        neg_data[qid].extend(neg)
    return neg_data


def read_teacher_score(score_files):
    teacher_score = collections.defaultdict(dict)
    for file in score_files.split(','):
        if file and not os.path.exists(file):
            logging.info(
                f"There is no score file:{file}, please check the parameters")
            sys.exit(0)
        for line in open(file):
            qid, did, score = line.strip().split()
            score = float(score.strip('[]'))
            teacher_score[qid][did] = score
    return teacher_score


def generate_random_neg(qids, pids, k=30):
    qid_negatives = {}
    for q in qids:
        negs = random.sample(pids, k)
        qid_negatives[q] = negs
    return qid_negatives


def curriculum_update(args: DataArguments, epoch: int):
    mi = 0
    mx = max(args.curriculum_neg_max - 10 * epoch, args.curriculum_neg_min)
    logging.info(f"sample neg from [{mi}:{mx})")
    return mi, mx


class TrainDatasetForBiE(Dataset):

    def __init__(self, args: DataArguments, tokenizer: PreTrainedTokenizer):
        self.corpus_dataset = datasets.Dataset.load_from_disk(args.corpus_file)
        self.query_dataset = datasets.Dataset.load_from_disk(
            args.train_query_file)
        self.train_qrels = read_train_file(args.train_qrels)
        self.corpus_id = read_mapping_id(args.corpus_id_file)
        self.query_id = read_mapping_id(args.train_query_id_file)

        if args.neg_file:
            self.train_negative = read_neg_file(args.neg_file)
        else:
            self.train_negative = generate_random_neg(
                list(self.query_id.keys()), list(self.corpus_id.keys()))

        self.teacher_score = None
        if args.teacher_score_files is not None:
            self.teacher_score = read_teacher_score(args.teacher_score_files)

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.train_qrels)
        self.epoch = 0
        self.sample_neg_from_topk_min = 0
        self.sample_neg_from_topk_max = args.sample_neg_from_topk

    def __len__(self):
        return self.total_len

    def update_negs(self, hard_negs):

        if hard_negs is None:
            neg_file = os.path.join(self.args.import_neg_dir,
                                    f'epoch_{self.epoch}.txt')
            logging.info(f'load hard neg from {neg_file}')
            hard_negs = read_neg_file(neg_file)
        self.train_negative = hard_negs
        if self.args.import_neg_dir == '' and self.args.export_neg_dir != '':
            Path(self.args.export_neg_dir).mkdir(parents=True, exist_ok=True)
            with open(
                    os.path.join(self.args.export_neg_dir,
                                 f'epoch_{self.epoch}.txt'), 'w') as f:
                for qid, negs in hard_negs.items():
                    line = f'{qid}\t{",".join(negs)}\n'
                    f.write(line)
        self.epoch += 1
        if self.args.curriculum_learning:
            mi, mx = curriculum_update(self.args, self.epoch)
            self.sample_neg_from_topk_min = mi
            self.sample_neg_from_topk_max = mx

    def create_query_example(self, id: Any):
        item = self.tokenizer.encode_plus(
            self.query_dataset[self.query_id[id]]['input_ids'],
            truncation='only_first',
            max_length=self.args.query_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def create_passage_example(self, id: Any):
        item = self.tokenizer.encode_plus(
            self.corpus_dataset[self.corpus_id[id]]['input_ids'],
            truncation='only_first',
            max_length=self.args.passage_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __getitem__(
        self, item
    ) -> Tuple[BatchEncoding, List[BatchEncoding], Optional[List[int]]]:
        group = self.train_qrels[item]

        qid = group[0]
        query = self.create_query_example(qid)

        teacher_scores = None
        passages = []
        pos_id = random.choice(group[1])
        passages.append(self.create_passage_example(pos_id))
        if self.teacher_score:
            teacher_scores = []
            score = self.teacher_score[qid][pos_id]
            teacher_scores.append(score)

        if self.args.extra_neg:
            query_negs = self.train_negative[qid][
                self.sample_neg_from_topk_min:self.sample_neg_from_topk_max]
            if len(query_negs) < self.args.train_group_size - 1:
                negs = random.sample(self.corpus_id.keys(),
                                     k=self.args.train_group_size - 1 -
                                     len(query_negs))
                negs.extend(query_negs)
            else:
                negs = random.sample(query_negs,
                                     k=self.args.train_group_size - 1)
            for id in negs:
                passages.append(self.create_passage_example(id))
                if self.teacher_score:
                    if id not in self.teacher_score[qid]:
                        score = -1e10  # 默认兜底
                        # print(f'error pair, qid={qid}, pid={id}')
                    else:
                        score = self.teacher_score[qid][id]
                    teacher_scores.append(score)

        return query, passages, teacher_scores


class DevDatasetForBiE(TrainDatasetForBiE):

    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer,
    ):
        args = deepcopy(args)
        args.corpus_file = args.dev_corpus_file
        args.corpus_id_file = args.dev_corpus_id_file
        args.train_query_file = args.dev_query_file
        args.train_query_id_file = args.dev_query_id_file
        args.neg_file = None
        args.extra_neg = False
        args.train_qrels = args.dev_qrels
        args.teacher_score_files = None
        super().__init__(args, tokenizer)


class PredictionDataset(Dataset):

    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 max_len=128):
        self.encode_data = datasets.Dataset.load_from_disk(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            self.encode_data[item]['input_ids'],
            truncation='only_first',
            max_length=self.max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item


@dataclass
class BiCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 24
    passage_max_len: int = 168

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]
        teacher_score = [f[2] for f in features]
        if teacher_score[0] is None:
            teacher_score = None
        else:
            teacher_score = torch.FloatTensor(teacher_score)

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer.pad(
            query,
            padding='longest',
            # max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            passage,
            padding='longest',
            # max_length=self.passage_max_len,
            return_tensors="pt",
        )

        return {
            "query": q_collated,
            "passage": d_collated,
            "teacher_score": teacher_score
        }


@dataclass
class PredictionCollator(DataCollatorWithPadding):
    is_query: bool = True

    def __call__(self, features):
        if self.is_query:
            return {"query": super().__call__(features), "passage": None}
        else:
            return {"query": None, "passage": super().__call__(features)}
