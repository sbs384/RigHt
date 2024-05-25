import numpy as np
import torch
import sys

from collections import Counter
import logging
import os
from pathlib import Path
from bi_encoder.arguments import DataArguments
from bi_encoder.data import PredictionCollator, PredictionDataset
from transformers import Trainer
from functools import lru_cache


@lru_cache(None)
def read_id(mapping_id_file):
    ids = []
    for line in open(mapping_id_file):
        id, offset = line.strip().split('\t')
        ids.append(int(id))
    return np.array(ids)


@lru_cache(None)
def get_prediction_dataset(data_path, tokenizer, max_len):
    return PredictionDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_len=max_len,
    )


def test_encode(trainer: Trainer,
                data_args: DataArguments,
                tokenizer,
                split='test'):
    corpus_file = data_args.test_corpus_file if split == 'test' else data_args.dev_corpus_file
    query_file = data_args.test_query_file if split == 'test' else data_args.dev_query_file
    query_len = 512 if split == 'test' else data_args.query_max_len
    candidate_len = 512 if split == 'test' else data_args.passage_max_len
    # 保存原本的collator
    data_collator = trainer.data_collator
    # 测试时不做截断处理，最大长度512
    # candidate encoding
    trainer.data_collator = PredictionCollator(tokenizer=tokenizer,
                                               is_query=False)
    corpus_dataset = get_prediction_dataset(
        data_path=corpus_file,
        tokenizer=tokenizer,
        max_len=candidate_len,
    )
    candidate_embeddings = trainer.predict(
        test_dataset=corpus_dataset).predictions
    # query encoding
    trainer.data_collator = PredictionCollator(tokenizer=tokenizer,
                                               is_query=True)
    query_dataset = get_prediction_dataset(
        data_path=query_file,
        tokenizer=tokenizer,
        max_len=query_len,
    )
    query_embeddings = trainer.predict(test_dataset=query_dataset).predictions
    trainer.data_collator = data_collator
    if split == 'test' and data_args.save_reps_only:
        logging.info("*** Saving Corpus Prediction ***")
        passage_path = os.path.join(data_args.prediction_save_path,
                                    'passage_reps')
        Path(passage_path).mkdir(parents=True, exist_ok=True)

        np.save(os.path.join(passage_path, 'passage.npy'),
                candidate_embeddings)
        with open(os.path.join(passage_path, 'offset2passageid.txt'),
                  "w") as writer:
            for line in open(data_args.test_corpus_id_file):
                cid, offset = line.strip().split('\t')
                writer.write(f'{offset}\t{cid}\t\n')

        logging.info("*** Saving Query Prediction ***")
        query_path = os.path.join(data_args.prediction_save_path, 'query_reps')
        Path(query_path).mkdir(parents=True, exist_ok=True)

        np.save(os.path.join(query_path, 'query.npy'), query_embeddings)
        with open(os.path.join(query_path, 'offset2queryid.txt'),
                  "w") as writer:
            for line in open(data_args.test_query_id_file):
                cid, offset = line.strip().split('\t')
                writer.write(f'{offset}\t{cid}\t\n')
        sys.exit(0)

    return query_embeddings, candidate_embeddings


@torch.no_grad()
def eval_test(trainer: Trainer,
              data_args: DataArguments,
              tokenizer,
              split='test'):
    trainer.model.eval()
    corpus_id_file = data_args.test_corpus_id_file if split == 'test' else data_args.dev_corpus_id_file
    query_id_file = data_args.test_query_id_file if split == 'test' else data_args.dev_query_id_file
    qrels = data_args.test_qrels if split == 'test' else data_args.dev_qrels

    query_embeddings, candidate_embeddings = test_encode(trainer,
                                                         data_args,
                                                         tokenizer,
                                                         split=split)

    candidate_embeddings = torch.tensor(candidate_embeddings, device='cuda')
    query_embeddings = torch.tensor(query_embeddings, device='cuda')
    # TODO: batch evaluation
    logits = query_embeddings.matmul(candidate_embeddings.T).cpu().numpy()

    p_lookup = read_id(corpus_id_file)
    q_lookup = read_id(query_id_file)
    ranking = {}
    for qid, q_logits in zip(q_lookup, logits):
        rank = np.argsort(-q_logits)
        rank = p_lookup[rank]
        ranking[qid] = rank

    rels = load_reference(qrels)
    metrics = compute_metrics(rels, ranking)
    return metrics


def eval_logits(logits, query_id_file, corpus_id_file, qrels):
    p_lookup = read_id(corpus_id_file)
    q_lookup = read_id(query_id_file)
    ranking = {}
    for qid, q_logits in zip(q_lookup, logits):
        rank = np.argsort(-q_logits)
        rank = p_lookup[rank]
        ranking[qid] = rank

    rels = load_reference(qrels)
    metrics = compute_metrics(rels, ranking)
    return metrics


def compute_metrics(qids_to_relevant_passageids,
                    qids_to_ranked_candidate_passages):
    """Compute MRR metric
    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    metrics = {}
    ranking = []
    # recall
    for k in [5, 10]:
        Recall = 0
        for qid in qids_to_ranked_candidate_passages:
            if qid in qids_to_relevant_passageids:
                ranking.append(0)
                target_pid = qids_to_relevant_passageids[qid]
                candidate_pid = qids_to_ranked_candidate_passages[qid]
                Recall += (len(set(target_pid) & set(candidate_pid[:k])) /
                           len(target_pid))
        if len(ranking) == 0:
            raise IOError(
                "No matching QIDs found. Are you sure you are scoring the evaluation set?"
            )

        Recall = Recall / len(qids_to_relevant_passageids)
        metrics[f'R@{k}'] = Recall

    # precision
    ranking = []
    for k in [1]:
        Precision = 0
        for qid in qids_to_ranked_candidate_passages:
            if qid in qids_to_relevant_passageids:
                ranking.append(0)
                target_pid = qids_to_relevant_passageids[qid]
                candidate_pid = qids_to_ranked_candidate_passages[qid]
                Precision += (len(set(target_pid) & set(candidate_pid[:k])))
        if len(ranking) == 0:
            raise IOError(
                "No matching QIDs found. Are you sure you are scoring the evaluation set?"
            )
        Precision = Precision / len(qids_to_relevant_passageids) / k
        metrics[f'P@{k}'] = Precision

    MRR = 0
    ranking = []
    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_relevant_passageids:
            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            for i in range(0, len(candidate_pid)):
                if candidate_pid[i] in target_pid:
                    MRR += 1 / (i + 1)
                    ranking.pop()
                    ranking.append(i + 1)
                    break
    if len(ranking) == 0:
        raise IOError(
            "No matching QIDs found. Are you sure you are scoring the evaluation set?"
        )

    MRR = MRR / len(qids_to_relevant_passageids)
    metrics[f'MRR'] = MRR
    return metrics


@lru_cache(None)
def load_reference(file):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
    """
    qids_to_relevant_passageids = {}
    with open(file) as f:
        for l in f:
            try:
                l = l.strip().split('\t')
                qid = int(l[0])
                if qid in qids_to_relevant_passageids:
                    pass
                else:
                    qids_to_relevant_passageids[qid] = []
                qids_to_relevant_passageids[qid].append(int(l[1]))
            except:
                raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_passageids
