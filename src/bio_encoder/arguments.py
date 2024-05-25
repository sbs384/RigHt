import os
from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    teacher_model_name_or_path: str = field(
        default=None, metadata={"help": "Path to teacher model"}
    )
    model_dir: str = field(default='')
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )
    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)

    sentence_pooling_method: str = field(default='cls')
    normlized: bool = field(default=False)
    adv_training: bool = field(default=False)
    adv_norm: float = field(default=0.001)
    
    # agg option
    agg_dim: int = field(default=0)
    semi_aggregate: bool = field(default=False)
    skip_mlm: bool = field(default=True)

@dataclass
class DataArguments:
    curriculum_learning: bool= field(default=False,metadata={"help":"enable curriculum learning"})
    curriculum_neg_max: int=field(default=200, metadata={"help":"init rank for sampling neg"})
    curriculum_neg_min: int=field(default=200, metadata={"help":"final rank for sampling neg"})
    sample_neg_from_topk: int = field(
        default=200, metadata={"help": "sample negatives from top-k"}
    )
    teacher_score_files: str = field(
        default=None, metadata={"help": "Path to score_file for distillation"}
    )

    corpus_file: str = field(
        default=None, metadata={"help": "Path to corpus"}
    )
    corpus_id_file: str = field(
        default=None, metadata={"help": "Path to corpus"}
    )
    dev_corpus_file: str = field(
        default=None, metadata={"help": "Path to corpus"}
    )
    dev_corpus_id_file: str = field(
        default=None, metadata={"help": "Path to corpus"}
    )

    test_corpus_file: str = field(
        default=None, metadata={"help": "Path to test corpus"}
    )
    test_corpus_id_file: str = field(
        default=None, metadata={"help": "Path to test corpus"}
    )
    train_query_file: str = field(
        default=None, metadata={"help": "Path to query data"}
    )
    train_query_id_file: str = field(
        default=None, metadata={"help": "Path to query data"}
    )
    train_qrels: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    
    dev_query_file: str = field(
        default=None, metadata={"help": "Path to query data"}
    )
    dev_query_id_file: str = field(
        default=None, metadata={"help": "Path to query data"}
    )
    dev_qrels: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    test_qrels: str = field(
        default=None, metadata={"help": "Path to test rels"}
    )
    neg_file: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    test_query_file: str = field(
        default=None, metadata={"help": "Path to negative"}
    )
    test_query_id_file: str = field(
        default=None, metadata={"help": "Path to query data"}
    )

    prediction_save_path: str = field(
        default=None, metadata={"help": "Path to save prediction"}
    )

    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=24,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=168,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    extra_neg: bool = field(default=False)
    
    neg_source: str = field(default='file')
    
    export_neg_dir: str = field(default='')

    import_neg_dir: str = field(default='')

    save_reps_only: bool = field(default=False)

    result_dir: str = field(default='./results')

    result_file: str = field(default='result', metadata={
                             'help': 'json file to save results'})

    def __post_init__(self):
        if self.corpus_file and not self.corpus_id_file:
            if not os.path.exists(os.path.join(self.corpus_file, 'mapping_id.txt')):
                raise FileNotFoundError(
                    f'There is no mapping_id.txt in {self.corpus_file}')
            self.corpus_id_file = os.path.join(
                self.corpus_file, 'mapping_id.txt')
        if self.dev_corpus_file and not self.dev_corpus_id_file:
            if not os.path.exists(os.path.join(self.dev_corpus_file, 'mapping_id.txt')):
                raise FileNotFoundError(
                    f'There is no mapping_id.txt in {self.dev_corpus_file}')
            self.dev_corpus_id_file = os.path.join(
                self.dev_corpus_file, 'mapping_id.txt')
        if self.test_corpus_file and not self.test_corpus_id_file:
            if not os.path.exists(os.path.join(self.test_corpus_file, 'mapping_id.txt')):
                raise FileNotFoundError(
                    f'There is no mapping_id.txt in {self.test_corpus_file}')
            self.test_corpus_id_file = os.path.join(
                self.test_corpus_file, 'mapping_id.txt')

        if self.train_query_file and not self.train_query_id_file:
            if not os.path.exists(os.path.join(self.train_query_file, 'mapping_id.txt')):
                raise FileNotFoundError(
                    f'There is no mapping_id.txt in {self.train_query_file}')
            self.train_query_id_file = os.path.join(
                self.train_query_file, 'mapping_id.txt')
        if self.dev_query_file and not self.dev_query_id_file:
            if not os.path.exists(os.path.join(self.dev_query_file, 'mapping_id.txt')):
                raise FileNotFoundError(
                    f'There is no mapping_id.txt in {self.dev_query_file}')
            self.dev_query_id_file = os.path.join(
                self.dev_query_file, 'mapping_id.txt')
        if self.test_query_file and not self.test_query_id_file:
            if not os.path.exists(os.path.join(self.test_query_file, 'mapping_id.txt')):
                raise FileNotFoundError(
                    f'There is no mapping_id.txt in {self.test_query_file}')
            self.test_query_id_file = os.path.join(
                self.test_query_file, 'mapping_id.txt')


@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_x_device: bool = field(default=False, metadata={
                                     "help": "share negatives across devices"})
    temperature: Optional[float] = field(default=1.0)
    contrastive_loss_weight: Optional[float] = field(default=None)
    kl_loss_weight: Optional[float] = field(default=None)
    balance_loss: Optional[bool] = field(default=False)

    def __post_init__(self):
        if self.balance_loss:
            if self.kl_loss_weight != None and self.contrastive_loss_weight != None:
                raise ValueError(
                    "kl_loss_weight and contrastive_loss_weight are both set! "
                )
            if self.kl_loss_weight != None:
                self.contrastive_loss_weight = 1.0 - self.kl_loss_weight
            else:
                self.kl_loss_weight = 1.0 - self.contrastive_loss_weight
        if self.contrastive_loss_weight == None:
            self.contrastive_loss_weight = 0.01
        if self.kl_loss_weight == None:
            self.kl_loss_weight = 0
