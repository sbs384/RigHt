#!/bin/bash
dataset='SQuAD'
model_name_or_path="bert-base-uncased"
data_root_dir='data/tokenized_data/SQuAD'
query_max_len=24
passage_max_len=168

batch_size=32
train_group_size=1
lr=5e-5
epoch=10
result_dir="./results/bi_encoder/baseline/${dataset}"
sample_neg_from_topk=80
output_dir="./output/bi_encoder/baseline/${dataset}"
result_file="epoch_${epoch}_lr_${lr}"
neg_source='ance'

    # --overwrite_output_dir \
    # 

python src/bi_encoder/run.py \
    --output_dir ${output_dir} \
    --model_name_or_path ${model_name_or_path} \
    --model_dir ${output_dir} \
    --tokenizer_name 'bert-base-uncased' \
    --do_predict  \
    --do_train \
    --corpus_file ${data_root_dir}/train_corpus \
    --train_query_file ${data_root_dir}/train_query \
    --train_qrels ${data_root_dir}/train_qrels.txt \
    --query_max_len ${query_max_len} \
    --passage_max_len ${passage_max_len} \
    --test_corpus_file ${data_root_dir}/test_corpus \
    --test_query_file ${data_root_dir}/test_query \
    --test_qrels ${data_root_dir}/test_qrels.txt \
    --per_device_train_batch_size ${batch_size} \
    --train_group_size ${train_group_size} \
    --sample_neg_from_topk ${sample_neg_from_topk} \
    --learning_rate ${lr} \
    --num_train_epochs ${epoch} \
    --dataloader_num_workers 6 \
    --optim 'adamw_torch' \
    --save_strategy 'epoch' \
    --warmup_ratio 0.1 \
    --save_total_limit 5 \
    --seed 42 \
    --logging_strategy 'epoch' \
    --result_dir ${result_dir} \
    --result_file ${result_file} \
    --fp16 \
    --overwrite_output_dir


    # --neg_source ${neg_source} \