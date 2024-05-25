#!/bin/bash
dataset='SQuAD'
model_name_or_path="bert-base-uncased"
data_root_dir='./data/tokenized_data/SQuAD'
max_len=200
train_group_size=32
batch_size=2
gradient_accumulation_steps=4
sample_neg_from_topk=120
num_train_epochs=5
output_dir="./output/cross_encoder/auto/${dataset}"
per_device_eval_batch_size=256

python src/cross_encoder/run.py \
    --output_dir ${output_dir} \
    --model_name_or_path ${model_name_or_path} \
    --fp16 \
    --do_train \
    --corpus_file ${data_root_dir}/train_corpus \
    --train_query_file ${data_root_dir}/train_query \
    --train_qrels ${data_root_dir}/train_qrels.txt \
    --dev_corpus_file ${data_root_dir}/dev_corpus \
    --dev_query_file ${data_root_dir}/dev_query \
    --dev_qrels ${data_root_dir}/dev_qrels.txt \
    --neg_file ${data_root_dir}/hard_negs.txt \
    --max_len ${max_len} \
    --per_device_train_batch_size ${batch_size} \
    --train_group_size ${train_group_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --learning_rate 1e-5 \
    --num_train_epochs ${num_train_epochs} \
    --dataloader_num_workers 6 \
    --optim 'adamw_torch' \
    --logging_steps 5000 \
    --save_strategy 'epoch' \
    --seed 42 \
    --overwrite_output_dir \
    --evaluation_strategy 'epoch' \
    --sample_neg_from_topk ${sample_neg_from_topk} \
    --evaluation_strategy 'epoch' \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model 'eval_accuracy' \
    --sample_neg_from_topk ${sample_neg_from_topk}



prediction_save_path="./output/results/${dataset}"
model_name_or_path=${output_dir}
# model_name_or_path="./output/cross_encoder/baseline/${dataset}/checkpoint-29565"
prediction_topk=200
prediction_topk_min=0
# prediction for kd

python src/cross_encoder/run.py \
    --output_dir ./tmp \
    --model_name_or_path ${model_name_or_path} \
    --fp16  \
    --corpus_file ${data_root_dir}/train_corpus \
    --max_len 200 \
    --do_predict  \
    --test_query_file ${data_root_dir}/train_query \
    --test_file ${data_root_dir}/train_qrels.txt \
    --prediction_save_path ${prediction_save_path}/train_qrels_score.txt \
    --dataloader_num_workers 6 \
    --tokenizer_name 'bert-base-uncased'

python src/cross_encoder/run.py \
    --output_dir ./tmp \
    --model_name_or_path ${model_name_or_path} \
    --fp16  \
    --corpus_file ${data_root_dir}/train_corpus \
    --max_len 200 \
    --do_predict  \
    --test_query_file ${data_root_dir}/train_query \
    --test_file ${prediction_save_path}/train_ranking.txt \
    --prediction_save_path ${prediction_save_path}/train_rerank_score_200.txt \
    --dataloader_num_workers 6 \
    --tokenizer_name 'bert-base-uncased' \
    --prediction_topk ${prediction_topk} \
    --prediction_topk_min ${prediction_topk_min} \
    --per_device_eval_batch_size ${per_device_eval_batch_size}