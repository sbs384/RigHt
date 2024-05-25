#!/bin/bash

dataset='SQuAD'
output_dir="./tmp"
model_name_or_path="./output/bi_encoder/baseline/${dataset}/checkpoint-24630"
data_root_dir="./data/tokenized_data/${dataset}"
prediction_save_path="./output/results/bi_encoder/${dataset}"
neg_file="${prediction_save_path}/hard_negs.txt"
# 推理时使用最大长度，与reqa一致
query_max_len=512
passage_max_len=512

# 保存训练集 resp
python src/bi_encoder/run.py \
    --output_dir ${output_dir} \
    --model_name_or_path ${model_name_or_path}  \
    --test_corpus_file ${data_root_dir}/train_corpus \
    --test_query_file ${data_root_dir}/train_query \
    --test_qrels ${data_root_dir}/train_qrels.txt \
    --query_max_len ${query_max_len} \
    --passage_max_len ${passage_max_len} \
    --do_predict \
    --prediction_save_path ${prediction_save_path} \
    --per_device_eval_batch_size 256 \
    --dataloader_num_workers 6 \
    --eval_accumulation_steps 100 \
    --tokenizer_name 'bert-base-uncased' \
    --save_reps_only

# rank the passages
python examples/retriever/SQuAD/test.py \
    --query_reps_path ${prediction_save_path}/query_reps \
    --passage_reps_path ${prediction_save_path}/passage_reps \
    --ranking_file  ${prediction_save_path}/train_ranking.txt \
    --depth 200 \
    --use_gpu 

# # delete the positive passages in top-k results
python examples/retriever/SQuAD/generate_hard_negtives.py \
    --ranking_file  ${prediction_save_path}/train_ranking.txt \
    --qrels_file ${data_root_dir}/train_qrels.txt \
    --output_neg_file ${neg_file}
        
