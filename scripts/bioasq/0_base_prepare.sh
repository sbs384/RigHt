#!/bin/bash

# 为每个fold训练基线双塔模型，准备导出难负例

dataset=$1 #'9b'
query_max_len=24
passage_max_len=168
sentence_pooling_method='cls'
batch_size=32
train_group_size=1
gradient_accumulation_steps=1
lr=5e-5
epoch=10
SEED=42
sample_neg_from_topk=20
ptm="dmis-lab/biobert-base-cased-v1.1"

# 训练每个fold的基线模型
# 不需要额外负例
for fold in 1 2 3 4 5
do
model_name_or_path=${ptm}
data_root_dir="data/tokenized_data/bioasq/${dataset}"
result_dir="./results/bi_encoder/baseline/${dataset}/${fold}"
output_dir="./output/bi_encoder/baseline/${dataset}/${fold}"
fold_train_dir="${data_root_dir}/fold_${fold}"

result_file="bs_${batch_size}_seed_${SEED}"
python src/bi_encoder/run.py \
    --output_dir ${output_dir} \
    --model_name_or_path ${model_name_or_path} \
    --model_dir ${output_dir} \
    --tokenizer_name ${ptm} \
    --do_train \
    --corpus_file ${fold_train_dir}/train_corpus \
    --train_query_file ${fold_train_dir}/train_query \
    --train_qrels ${fold_train_dir}/train_qrels.txt \
    --query_max_len ${query_max_len} \
    --passage_max_len ${passage_max_len} \
    --per_device_train_batch_size ${batch_size} \
    --train_group_size ${train_group_size} \
    --sample_neg_from_topk ${sample_neg_from_topk} \
    --learning_rate 5e-5 \
    --num_train_epochs ${epoch} \
    --dataloader_num_workers 6 \
    --optim 'adamw_torch' \
    --save_strategy 'epoch' \
    --sentence_pooling_method ${sentence_pooling_method} \
    --warmup_ratio 0.1 \
    --save_total_limit 1 \
    --seed ${SEED} \
    --logging_strategy 'epoch' \
    --result_dir ${result_dir} \
    --result_file ${result_file} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --overwrite_output_dir \
    --test_corpus_file ${data_root_dir}/test_corpus \
    --test_query_file ${data_root_dir}/test_query \
    --test_qrels ${data_root_dir}/test_qrels.txt \
    --do_predict 
done

    # --fp16 \

    # --adv_training \
    # --adv_norm 0.001

    # --neg_file ${data_root_dir}/hard_negs.txt \
    # --neg_source ${neg_source} \


