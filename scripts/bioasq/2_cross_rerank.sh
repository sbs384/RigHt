#!/bin/bash
# 利用难负例训练单塔模型

ptm="/home/zhaobo/hf_models/biobert_base_cased_v1.1"
dataset=$1 #'9b'
data_root_dir="data/tokenized_data/bioasq/${dataset}"
sentence_pooling_method='cls'
model_name_or_path=${ptm}

max_len=200
train_group_size=32
batch_size=2
gradient_accumulation_steps=8
sample_neg_from_topk=200
num_train_epochs=5
per_device_eval_batch_size=256
SEED=42
# 日志步数，数据少可以调低一点
logging_steps=2000
for fold in 1 2 3 4 5
do
    output_dir="./output/cross_encoder/baseline/${dataset}/${fold}/b${batch_size}_e${num_train_epochs}_g${train_group_size}"
    fold_train_dir="${data_root_dir}/fold_${fold}"
    # neg_dir="${fold_train_dir}/hard_negs"

    python src/cross_encoder/run.py \
        --output_dir ${output_dir} \
        --model_name_or_path ${model_name_or_path} \
        --fp16 \
        --do_train \
        --corpus_file ${fold_train_dir}/train_corpus \
        --train_query_file ${fold_train_dir}/train_query \
        --train_qrels ${fold_train_dir}/train_qrels.txt \
        --dev_corpus_file ${fold_train_dir}/dev_corpus \
        --dev_query_file ${fold_train_dir}/dev_query \
        --dev_qrels ${fold_train_dir}/dev_qrels.txt \
        --neg_file ${fold_train_dir}/hard_negs.txt \
        --max_len ${max_len} \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 32 \
        --train_group_size ${train_group_size} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --warmup_steps 1000 \
        --weight_decay 0.01 \
        --learning_rate 1e-5 \
        --num_train_epochs ${num_train_epochs} \
        --dataloader_num_workers 6 \
        --optim 'adamw_torch' \
        --logging_strategy 'epoch' \
        --save_strategy 'epoch' \
        --seed ${SEED} \
        --overwrite_output_dir \
        --evaluation_strategy 'epoch' \
        --save_total_limit 1 \
        --load_best_model_at_end \
        --metric_for_best_model 'eval_accuracy' \
        --sample_neg_from_topk ${sample_neg_from_topk}
        
        # --neg_dir ${neg_dir}


    prediction_save_path="./output/results/${dataset}/${fold}"
    # model_name_or_path="./output/cross_encoder/baseline/${dataset}/${fold}/b2_e10_g32/checkpoint-${ckpt}"
    # 加了 load_best_model_at_end 后，会把best model保存到 output_dir 顶层，因此可以直接从那里load
    model_name_or_path=${output_dir}
    prediction_topk=200
    prediction_topk_min=0

    python src/cross_encoder/run.py \
        --output_dir ./tmp \
        --model_name_or_path ${model_name_or_path} \
        --fp16  \
        --corpus_file ${fold_train_dir}/train_corpus \
        --max_len 200 \
        --do_predict  \
        --test_query_file ${fold_train_dir}/train_query \
        --test_file ${fold_train_dir}/train_qrels.txt \
        --prediction_save_path ${prediction_save_path}/train_qrels_score.txt \
        --dataloader_num_workers 6 \
        --tokenizer_name ${ptm} \
        --overwrite_prediction

    python src/cross_encoder/run.py \
        --output_dir ./tmp \
        --model_name_or_path ${model_name_or_path} \
        --fp16  \
        --corpus_file ${fold_train_dir}/train_corpus \
        --max_len 200 \
        --do_predict  \
        --test_query_file ${fold_train_dir}/train_query \
        --test_file ${prediction_save_path}/train_ranking.txt \
        --prediction_save_path ${prediction_save_path}/train_rerank_score_200.txt \
        --dataloader_num_workers 6 \
        --tokenizer_name ${ptm} \
        --prediction_topk ${prediction_topk} \
        --prediction_topk_min ${prediction_topk_min} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --overwrite_prediction

done