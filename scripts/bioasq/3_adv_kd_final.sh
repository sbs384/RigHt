#!/bin/bash
# 利用难负例训练双塔模型

# 必须用1.2, 1.2才有 mlm head的权重
ptm="/home/zhaobo/hf_models/biobert_base_cased_mix"
dataset=$1 #'9b'
data_root_dir="data/tokenized_data/bioasq/${dataset}"
sentence_pooling_method='cls'
model_name_or_path=${ptm}

query_max_len=24
passage_max_len=168

train_group_size=8
gradient_accumulation_steps=8
batch_size=4

epoch=10
per_device_eval_batch_size=32
SEED=42
CLSDIM=128
AGGDIM=640

lr=5e-5
contrastive_loss_weight=1 # 0.01 for SQuAD, 0.1 for BioASQ
kl_loss_weight=0.01 # 0.15
adv_norm=0.005 # 0.001 for SQuAD, 0.005 for bioasq

sample_neg_from_topk=20
# result_file="bs_${batch_size}_ga${gradient_accumulation_steps}_s${sample_neg_from_topk}"
neg_source='ance'
save_total_limit=1

# load_best_model_at_end: 训练完毕加载最优ckpt
# metric_for_best_model: 最优模型的指标

    # --overwrite_output_dir \
for SEED in 42 # 999 10000 #2024 3000 9958 999 10000 # 888 # 42 0 666  2023 # 
do
for fold in 1 # 2 3 4 5
do
    result_dir="./results/final/agg_adv_kd/${dataset}"
    # output_dir="./output/bi_encoder/kd/${dataset}/${fold}/${SEED}"
    output_dir="./output/bi_encoder/final/agg/${dataset}/"
    rm -r ${output_dir}
    prediction_save_path="./output/results/${dataset}/${fold}"
    fold_train_dir="${data_root_dir}/fold_${fold}"
    result_file="${SEED}"
    python src/bi_encoder/run.py \
        --output_dir ${output_dir} \
        --model_name_or_path ${model_name_or_path} \
        --model_dir ${output_dir} \
        --tokenizer_name ${ptm} \
        --do_predict  \
        --do_train \
        --save_strategy 'epoch' \
        --corpus_file ${fold_train_dir}/train_corpus \
        --train_query_file ${fold_train_dir}/train_query \
        --train_qrels ${fold_train_dir}/train_qrels.txt \
        --query_max_len ${query_max_len} \
        --passage_max_len ${passage_max_len} \
        --test_corpus_file ${data_root_dir}/test_corpus \
        --test_query_file ${data_root_dir}/test_query \
        --test_qrels ${data_root_dir}/test_qrels.txt \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --train_group_size ${train_group_size} \
        --sample_neg_from_topk ${sample_neg_from_topk} \
        --learning_rate 5e-5 \
        --num_train_epochs ${epoch} \
        --dataloader_num_workers 6 \
        --optim 'adamw_torch' \
        --sentence_pooling_method ${sentence_pooling_method} \
        --warmup_ratio 0.1 \
        --save_total_limit ${save_total_limit} \
        --seed ${SEED} \
        --logging_strategy 'epoch' \
        --result_dir ${result_dir} \
        --result_file ${result_file} \
        --fp16 \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --overwrite_output_dir \
        --contrastive_loss_weight ${contrastive_loss_weight} \
        --kl_loss_weight ${kl_loss_weight} \
        --neg_source ${neg_source} \
        --extra_neg \
        --dev_corpus_file ${fold_train_dir}/dev_corpus \
        --dev_query_file ${fold_train_dir}/dev_query \
        --dev_qrels ${fold_train_dir}/dev_qrels.txt \
        --do_eval \
        --adv_training \
        --adv_norm ${adv_norm} \
        --add_pooler \
        --projection_out_dim ${CLSDIM} \
        --agg_dim ${AGGDIM} \
        --teacher_score_files ${prediction_save_path}/train_qrels_score.txt,${prediction_save_path}/train_rerank_score_200.txt \
        --skip_mlm False \
        

done
done

        # 
