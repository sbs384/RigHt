#!/bin/bash
dataset='SQuAD'
model_name_or_path="/home/zhaobo/hf_models/bert_base_uncased" # 'Shitao/RetroMAE' # 
data_root_dir='data/tokenized_data/SQuAD'
query_max_len=24
passage_max_len=168

CLSDIM=128
AGGDIM=640

sentence_pooling_method='cls'
batch_size=4
train_group_size=8
gradient_accumulation_steps=8
lr=5e-5
epoch=10

sample_neg_from_topk=20
output_dir="./output/bi_encoder/final/agg/${dataset}/"
save_total_limit=1
# result_file="bs_${batch_size}_ga${gradient_accumulation_steps}_s${sample_neg_from_topk}"
neg_source='ance'
adv_norm=0.001

teacher_score_dir="./output/results/${dataset}"
# teacher_model_name_or_path='./output/cross_encoder/v2_b2_e5_g33/checkpoint-29565'
    # --overwrite_output_dir \
# result_file="bs_${batch_size}_ga${gradient_accumulation_steps}_s${sample_neg_from_topk}_clw_${weight}"
kl_loss_weight=1
contrastive_loss_weight=0.01
rm -r ${output_dir}
result_dir="./results/squad/final/agg_adv_kd/${dataset}"
for SEED in 42 #  3000 9958 999 888 10000 2023 2024 #  42 0 666 #
do
# result_file="bs_${batch_size}_ga${gradient_accumulation_steps}_s${sample_neg_from_topk}_clw_${weight}"
result_file="${SEED}"
python src/bi_encoder/run.py \
    --output_dir ${output_dir} \
    --model_name_or_path ${model_name_or_path} \
    --model_dir ${output_dir} \
    --tokenizer_name ${model_name_or_path} \
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
    --learning_rate 5e-5 \
    --num_train_epochs ${epoch} \
    --dataloader_num_workers 6 \
    --optim 'adamw_torch' \
    --save_strategy 'epoch' \
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
    --dev_corpus_file ${data_root_dir}/dev_corpus \
    --dev_query_file ${data_root_dir}/dev_query \
    --dev_qrels ${data_root_dir}/dev_qrels.txt \
    --kl_loss_weight ${kl_loss_weight} \
    --do_eval \
    --add_pooler \
    --projection_out_dim ${CLSDIM} \
    --agg_dim ${AGGDIM} \
    --extra_neg \
    --neg_source ${neg_source} \
    --teacher_score_files ${teacher_score_dir}/train_qrels_score.txt,${teacher_score_dir}/train_rerank_score_200.txt \
    --adv_training \
    --adv_norm ${adv_norm} \
    --skip_mlm False \

    
done



