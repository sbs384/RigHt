#!/bin/bash
# 导出每个fold的难负例

dataset=$1 #'9b'
query_max_len=512
passage_max_len=512
sentence_pooling_method='cls'
batch_size=32
train_group_size=1
gradient_accumulation_steps=1
lr=5e-5
epoch=10
SEED=42
sample_neg_from_topk=20
ptm="dmis-lab/biobert-base-cased-v1.1"
output_dir="./tmp"
data_root_dir="data/tokenized_data/bioasq/${dataset}"

# 训练每个fold的基线模型
# 不需要额外负例
for fold in 1 2 3 4 5
do
    model_name_or_path="./output/bi_encoder/baseline/${dataset}/${fold}/" # checkpoint-770
    output_dir="./output/bi_encoder/baseline/${dataset}/${fold}"
    fold_train_dir="${data_root_dir}/fold_${fold}"
    prediction_save_path="./output/results/${dataset}/${fold}"
    neg_file="${fold_train_dir}/hard_negs.txt"
    # 推理时使用最大长度，与reqa一致

    # 保存训练集 resp
    python src/bi_encoder/run.py \
        --output_dir ${output_dir} \
        --model_name_or_path ${model_name_or_path}  \
        --test_corpus_file ${fold_train_dir}/train_corpus \
        --test_query_file ${fold_train_dir}/train_query \
        --test_qrels ${fold_train_dir}/train_qrels.txt \
        --query_max_len ${query_max_len} \
        --passage_max_len ${passage_max_len} \
        --do_predict \
        --prediction_save_path ${prediction_save_path} \
        --per_device_eval_batch_size 256 \
        --dataloader_num_workers 6 \
        --eval_accumulation_steps 100 \
        --tokenizer_name ${ptm} \
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
        --qrels_file ${fold_train_dir}/train_qrels.txt \
        --output_neg_file ${neg_file}
            
done