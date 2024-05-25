import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str,
                        default='bert-base-uncased')
    parser.add_argument("--input_dir", type=str,
                        default=f'./data/SQuAD_formatted')
    parser.add_argument("--output_dir", type=str,
                        default=f'./data/tokenized_data/SQuAD')
    parser.add_argument("--max_seq_length", type=int, default=512)
    return parser.parse_args()


def save_to_json(input_file, output_file, id_file):
    with open(output_file, 'w',
              encoding='utf-8') as f, open(id_file, 'w',
                                           encoding='utf-8') as fid:
        cnt = 0
        for line in open(input_file, encoding='utf-8'):
            line = line.strip('\n').split('\t')
            if len(line) == 2:
                data = {"id": line[0], 'text': line[1]}
            else:
                if len(line)<2:
                    print('here:',line)
                    import sys
                    sys.exit(0)
                data = {"id": line[0], 'title': line[1], 'text': line[2]}
            f.write(json.dumps(data) + '\n')
            fid.write(line[0] + '\t' + str(cnt) + '\n')
            cnt += 1


def preprocess_qrels(train_qrels, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in open(train_qrels, encoding='utf-8'):
            line = line.strip().split('\t')
            f.write(line[0] + '\t' + line[1] + '\n')


def tokenize_function(examples):
    return tokenizer(examples["text"],
                     add_special_tokens=False,
                     truncation=True,
                     max_length=max_length,
                     return_attention_mask=False,
                     return_token_type_ids=False)

def tokenize_test_function(examples):
    return tokenizer(examples["text"],
                     add_special_tokens=False,
                     truncation=False,
                     return_attention_mask=False,
                     return_token_type_ids=False)

args = get_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
max_length = args.max_seq_length

if __name__ == '__main__':
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for split in ['train','dev','test']:
        if not os.path.exists(f'{args.input_dir}/qrels_{split}.tsv'):
            continue
        Path(os.path.join(args.output_dir, f'{split}_corpus')).mkdir(parents=True,
                                                            exist_ok=True)
        preprocess_qrels(f'{args.input_dir}/qrels_{split}.tsv',
                        os.path.join(args.output_dir, f'{split}_qrels.txt'))

        # corpus
        save_to_json(f'{args.input_dir}/{split}_corpus.tsv', f'{args.input_dir}/{split}_corpus.json',
                    os.path.join(args.output_dir, f'{split}_corpus/mapping_id.txt'))
        corpus = load_dataset('json',
                            data_files=f'{args.input_dir}/{split}_corpus.json',
                            split='train')
        corpus = corpus.map(tokenize_function,
                            num_proc=8,
                            remove_columns=["title", "text"],
                            batched=True)
        corpus.save_to_disk(os.path.join(args.output_dir, f'{split}_corpus'))
        print(f'{split} corpus dataset:', corpus)


    for split in ['train','dev','test']:
        if not os.path.exists(f'{args.input_dir}/{split}_query.txt'):
            continue
        Path(os.path.join(args.output_dir, f'{split}_query')).mkdir(parents=True,
                                                        exist_ok=True)
        save_to_json(f'{args.input_dir}/{split}_query.txt', f'{args.input_dir}/{split}_query.json',
                     os.path.join(args.output_dir, f'{split}_query/mapping_id.txt'))
        query = load_dataset('json',
                                data_files=f'{args.input_dir}/{split}_query.json',
                                split='train')
        query = query.map(tokenize_function,
                                    num_proc=8,
                                    remove_columns=["text"],
                                    batched=True)
        query.save_to_disk(os.path.join(args.output_dir, f'{split}_query'))
        print(f'{split} query dataset:', query)

