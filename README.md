# RigHt
This repository provides the PyTorch implementation of our paper "Rectifying and Discriminating Hard Negatives for
Retrieval Question Answering".

## Environment
We mainly rely on these packages:
```bash
# Install huggingface transformers and pytorch
transformers==4.23.1
torch==1.12.1
```
Our experiment is conducted on a NVIDIA RTX 3090 24G with CUDA version 11.6.

## Dataset
We follow [RBAR](https://github.com/Ba1Jun/BioReQA) to process data. The dataset and scripts can be found in their repo.

## Models
Due to the lack of mlm head parameters in biobert-base-cased-v1.1 we mix the biobert-base-cased-v1.1 and biobert-base-cased-v1.2 for the training process using the InTeR and DRIve mechanism on ReQA BioASQ datasets. The mix process is realized by running the script "tools/mix_model.ipynb".

## Example
We show the example of running scripts on SQuAD dataset. 

#### 1. Convert the SQuAD dataset for training
Run the script "retriever/SQuAD/convert.ipynb" to change the dataset format to a general one.
Preprosess the dataset through "retriever/SQuAD/preprocess.py", including tokenization, truncation and transformation.

It is mostly the same for BioASQ datasets, except for that BioASQ uses variable validation sets following the 5-Fold setting, while SQuAD uses a fixed dev set.

#### 2. Train the basic dual-encoder
Run the script "scripts/squad/0_base_prepare.sh".

#### 3. Obtain hard examples.
Run the script "scripts/squad/1_min_hard_neg".

#### 4. Train the cross-encoder to rectify the mislabeling of false negative samples.
Run the script "scripts/squad/2_cross_rerank".

#### 5. Utilize both InTeR and DRIve mechanism to enhance the training of the dual-encoder model with hard negative samples.
Run the script "scripts/squad/3_adv_kd_final".

## License and Disclaimer
Please see the LICENSE file for details. Downloading data indicates your acceptance of our disclaimer.

## Contact
For help or issues, please create an issue.
