# SqlNet
This repo provides an implementation of SQLNet and Seq2SQL neural networks for predicting SQL queries on WikiSQL dataset. The paper is available at here.

# Installation
The Data is in :https://huggingface.co/datasets/AI4DS/sql_generator_no_cot

#Downloading the Glove Embedding.
Download the pretrained glove embedding from here(https://github.com/stanfordnlp/GloVe)
using bash download_glove.sh

#Extract the glove embedding for training.
Run the following command to process the pretrained glove embedding for training the word embedding:

python extract_vocab.py

# Train
The training script is train.py. To see the detailed parameters for running:

python train.py -h
