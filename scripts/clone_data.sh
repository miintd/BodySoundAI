#!bin/bash

# remove old folders if they exist
[ -d "datasets" ] && rm -rf datasets
[ -d "feature" ] && rm -rf feature

# clone the dataset
wget https://huggingface.co/datasets/diemmii/opera-data/resolve/main/opera_dataset.zip
unzip opera_dataset.zip