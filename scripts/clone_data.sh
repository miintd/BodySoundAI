#!bin/bash

# remove old folders if they exist
[ -d "datasets" ] && rm -rf datasets
[ -d "feature" ] && rm -rf feature

# clone the dataset
wget https://huggingface.co/datasets/diemmii/opera-data/resolve/main/opera_data.zip
unzip opera_data.zip