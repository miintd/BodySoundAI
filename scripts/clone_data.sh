#!/bin/bash

# install git lfs and clone the dataset
git lfs install
git clone --depth 1 https://huggingface.co/datasets/diemmii/opera-data		
cd opera-data
git sparse-checkout init --cone
git sparse-checkout set datasets
git pull
mv datasets ../
cd ..
echo "Dataset cloned successfully."

cd opera-data
git sparse-checkout set feature
git pull
mv feature ../
cd ..
echo "Feature cloned successfully."