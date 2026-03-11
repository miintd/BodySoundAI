#!/bin/bash

echo "Creating conda environment..."
conda env create -f environment.yml

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate audio