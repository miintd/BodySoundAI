#!/bin/bash

echo "Checking for existing audio environment..."
if conda env list | grep -q '^audio'; then
    echo "⚠️  Found existing 'audio' environment. Removing..."
    conda env remove -n audio --yes
    echo "✅ Old environment removed"
else
    echo "No existing audio environment found"
fi

echo ""
echo "Creating/updating conda environment from environment.yml..."
conda env create -f environment.yml

echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate audio