#!/bin/bash

# This script sets up a conda environment for the story analyzer

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh

# Create conda environment
conda create -y -n story_analyzer python=3.9

# Activate environment
conda activate story_analyzer

# Install dependencies for BERT
conda install -y -c huggingface -c conda-forge transformers
conda install -y -c pytorch pytorch

# Install dependencies for Llama
pip install llama-cpp-python

echo ""
echo "==================================================="
echo "Setup complete! To use the story analyzer:"
echo "1. Activate the environment: conda activate story_analyzer"
echo "2. Run the analyzer: python story_analyzer.py path/to/your/story.txt"
echo "==================================================="