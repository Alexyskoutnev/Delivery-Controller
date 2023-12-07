#!/bin/bash

# Create and activate the conda environment
conda env create -f environment.yml
source activate RL_Pizza
pip install -r requirements.txt  # if you have a requirements.txt file
echo "Conda environment created and activated."