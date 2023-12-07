#!/bin/bash
#SBATCH --job-name=train_depressive_symptoms_only_tabular
#SBATCH --time=6:00:00
#SBATCH -p gpu
#SBATCH --gpus 3
#SBATCH -c 20
#SBATCH --mail-type=BEGIN,END,FAIL

ml python/3.9.0

python3 main_as.py
