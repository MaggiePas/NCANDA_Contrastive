#!/bin/bash
#SBATCH --job-name=train_depressive_symptoms_only_tabular
#SBATCH --time=06:00:00
#SBATCH -p kpohl
#SBATCH --gpus 1
#SBATCH -c 10
#SBATCH --mail-type=BEGIN,END,FAIL

ml python/3.9.0

python3 main.py
