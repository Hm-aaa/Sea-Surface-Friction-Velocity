#!/bin/bash
#SBATCH --job-name=U*
#SBATCH --partition=x86_64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclude=2488-node02
#SBATCH --cpus-per-task=32
# SBATCH --array=0-9%4

# models=(
#     xgboost
#     lightgbm
#     random_forest
#     adaboost
#     catboost
#     svm
#     gbdt
#     cart
#     plsr
#     mlp
# )

# model=${models[$SLURM_ARRAY_TASK_ID]}

# python CdStacking.py $model u_star_down

python CdStacking.py

