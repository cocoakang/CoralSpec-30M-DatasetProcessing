#!/bin/bash

#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J create_sif
#SBATCH -o dcgm/create_sif.%J.out
#SBATCH -e dcgm/create_sif.%J.err
#SBATCH --mail-user=kaizhang.kang@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=12:30:00
#SBATCH --job-name=dataset_pub
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

DATASET_PATH=/ibex/project/c2248/coral_spectral_dataset/

python generate_coral_mask.py --data_path $DATASET_PATH