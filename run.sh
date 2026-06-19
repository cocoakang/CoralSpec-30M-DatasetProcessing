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

DATASET_PATH=/mnt/e/coral_spectral_dataset_update_zips_d_unzip/

python generate_coral_mask.py --data_path $DATASET_PATH

python classifier_evaluation.py --data_path $DATASET_PATH

python plot_eval_results.py --data_path $DATASET_PATH