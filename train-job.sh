#!/bin/bash
#SBATCH --account=def-gunes
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G 
#SBATCH --time=6:00:00 
#SBATCH--nodes=2
python python train.py --lr 0.001 --batch_size 64 --num_epochs 2