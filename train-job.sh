#!/bin/bash
#SBATCH --account=def-gunes
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G 
#SBATCH --time=2:00:00 
#SBATCH--nodes=2
export WANDB_MODE=dryrun
cd /home/melabbas/projects/def-gunes/melabbas/solar-flares-forecasting
module purge
module --force purge
source solar-flares-forecasting/bin/activate
# python experiments/cnn-sun-et-al/train.py --lr 0.001 --batch_size 64 --num_epochs 10 --debug
python datasets/sharp_sun_et_al/compute_mean_variance.py