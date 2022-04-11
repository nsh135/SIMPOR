#!/bin/bash -l

#SBATCH -o "OUT/out"
#SBATCH -e "OUT/out"
#SBATCH -p Contributors
##SBATCH -w GPU43
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
##SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --job-name=SIMPOR

source /apps/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow2
python main.py --dataset $1 --n_threads 128 --IR 3 --n_runs 4  --gridSearch True 
# python main.py --dataset $1 --n_threads 128 --IR 7 --n_runs 1   
#