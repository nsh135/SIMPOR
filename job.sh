#!/bin/bash -l
sbatch <<EOT
#!/bin/bash

#SBATCH -o "OUT/$1"
#SBATCH -e "OUT/$1"
#SBATCH -p Contributors
##SBATCH -w GPU43
#SBATCH --gpus=4
#SBATCH --cpus-per-task=31
##SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --job-name=SIMPOR

source /apps/anaconda3/etc/profile.d/conda.sh
conda activate tensorflow2
python main.py --dataset $1 --n_threads 128 --IR 3 --n_runs 5  --gridSearch True 
## python main.py --dataset $1 --n_threads 128 --IR 7 --n_runs 1   

exit 0
EOT