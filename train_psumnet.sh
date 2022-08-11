#!/bin/bash
#SBATCH -w gnode67
#SBATCH -A research
#SBATCH -c 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=ALL


source ~/venv/bin/activate

# module load python/3.6.8
module load u18/python/3.7.4
python3 main.py --config config/ntux_configs/body.yaml