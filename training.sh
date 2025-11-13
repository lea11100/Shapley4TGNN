#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH -J "Shapley_Experiment"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:3
#SBATCH -A hpc-prf-wiki


module load lang/Python/3.9.5-GCCcore-10.3.0
module load system/CUDA/12.4.1
source .venv/bin/activate
python -m Evaluation.MOOC.training &
python -m Evaluation.Reddit.training &
python -m Evaluation.Wikipedia.training &
wait