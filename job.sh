#!/bin/bash
#SBATCH -t 18:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -J "Shapley_Experiment_wikipedia"
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -A hpc-prf-wiki


module load lang/Python/3.9.5-GCCcore-10.3.0
module load system/CUDA/12.4.1
source .venv/bin/activate
python experiments/omniopt-vs-nsga2.py