#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -N 2
#SBATCH -n 20
#SBATCH -J "Shapley_Experiment"
#SBATCH -p gpu
#SBATCH -A hpc-prf-wiki


module load lang/Python/3.9.5-GCCcore-10.3.0
module load system/CUDA/12.4.1
source .venv/bin/activate
python experiments/omniopt-vs-nsga2.py