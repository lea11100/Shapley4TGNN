#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH -J "Shapley_Experiment_TempME"
#SBATCH -p gpu
#SBATCH -A hpc-prf-wiki


module load lang/Tkinter/3.9.6-GCCcore-11.2.0
module load lang/Python/3.9.5-GCCcore-10.3.0
module load system/CUDA/12.4.1
source /scratch/hpc-prf-wiki/sussekl/venv/bin/activate

python -m Evaluation.eval2 --dataset Reddit --explainer tempme --preprocessing True
