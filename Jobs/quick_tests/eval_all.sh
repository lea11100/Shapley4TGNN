#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -J "Shapley_Experiment"
#SBATCH -p gpu
#SBATCH -A hpc-prf-wiki

module load lang/Tkinter/3.9.6-GCCcore-11.2.0
module load lang/Python/3.9.5-GCCcore-10.3.0
module load lib/libffi/3.4.5-GCCcore-13.3.0
module load system/CUDA/12.4.1
source venv/bin/activate

python -m Evaluation.run_exec --dataset Generated --explainer all --num_samples 10
