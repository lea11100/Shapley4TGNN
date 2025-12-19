#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH -J "Shapley_Experiment_%a"
#SBATCH -p gpu
#SBATCH -A hpc-prf-wiki
#SBATCH --array=0-5

module load lang/Tkinter/3.9.6-GCCcore-11.2.0
module load lang/Python/3.9.5-GCCcore-10.3.0
module load system/CUDA/12.4.1
source /scratch/hpc-prf-wiki/sussekl/venv/bin/activate

options=( "MOOC" "Wikipedia" "Reddit" "Generated" "MOOCTGN" "MOOCOneHop")

srun --ntasks=1 --gres=gpu:a100:1 python -m Evaluation.run_exec --dataset ${options[$SLURM_ARRAY_TASK_ID]} --explainer all
