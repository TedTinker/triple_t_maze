#!/bin/bash -l
#SBATCH --partition=taniu
#SBATCH --gres=gpu:1
#SBATCH --time 4:00:00
#SBATCH --mem=32G
##SBATCH --constraint 32

module load singularity cuda/11.1
singularity exec --nv t_maze.sif python triple_t_maze/main.py --id ${SLURM_ARRAY_TASK_ID} --explore_type entropy_a_none_t_0 --eta 0 --target_entropy 0