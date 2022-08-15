#!/bin/bash -l
#SBATCH --partition=taniu
#SBATCH --gres=gpu:1
#SBATCH --time 4:00:00
#SBATCH --mem=32G
##SBATCH --constraint 32

module load singularity cuda/11.1
singularity exec --nv t_maze.sif python triple_t_maze/post_main.py --explore_type none