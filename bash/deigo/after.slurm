#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --time 24:00:00
#SBATCH --mem=32G
##SBATCH --constraint 32

module load singularity
singularity exec t_maze.sif python triple_t_maze/post_main.py --explore_type $explore_type