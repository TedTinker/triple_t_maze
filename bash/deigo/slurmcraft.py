#%%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default = "default")
args = parser.parse_args()

import os 
try:    os.chdir("triple_t_maze/bash/deigo")
except: os.chdir("bash/deigo")

slurm_dict = {}
f = open("slurms.txt", "r")
slurms = f.readlines()
for line in slurms:
    if(line == "\n"): pass 
    else:
        name, text = line.split(":")
        slurm_dict[name.strip()] = text.strip()
        


with open("{}.slurm".format(args.name), "a") as f:
    f.write(
"""
#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --time 4:00:00
#SBATCH --mem=32G
##SBATCH --constraint 32

module load singularity
singularity exec t_maze.sif python triple_t_maze/main.py --id ${{SLURM_ARRAY_TASK_ID}} --explore_type {} {}
""".format(args.name, slurm_dict[args.name])[1:])
# %%

