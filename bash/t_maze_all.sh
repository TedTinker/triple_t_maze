#!/bin/bash -l

eval $1

jid1=$(sbatch --array=1-${agents} triple_t_maze/bash/t_maze_none.slurm)
echo $jid1
jid2=$(sbatch --array=1-${agents} triple_t_maze/bash/t_maze_entropy.slurm)
echo $jid2
jid3=$(sbatch --array=1-${agents} triple_t_maze/bash/t_maze_curious.slurm)
echo $jid3

jid1=(${jid1// / })
jid1=${jid1[3]}     

jid2=(${jid2// / })
jid2=${jid2[3]}     

jid3=(${jid3// / })
jid3=${jid3[3]}     

jid1=$(sbatch --dependency afterok:$jid1 triple_t_maze/bash/t_maze_after_none.slurm)
echo $jid1
jid2=$(sbatch --dependency afterok:$jid2 triple_t_maze/bash/t_maze_after_entropy.slurm)
echo $jid2
jid3=$(sbatch --dependency afterok:$jid3 triple_t_maze/bash/t_maze_after_curious.slurm)
echo $jid3

jid1=(${jid1// / })
jid1=${jid1[3]}     

jid2=(${jid2// / })
jid2=${jid2[3]}     

jid3=(${jid3// / })
jid3=${jid3[3]}   

jid=$(sbatch --dependency afterok:$jid1:$jid2:$jid3 triple_t_maze/bash/t_maze_after_all.slurm)
echo $jid