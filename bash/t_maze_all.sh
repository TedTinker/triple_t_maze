#!/bin/bash -l

eval $1
eval $2

jid_list=()

for job in ${job_list[*]}
do
   jid=$(sbatch --array=1-${agents} triple_t_maze/bash/t_maze_$job.slurm)
   echo $jid
   jid=(${jid// / })
   jid=${jid[3]}     
   jid=$(sbatch --dependency afterok:$jid triple_t_maze/bash/t_maze_after_$job.slurm)
   echo $jid
   jid=(${jid// / })
   jid=${jid[3]}     
   jid_list+=($jid)
done

jid=$(sbatch --dependency afterok:$(echo ${jid_list[*]} | tr ' ' :) triple_t_maze/bash/t_maze_after_all.slurm)
echo $jid