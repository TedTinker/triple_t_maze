#!/bin/bash -l

eval $1
eval $2

jid_list=()

for job in ${job_list[*]}
do
    if [ $job == "break" ]
    then
        :
    else
        jid=$(sbatch --array=1-${agents} triple_t_maze/bash/t_maze_$job.slurm)
        echo $jid
        jid=(${jid// / })
        jid=${jid[3]}     
        jid=$(sbatch --dependency afterok:$jid triple_t_maze/bash/t_maze_after_$job.slurm)
        echo $jid
        jid=(${jid// / })
        jid=${jid[3]}     
        jid_list+=($jid)
    fi
done

jobs=0
order="("
for job in ${job_list[*]}
do 
    if [ $jobs -ne 0 ] 
    then
        order+="+"
    fi
    order+="$job" 
    jobs+=1
done
order+=")"

jid=$(sbatch --dependency afterok:$(echo ${jid_list[*]} | tr ' ' :) --export order=$order triple_t_maze/bash/t_maze_after_all.slurm)
echo $jid