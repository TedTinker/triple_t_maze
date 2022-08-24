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
        jid=$(sbatch --dependency afterok:$jid --export explore_type=$job triple_t_maze/bash/after.slurm)
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

jid=$(sbatch --dependency afterok:$(echo ${jid_list[*]} | tr ' ' :) --export explore_type=$order triple_t_maze/bash/after.slurm)
echo $jid