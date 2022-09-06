#!/bin/bash -l

eval $1
eval $2

jid_list=()

for job in ${job_list[*]}
do
    if [ $job == "break" ]
    then
        :
    elif [ $job == "empty_space" ]
    then
        :
    else
        singularity exec t_maze.sif python triple_t_maze/bash/slurmcraft.py --name $job
        jid=$(sbatch --array=1-${agents} triple_t_maze/bash/$job.slurm)
        echo $jid
        jid=(${jid// / })
        jid=${jid[3]}     
        jid=$(sbatch --dependency afterok:$jid --export explore_type=$job triple_t_maze/bash/after.slurm)
        echo $jid
        jid=(${jid// / })
        jid=${jid[3]}     
        jid_list+=($jid)
        rm triple_t_maze/bash/$job.slurm
    fi
done

first_job=true
order="("
for job in ${job_list[*]}
do 
    if [ $first_job = false ] 
    then
        order+="+"
    fi
    order+="$job" 
    first_job=false
done
order+=")"

jid=$(sbatch --dependency afterok:$(echo ${jid_list[*]} | tr ' ' :) --export explore_type=$order triple_t_maze/bash/after.slurm)
echo $jid