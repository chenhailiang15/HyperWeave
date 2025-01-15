#!/bin/bash

system="Normal"
strategy="FIFO"
gpu_mem_percent=0.9

mps_flage="True"
sync_flage="True"
node_kind="4*2080"                           #"4*3090, 3*2080ti, 4*2080"
model_kind="all_model"                       #"cv_model, all_model"
job_num=100
print_level=11



python WeaveMaster.py --system ${system} --strategy ${strategy} \
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum

system="Muri"
python WeaveMaster.py --system ${system} --strategy ${strategy} \
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum

system="Weave"
python WeaveMaster.py --system ${system} --strategy ${strategy} \
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum
