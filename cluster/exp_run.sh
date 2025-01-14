#!/bin/bash
   
system="Normal"
strategy="FIFO"
mps_flage="False"
sync_flage="True"
node_kind="4*3090"                           #"4*3090, 3*2080ti, 4*2080"
model_kind="cv_model"                       #"cv_model, all_model"
gpu_mem_percent=0.8
job_num=10
print_level=11





python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum