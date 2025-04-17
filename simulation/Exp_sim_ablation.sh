#!/bin/bash

gpu_mem_percent=0.9
node_kind="cluster"                           #"s4*3090, s8*3090, 4*3090, 3*2080ti, 4*2080"
model_kind="all_model"                       #"cv_model, all_model"
job_num=1000
print_level=10
couple_init_iter_percent=1
overshared_factor=2
validation="False"
job_come_time_factor=1

trace_id=101
job_together_flage="True"


system="HyperWeave"
strategy="FIFO"
mps_flage="True"
sync_flage="True"
gpu_select_mode="random"

python platform/HyperWeaveMaster.py --system ${system} --strategy ${strategy} \
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum  --overshared_factor ${overshared_factor} \
    --job_together_flage ${job_together_flage} --validation ${validation} --trace_id ${trace_id} \
    --couple_init_iter_percent ${couple_init_iter_percent} --mps_flage ${mps_flage} --job_come_time_factor ${job_come_time_factor}\
    --sync_flage ${sync_flage} --gpu_select_mode ${gpu_select_mode}



system="HyperWeave"
strategy="BN-SRSF"
mps_flage="True"
sync_flage="True"

python platform/HyperWeaveMaster.py --system ${system} --strategy ${strategy} \
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum  --overshared_factor ${overshared_factor} \
    --job_together_flage ${job_together_flage} --validation ${validation} --trace_id ${trace_id} \
    --couple_init_iter_percent ${couple_init_iter_percent} --mps_flage ${mps_flage} --job_come_time_factor ${job_come_time_factor}\
    --sync_flage ${sync_flage}

system="HyperWeave"
strategy="BN-SRSF"
mps_flage="True"
sync_flage="False"

python platform/HyperWeaveMaster.py --system ${system} --strategy ${strategy} \
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum  --overshared_factor ${overshared_factor} \
    --job_together_flage ${job_together_flage} --validation ${validation} --trace_id ${trace_id} \
    --couple_init_iter_percent ${couple_init_iter_percent} --mps_flage ${mps_flage}  --job_come_time_factor ${job_come_time_factor}\
    --sync_flage ${sync_flage}

