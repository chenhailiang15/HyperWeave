#!/bin/bash

node_kind="s4*3090" #"s4*3090, 4*3090, 3*2080ti, 4*2080"
gpu_id_list=[0,1,2,3]

# node_kind="4*3090" #"s4*3090, 4*3090, 3*2080ti, 4*2080"
# gpu_id_list=[0,1,2,3,4,5,6,7]


gpu_mem_percent=0.9                           
model_kind="all_model"                       #"cv_model, all_model"
job_num=100
print_level=10
job_come_time_factor=6
job_together_flage="False"

overshared_factor=2
trace_id=0

system="Weave"
strategy="BN-SRSF"
mps_flage="True"
sync_flage="True"

python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --trace_id ${trace_id}\
    --overshared_factor ${overshared_factor} --job_come_time_factor ${job_come_time_factor} --job_together_flage ${job_together_flage}


system="Muri"
strategy="SRSF"
mps_flage="True"
sync_flage="False"


python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --trace_id ${trace_id}\
    --overshared_factor ${overshared_factor} --job_come_time_factor ${job_come_time_factor} --job_together_flage ${job_together_flage}


system="Normal"
strategy="SRSF"
mps_flage="True"
sync_flage="False"

python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --trace_id ${trace_id}\
    --overshared_factor ${overshared_factor} --job_come_time_factor ${job_come_time_factor} --job_together_flage ${job_together_flage}

    



system="Normal"
strategy="SRSF"
mps_flage="False"
sync_flage="False"

python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --trace_id ${trace_id}\
    --overshared_factor ${overshared_factor} --job_come_time_factor ${job_come_time_factor} --job_together_flage ${job_together_flage}



system="Normal"
strategy="SRTF"
mps_flage="False"
sync_flage="False"

python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --trace_id ${trace_id}\
    --overshared_factor ${overshared_factor} --job_come_time_factor ${job_come_time_factor} --job_together_flage ${job_together_flage}

system="Normal"
strategy="FIFO"
mps_flage="False"
sync_flage="False"

python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --trace_id ${trace_id}\
    --overshared_factor ${overshared_factor} --job_come_time_factor ${job_come_time_factor} --job_together_flage ${job_together_flage}


system="Weave"
strategy="BN-SRSF"
mps_flage="True"
sync_flage="False"

python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --trace_id ${trace_id}\
    --overshared_factor ${overshared_factor} --job_come_time_factor ${job_come_time_factor} --job_together_flage ${job_together_flage}


system="Weave"
strategy="BN-SRSF"
mps_flage="False"
sync_flage="False"

python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --trace_id ${trace_id}\
    --overshared_factor ${overshared_factor} --job_come_time_factor ${job_come_time_factor} --job_together_flage ${job_together_flage}


#关闭MPS
# job_num=0
# system="Weave"
# strategy="SRSF"
# mps_flage="False"
# sync_flage="False"
# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --trace_id ${trace_id}