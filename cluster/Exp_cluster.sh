#!/bin/bash

gpu_id_list=[4,5,6,7]
gpu_mem_percent=0.9
node_kind="4*3090"                           #"s4*3090, 4*3090, 3*2080ti, 4*2080"
model_kind="all_model"                       #"cv_model, all_model"
job_num=2
print_level=10


system="Weave"
strategy="SRSF"
mps_flage="True"
sync_flage="True"
python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}

system="Weave"
strategy="SRSF"
mps_flage="False"
sync_flage="True"
python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}

system="Weave"
strategy="SRSF"
mps_flage="True"
sync_flage="False"
python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}

system="Weave"
strategy="SRSF"
mps_flage="False"
sync_flage="False"
python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}

system="Weave"
strategy="BN-SRSF"
mps_flage="True"
sync_flage="True"
python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}

system="Normal"
strategy="SRSF"
mps_flage="False"
sync_flage="False"

python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}

system="Muri"
strategy="SRSF"
mps_flage="False"
sync_flage="False"

python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}

system="Weave"
strategy="FIFO"
mps_flage="True"
sync_flage="True"
python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}

system="Weave"
strategy="SRTF"
mps_flage="True"
sync_flage="True"
python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}