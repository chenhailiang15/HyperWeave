#!/bin/bash

gpu_id_list=[0,1,2,3]
gpu_mem_percent=0.9
node_kind="4*A100"                           #"s4*3090, 4*3090, 3*2080ti, 4*2080"
model_kind="all_model"                       #"cv_model, all_model"
job_num=100
print_level=10


system="HyperWeave"
strategy="BN-SRSF"
mps_flage="True"
sync_flage="True"
overshared_factor=1
python platform_h/HyperWeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --overshared_factor ${overshared_factor}


system="HyperWeave"
strategy="BN-SRSF"
mps_flage="True"
sync_flage="True"
overshared_factor=2
python platform_h/HyperWeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --overshared_factor ${overshared_factor}

system="HyperWeave"
strategy="BN-SRSF"
mps_flage="True"
sync_flage="True"
overshared_factor=3
python platform_h/HyperWeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --overshared_factor ${overshared_factor}


system="HyperWeave"
strategy="BN-SRSF"
mps_flage="True"
sync_flage="True"
overshared_factor=4
python platform_h/HyperWeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --overshared_factor ${overshared_factor}

system="HyperWeave"
strategy="BN-SRSF"
mps_flage="True"
sync_flage="True"
overshared_factor=5
python platform_h/HyperWeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --overshared_factor ${overshared_factor}


system="HyperWeave"
strategy="BN-SRSF"
mps_flage="False"
sync_flage="False"
job_num=0
python platform_h/HyperWeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}

