#!/bin/bash


gpu_mem_percent=0.9
node_kind="4*3090"                           #"4*3090, 3*2080ti, 4*2080"
model_kind="all_model"                       #"cv_model, all_model"
job_num=100
print_level=11

system="Normal"
strategy="FIFO"
mps_flage="False"
sync_flage="False"

python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum

system="Muri"
strategy="FIFO"
mps_flage="False"
sync_flage="False"

python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum

# system="Weave"
# strategy="FIFO"
# sync_flage="True"
# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum


# system="Muri"
# strategy="FIFO"
# sync_flage="False"
# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum

# system="Normal"
# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum

# system="Muri"
# strategy="FIFO"
# gpu_mem_percent=0.9


# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum



# system="Weave"
# strategy="FIFO"
# gpu_mem_percent=0.8

# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum