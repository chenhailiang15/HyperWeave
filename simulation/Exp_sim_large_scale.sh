#!/bin/bash

#!/bin/bash


gpu_mem_percent=0.9
node_kind="cluster"                           #"s4*3090, 4*3090, 3*2080ti, 4*2080"
model_kind="all_model"                       #"cv_model, all_model"
print_level=5
validation="False"
overshared_factor=3
bucket_length=10000000
sync_flage="True"


job_together_flage="False"
node_num=100   #max 1814
job_num=10000


system=$1
strategy=$2
trace_id=$3

python WeaveMaster.py --system ${system} --strategy ${strategy} \
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum  --overshared_factor ${overshared_factor} --job_together_flage ${job_together_flage}\
    --validation ${validation} --node_num ${node_num} --trace_id ${trace_id} --bucket_length ${bucket_length} --sync_flage ${sync_flage}


# system="Weave"
# strategy="BN-SRSF"
# mps_flage="True"
# sync_flage="True"
# overshared_factor=2
# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --overshared_factor ${overshared_factor}

# system="Weave"
# strategy="BN-SRSF"
# mps_flage="True"
# sync_flage="True"
# overshared_factor=4
# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --overshared_factor ${overshared_factor}

# system="Weave"
# strategy="BN-SRSF"
# mps_flage="True"
# sync_flage="True"
# overshared_factor=5
# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list} --overshared_factor ${overshared_factor}


# system="Weave"
# strategy="SRSF"
# mps_flage="False"
# sync_flage="False"
# job_num=0
# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}

# system="Weave"
# strategy="SRSF"
# mps_flage="True"
# sync_flage="False"
# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}

# system="Weave"
# strategy="SRSF"
# mps_flage="False"
# sync_flage="True"
# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}

# system="Weave"
# strategy="SRSF"
# mps_flage="False"
# sync_flage="False"
# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}

# system="Weave"
# strategy="BN-SRSF"
# mps_flage="True"
# sync_flage="True"
# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}

# system="Muri"
# strategy="SRSF"
# mps_flage="False"
# sync_flage="False"

# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}

# system="Normal"
# strategy="SRSF"
# mps_flage="False"
# sync_flage="False"

# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}



# system="Weave"
# strategy="FIFO"
# mps_flage="True"
# sync_flage="True"
# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
#     --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}

# system="Weave"
# strategy="SRTF"
# mps_flage="True"
# sync_flage="True"
# python WeaveMaster.py --system ${system} --strategy ${strategy} --mps_flage ${mps_flage} --sync_flage ${sync_flage}\
#     --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    # --print_level ${print_level} --write_trace --write_sum --gpu_id_list ${gpu_id_list}
