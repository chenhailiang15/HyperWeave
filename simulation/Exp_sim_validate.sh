#!/bin/bash

#!/bin/bash


gpu_mem_percent=0.9
node_kind="s4*3090"                           #"s4*3090, s8*3090, 4*3090, 3*2080ti, 4*2080"
model_kind="all_model"                       #"cv_model, all_model"
job_num=100
print_level=10
couple_init_iter_percent=0.2
overshared_factor=2
job_together_flage="False"
validation="True"
trace_id=5
job_come_time_factor=1


system="Weave"
strategy="BN-SRSF"
mps_flage="True"

python WeaveMaster.py --system ${system} --strategy ${strategy} \
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum  --overshared_factor ${overshared_factor} \
    --job_together_flage ${job_together_flage} --validation ${validation} --trace_id ${trace_id} \
    --couple_init_iter_percent ${couple_init_iter_percent} --mps_flage ${mps_flage} --job_come_time_factor ${job_come_time_factor}


system="Muri"
strategy="SRSF"
mps_flage="True"

python WeaveMaster.py --system ${system} --strategy ${strategy} \
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum  --overshared_factor ${overshared_factor} \
    --job_together_flage ${job_together_flage} --validation ${validation} --trace_id ${trace_id} \
    --couple_init_iter_percent ${couple_init_iter_percent} --mps_flage ${mps_flage}  --job_come_time_factor ${job_come_time_factor}

system="Normal"
strategy="SRSF"
mps_flage="True"

python WeaveMaster.py --system ${system} --strategy ${strategy} \
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum  --overshared_factor ${overshared_factor} \
    --job_together_flage ${job_together_flage} --validation ${validation} --trace_id ${trace_id} \
    --couple_init_iter_percent ${couple_init_iter_percent} --mps_flage ${mps_flage}  --job_come_time_factor ${job_come_time_factor}

system="Normal"
strategy="SRSF"
mps_flage="False"

python WeaveMaster.py --system ${system} --strategy ${strategy} \
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum  --overshared_factor ${overshared_factor} \
    --job_together_flage ${job_together_flage} --validation ${validation} --trace_id ${trace_id} \
    --couple_init_iter_percent ${couple_init_iter_percent} --mps_flage ${mps_flage}  --job_come_time_factor ${job_come_time_factor}

system="Normal"
strategy="SRTF"
mps_flage="False"

python WeaveMaster.py --system ${system} --strategy ${strategy} \
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum  --overshared_factor ${overshared_factor} \
    --job_together_flage ${job_together_flage} --validation ${validation} --trace_id ${trace_id} \
    --couple_init_iter_percent ${couple_init_iter_percent} --mps_flage ${mps_flage} --job_come_time_factor ${job_come_time_factor}


system="Normal"
strategy="FIFO"
mps_flage="False"

python WeaveMaster.py --system ${system} --strategy ${strategy} \
    --node_kind ${node_kind} --model_kind ${model_kind} --gpu_mem_percent ${gpu_mem_percent} --job_num ${job_num}\
    --print_level ${print_level} --write_trace --write_sum  --overshared_factor ${overshared_factor} \
    --job_together_flage ${job_together_flage} --validation ${validation} --trace_id ${trace_id} \
    --couple_init_iter_percent ${couple_init_iter_percent} --mps_flage ${mps_flage} --job_come_time_factor ${job_come_time_factor}

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
