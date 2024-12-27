
#!/bin/bash

#Ber batch size should be about 8

# Is_Master=false

# if [ "${Is_Master}" = true ]; then
#     node_rank=0
#     net_card="eno2"
# else
#     node_rank=1
#     net_card="eno1"
# fi
# conda activate torch_mp_chl
system="Muri"
mode="train"
node_rank=0
net_card="eno1"
MASTER_ADDR="localhost"   # one node:localhost  multi node: master ip
MASTER_PORT0="12345"
MASTER_PORT1="12346"
MASTER_PORT2="12347"
MASTER_PORT3="12348"

model_name0="AlexNet"
model_name1="GCN"
world_size=2
nprocs_list=[2,0]

gpu_id_list="[[0,1],[]]"


total_epochs=2
total_epochs1=100
batch_size=16           #8 for Bert (default:16)
worker_num=4

squad_data_size=1000
layer_num=100        #5000 for GCN (default:10)
layer_feature=100        #100 for GCN (default:10)

sample_interval=0.1

max_sync_num=1
# shm_name_list="{}"

idx_on_gpu0=0
job_idx0=10
idx_on_gpu1=1
job_idx1=11
idx_on_gpu2=2
job_idx2=12
idx_on_gpu3=3
job_idx3=13


python WeaveExecutor.py --MASTER_ADDR ${MASTER_ADDR} --MASTER_PORT ${MASTER_PORT0} --net_card ${net_card}  --model_name ${model_name0} --node_rank ${node_rank} \
--world_size ${world_size} --nprocs_list ${nprocs_list} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
--batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
--sample_interval ${sample_interval} --max_sync_num=${max_sync_num} --idx_on_gpu ${idx_on_gpu0} --job_idx ${job_idx0} --system ${system} 
#--shm_name_list=${shm_name_list} \
# & python WeaveExecutor.py --MASTER_ADDR ${MASTER_ADDR} --MASTER_PORT ${MASTER_PORT1} --net_card ${net_card}  --model_name ${model_name1} --node_rank ${node_rank} \
# --world_size ${world_size} --nprocs_list ${nprocs_list} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
# --batch_size ${batch_size} --total_epochs ${total_epochs1} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
# --sample_interval ${sample_interval} --max_sync_num=${max_sync_num} --shm_name_list=${shm_name_list} \
# --idx_on_gpu ${idx_on_gpu1} --job_idx ${job_idx1} --system ${system}
# & python WeaveExecutor.py --MASTER_ADDR ${MASTER_ADDR} --MASTER_PORT ${MASTER_PORT2} --net_card ${net_card}  --model_name ${model_name} --node_rank ${node_rank} \
# --world_size ${world_size} --nprocs_list ${nprocs_list} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
# --batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
# --sample_interval ${sample_interval} --max_sync_num=${max_sync_num} --shm_name_list=${shm_name_list} \
# --idx_on_gpu ${idx_on_gpu2} --job_idx ${job_idx2} --system ${system} 
# & python WeaveExecutor.py --MASTER_ADDR ${MASTER_ADDR} --MASTER_PORT ${MASTER_PORT3} --net_card ${net_card}  --model_name ${model_name} --node_rank ${node_rank} \
# --world_size ${world_size} --nprocs_list ${nprocs_list} --gpu_id_list ${gpu_id_list} --layer_num ${layer_num} --layer_feature ${layer_feature} \
# --batch_size ${batch_size} --total_epochs ${total_epochs} --worker_num ${worker_num} --squad_data_size ${squad_data_size} \
# --sample_interval ${sample_interval} --max_sync_num=${max_sync_num} --shm_name_list=${shm_name_list} \
# --idx_on_gpu ${idx_on_gpu3} --job_idx ${job_idx3} --system ${system}