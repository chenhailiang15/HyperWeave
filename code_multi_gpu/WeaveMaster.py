import time
from NodeCommunicate import NodeMessageSender
import json
import secrets
import string
import os
import threading
from util import *


 


# if __name__=="__main__":
#     worker_ip="10.26.128.51"
#     worker_port=8000
#     node_message_sender=NodeMessageSender(worker_ip,worker_port)
#     while True:
#         send_info=input("发送数据：")
#         node_message_sender.send(send_info)
        
        

        
''' how to record a shedule
model para:






'''

def get_strategy():
    strategy_all={}
    ####################第一个任务信息
    task_first={}
    task_first["MASTER_ADDR"]="10.26.128.51"
    task_first["MASTER_PORT"]=12355
    task_first["nnodes"]=1
    task_first["nprocs_per_node"]=2
    
    task_first["model_name"]="AlexNet"
    task_first["total_epochs"]=3
    task_first["batch_size"]=16
    task_first["worker_num"]=4
    
    task_first["max_sync_num"]=2
    
    
    #master特定信息
    task_first["master_spec"]={}
    task_first["master_spec"]["net_card"]="eno1"
    task_first["master_spec"]["node_rank"]=0
    task_first["master_spec"]["gpu_id_list"]=[1,2]
    #worker特定信息
    task_first["worker_spec"]={}
    task_first["worker_spec"]["net_card"]="eno1"
    task_first["worker_spec"]["node_rank"]=1
    task_first["worker_spec"]["gpu_id_list"]=[2,3]
    
    ####################第二个任务信息
    task_second={}
    task_second["MASTER_ADDR"]="10.26.128.51"
    task_second["MASTER_PORT"]=12345
    task_second["nnodes"]=1
    task_second["nprocs_per_node"]=2
    
    task_second["model_name"]="ResNet18"
    task_second["total_epochs"]=3
    task_second["batch_size"]=16
    task_second["worker_num"]=4
    
    task_second["max_sync_num"]=2
    #master特定信息
    task_second["master_spec"]={}
    task_second["master_spec"]["net_card"]="eno1"
    task_second["master_spec"]["node_rank"]=0
    task_second["master_spec"]["gpu_id_list"]=[1,2]
    #worker特定信息
    task_second["worker_spec"]={}
    task_second["worker_spec"]["net_card"]="eno1"
    task_second["worker_spec"]["node_rank"]=1
    task_second["worker_spec"]["gpu_id_list"]=[2,3]
    
    ##############共同共享内存名
    task_first["shm_name_list"]=[]
    task_second["shm_name_list"]=[]
    for i in range(task_first["max_sync_num"]):
        shm_name=generate_shm_name(16)
        task_first["shm_name_list"].append(shm_name)
        task_second["shm_name_list"].append(shm_name)
        
    strategy_all["task_first"]=task_first
    strategy_all["task_second"]=task_second
    return strategy_all
def get_strategy2():
    strategy_all={}
    ####################第一个任务信息
    task_first={}
    task_first["MASTER_ADDR"]="10.26.128.51"
    task_first["MASTER_PORT"]=12365
    task_first["nnodes"]=1
    task_first["nprocs_per_node"]=2
    
    task_first["model_name"]="AlexNet"
    task_first["total_epochs"]=3
    task_first["batch_size"]=16
    task_first["worker_num"]=4
    
    task_first["max_sync_num"]=2
    
    
    #master特定信息
    task_first["master_spec"]={}
    task_first["master_spec"]["net_card"]="eno1"
    task_first["master_spec"]["node_rank"]=0
    task_first["master_spec"]["gpu_id_list"]=[1,2]
    #worker特定信息
    task_first["worker_spec"]={}
    task_first["worker_spec"]["net_card"]="eno1"
    task_first["worker_spec"]["node_rank"]=1
    task_first["worker_spec"]["gpu_id_list"]=[2,3]
    
    ####################第二个任务信息
    task_second={}
    task_second["MASTER_ADDR"]="10.26.128.51"
    task_second["MASTER_PORT"]=12375
    task_second["nnodes"]=1
    task_second["nprocs_per_node"]=2
    
    task_second["model_name"]="ResNet18"
    task_second["total_epochs"]=3
    task_second["batch_size"]=16
    task_second["worker_num"]=4
    
    task_second["max_sync_num"]=2
    #master特定信息
    task_second["master_spec"]={}
    task_second["master_spec"]["net_card"]="eno1"
    task_second["master_spec"]["node_rank"]=0
    task_second["master_spec"]["gpu_id_list"]=[1,2]
    #worker特定信息
    task_second["worker_spec"]={}
    task_second["worker_spec"]["net_card"]="eno1"
    task_second["worker_spec"]["node_rank"]=1
    task_second["worker_spec"]["gpu_id_list"]=[2,3]
    
    ##############共同共享内存名
    task_first["shm_name_list"]=[]
    task_second["shm_name_list"]=[]
    for i in range(task_first["max_sync_num"]):
        shm_name=generate_shm_name(16)
        task_first["shm_name_list"].append(shm_name)
        task_second["shm_name_list"].append(shm_name)
        
    strategy_all["task_first"]=task_first
    strategy_all["task_second"]=task_second
    return strategy_all


def parameter_analyse(strategy,specific,prior=False):
    MASTER_ADDR=strategy["MASTER_ADDR"]
    MASTER_PORT=strategy["MASTER_PORT"]
    
    node_rank=strategy[specific]["node_rank"]
    nnodes=strategy["nnodes"]
    nprocs_per_node=strategy["nprocs_per_node"]
    gpu_id_list=strategy[specific]["gpu_id_list"]
    gpu_id_list=f"{gpu_id_list}".replace(" ", "")
    
    model_name=strategy["model_name"]
    batch_size=strategy["batch_size"]
    total_epochs=strategy["total_epochs"]
    worker_num=strategy["worker_num"]
    
    max_sync_num=strategy["max_sync_num"]
    shm_name_list=strategy["shm_name_list"]
    shm_name_list=f"{shm_name_list}".replace(" ", "")
    net_card=strategy[specific]["net_card"]
    
    command=f"python WeaveExecutor.py --model_name {model_name} --node_rank {node_rank} --nnodes {nnodes}  --nprocs_per_node {nprocs_per_node} \
        --gpu_id_list {gpu_id_list}  --batch_size {batch_size} --total_epochs {total_epochs} --worker_num {worker_num} --max_sync_num {max_sync_num}\
        --shm_name_list {shm_name_list} --net_card {net_card} --MASTER_ADDR {MASTER_ADDR} --MASTER_PORT {MASTER_PORT}"
    if prior:
        command=command+" --prior"
    return command
    
    
    
def run_command(command):
      os.system(command)
    

def execution_local(strategy_all):
    command_taskfirst=parameter_analyse(strategy_all["task_first"],"master_spec",prior=True)
    command_tasksecond=parameter_analyse(strategy_all["task_second"], "master_spec",)
    print(command_taskfirst)
    print(command_tasksecond)
    thread1=threading.Thread(target=run_command,args=(command_taskfirst,))
    thread1.start()
    thread2=threading.Thread(target=run_command,args=(command_tasksecond,))
    thread2.start()
    
    
    # os.system(command_taskfirst+" & "+command_tasksecond)
    # os.system(command_tasksecond)
    return thread1,thread2
    
    
    
# worker_ip="10.26.128.51"
# worker_port=8000
# node_message_sender=NodeMessageSender(worker_ip,worker_port)
if __name__=="__main__":
    start_time=time.time()
    strategy_all=get_strategy()
    # node_message_sender.send(json.dumps(strategy_all))
    thread11,thread12=execution_local(strategy_all)
    strategy_all=get_strategy2()
    thread21,thread22=execution_local(strategy_all)
    
    
    
    thread11.join()
    thread12.join()
    thread21.join()
    thread22.join()
    end_time=time.time()
    
    print(f"total time:{round(end_time-start_time,2)}")
    
    
         
