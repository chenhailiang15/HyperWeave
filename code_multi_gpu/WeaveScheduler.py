import math
import random
import os
import threading
from util import *


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


class WeaveSchedulor:
    def __init__(self,master, command_sender, strategy):
        self.master=master
        self.command_sender=command_sender
        self.strategy=strategy
        self.gpu_list=[i for i in range(7)]
        
    
    def do_schedule(self,job_list):
        if self.strategy=="random":
            rest_job=self.schedule_random(job_list)
        else:
            print("strategy wrong!")
            exit(-1)
        return rest_job
    
    def schedule_random(self, job_list):
        for job in job_list:
            gpu_num=math.ceil(job[4])
            select_gpu=random.sample(self.gpu_list, gpu_num)
            self.master.execute_schedule(job[1],job[2],job[3],select_gpu)
        return []
    
    
    
    
    
    
    
    