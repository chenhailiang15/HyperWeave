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
    def __init__(self,master, strategy,print_level=0):
        
        self.master=master
        self.strategy=strategy
        self.gpu_list=[i for i in range(7)]
        self.print_level=print_level
        
    
    def do_schedule(self,job_list):
        if self.strategy=="random":
            rest_job=self.schedule_random(job_list)
        else:
            print("strategy wrong!")
            exit(-1)
        return rest_job
    
    def schedule_random(self, job_list):
        if self.print_level>2:
            print("start schedule ...")
        for job in job_list:
            gpu_num=math.ceil(job.plan_gpu)
            select_gpu=random.sample(self.gpu_list, gpu_num)
            self.execute_schedule(job,select_gpu)
        return []
    
    def schedule_weave_over_sharing(self):

        return []
    
    
    
    
    
    
    
    
    
    
    def execute_schedule(self,job, select_gpu_list):
        world_size=len(select_gpu_list)
        master_gpu_id_list=[x for x in select_gpu_list if x <4]
        worker_gpu_id_list=[x-4 for x in select_gpu_list if x >= 4]
        
        nprocs_list=[len(master_gpu_id_list), len(worker_gpu_id_list)]
        gpu_id_list=[master_gpu_id_list,worker_gpu_id_list]

        is_cross=True if len(master_gpu_id_list)>0 and len(worker_gpu_id_list)>0 else False

        if len(master_gpu_id_list)>0:
            self.job_set_execute_info(job, True, is_cross, world_size, nprocs_list, gpu_id_list )
            self.master.execute_job_in_master(job)
            
        if len(worker_gpu_id_list)>0:
            self.job_set_execute_info(job, False, is_cross, world_size, nprocs_list, gpu_id_list )
            self.master.send_job_to_worker(job)
            
    def job_set_execute_info(self, job, is_master, is_cross, world_size, nprocs_list, gpu_id_list):
        if is_master:
            node_rank=0
            net_card="eno2"
        else:
            node_rank=1
            net_card="eno1"
        

        if is_cross and is_master:
            while is_port_in_use(self.master.master_port):
                if self.print_level>9:
                    print("change port")
                self.master.master_port+=1


        if is_cross or is_master:
            MASTER_ADDR="10.26.128.115"
            if is_master:
                MASTER_PORT=self.master.master_port
                self.master.master_port+=1
            else:
                MASTER_PORT=self.master.master_port-1
            
        elif (not is_cross) and (not is_master):
            MASTER_ADDR="10.26.128.51"
            MASTER_PORT=self.master.worker_port
            self.master.worker_port+=1
        
            
            
        job.set_execute_info(MASTER_ADDR, MASTER_PORT, net_card, node_rank,world_size ,nprocs_list, gpu_id_list)
        
    
    
    
    
    