#!/usr/bin/env python
## -*- coding: utf-8 -*-
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import os
import torch
import argparse
import threading
import time
import datetime
import ast
import queue
import pandas as pd
import numpy as np


from Recorder import Record
from models.Framework import model_framework
from util import *
from WeaveSynchronizer import Synchronizer




def ddp_setup(local_rank, args):
    """
       Set up the distributed environment.
    """
    node_rank=args.node_rank
    nprocs_per_node=args.nprocs_per_node
    global_rank = node_rank * nprocs_per_node + local_rank
    # Initialize the process group.
    # 'backend' specifies the communication backend to be used, "nccl" is optimized for GPU training.
    print("world size:",nprocs_per_node * args.nnodes)
    init_process_group(backend="nccl", rank=global_rank, world_size=nprocs_per_node* args.nnodes)

    # Set the current CUDA device to the specified device (identified by rank).
    # This ensures that each process uses a different GPU in a multi-GPU setup.
    
    torch.cuda.set_device(local_rank)
    

def single_training(local_rank,args):
    """
       Main training function for distributed data parallel (DDP) setup.
    """
    model=model_framework(local_rank,args)
    #设置
    os.environ["MASTER_ADDR"]=args.MASTER_ADDR
    os.environ["MASTER_PORT"]=args.MASTER_PORT
    # os.environ["NCCL_SOCKET_IFNAME"]=args.net_card
    
    # Set up the distributed environment, including setting the master address, port, and backend.
    ddp_setup(local_rank, args)

    #判断要不要启动同步器
    
    model.init_sync_er()
    # Load the necessary training objects - dataset, model, and optimizer.
    model.load_mode_data()
    # Train the model for the specified number of epochs.
    model.run()
    # Cleanup the distributed environment after training is complete.
    destroy_process_group()

    
def Record_resource(args, gpu_id, out_dir, out_file_name,event,queue):
    record=Record(gpu_id=gpu_id,net_card="", sample_interval=args.sample_interval,out_dir=out_dir, out_file_name=out_file_name,event=event,print_flage=args.print_flage)
    record.run_analyze(queue,args.shm_name)




def analyze_tasks(args,dataset_dir,queue,sync=None):
    if args.system=="Muri":
        file_writer=open(args.muri_file_path_name, "w")
    args.total_epochs=2
    # args.gpu_id_list=[0,1,2,3]
    args.node_rank=0
    args.dataset_dir=dataset_dir
    
    model_name_list=["AlexNet","ResNet18","ResNet50","MobileNetv2","VGG16", "GCN", "GraphSage","Transformer", "Bert"]#"AlexNet","ResNet18","ResNet50","MobileNetv2","VGG16"
    max_parrallel=4
    for model_name in model_name_list:
        for batch_size in model_to_batch_size_g[model_name]:
            for parrallel in range(1,max_parrallel+1):
                if model_name =="GCN":
                    args.layer_num=100
                    args.layer_feature=100
                    args.batch_num=100
                    
                # try:
                print("start analyze: ", model_name+"-"+batch_size.__str__()+"-"+parrallel.__str__())
                if args.system=="Weave":
                    queue.put(model_name+"-"+batch_size.__str__()+"-"+parrallel.__str__())
                args.model_name=model_name
                args.batch_size=batch_size
                args.nprocs_per_node=parrallel
                mp.spawn(single_training, args=(args,), nprocs=args.nprocs_per_node)
                if args.system=="Muri":
                    out_line=f"{model_name}-{batch_size}-{parrallel}-[{sync.get_value(0)},{sync.get_value(1)},{sync.get_value(2)},{sync.get_value(3)}]"
                    file_writer.write(out_line+"\n")
                    file_writer.flush()
                # print(f"time:{sync.get_value(0)},{sync.get_value(1)},{sync.get_value(2)},{sync.get_value(3)}")
                sync.set_value(0,0)
                sync.set_value(1,0)
                sync.set_value(2,0)
                sync.set_value(3,0)
                
                # try:

                #     mp.spawn(single_training, args=(args,), nprocs=args.nprocs_per_node)
                # except:
                    
                #     print("parameter is in appropriate!")
                
    return


def offline_analyze(system):

    args=args_weave()
    args.system=system
    args.mode="analyze"
    args.net_card="eno1"
    shm_name=generate_shm_name()
    args.shm_name_list={0:{0:shm_name}}
    my_queue=queue.Queue()
    # args.set_queue(my_queue)
    dataset_dir=get_dataset_dir()
    output_dir=get_output_dir()
    record_file_name=generate_file_name_for_analyze()
    if args.system == "Weave":
        event=threading.Event()
        subthread_record=threading.Thread(target=Record_resource,args=(args,-1,output_dir,record_file_name,event,my_queue))
        subthread_record.start()
        sync_er=None
    elif args.system == "Muri":
        args.muri_file_path_name=output_dir+"/Muri_"+record_file_name
        sync_er=Synchronizer(shm_name, shm_size=16)
    #*************************************************
    analyze_tasks(args,dataset_dir,my_queue,sync_er)
    #*************************************************

    if args.system == "Weave":
        event.set()
        subthread_record.join()
    
        
    print("process end!")
    
    
class AnalyzeDataLoader:
    def __init__(self,file_name, print_level=0):
        self.index={}
        self.index["cpu"]=0
        self.index["mem"]=1
        self.index["gpu"]=2
        self.index["gmem"]=3
        self.index["time"]=4
        self.expand=1
        
        self.data={}
        self.load_csv(file_name)
        if print_level>=1:
            print(f"AnalyzeDataLoader init over: {file_name}")
        
    def load_csv(self, file_name,header=None):
        dataset_dir=get_dataset_dir()
        file=open(dataset_dir+"cluster_exp_data"+"/"+file_name,"r")
        for line in file.readlines():
            model_info=line.split("-[(")[0]
            # base_cost=np.array(ast.literal_eval(line.split("-[(")[1].split("), (")[0]))
            stage_init_cost=np.array(ast.literal_eval(line.split("-[(")[1].split("), (")[1]))
            stage_sample_cost=np.array(ast.literal_eval(line.split("-[(")[1].split("), (")[2]))
            stage_train_cost=np.array(ast.literal_eval(line.split("-[(")[1].split("), (")[3].split(")]")[0]))
            temp_dict={}
            temp_dict["stage_init"]=stage_init_cost
            temp_dict["stage_sample"]=stage_sample_cost
            temp_dict["stage_train"]=stage_train_cost
            self.data[model_info]=temp_dict
            
    def get_value(self, model_info, stage_info="stage_train", resource_kind="gpu"):
        return self.data[model_info][stage_info][self.index[resource_kind]]*self.expand
    
    def get_job_value(self, job, stage_info, resource_kind):
        return self.data[job.get_name_batchsize_epoch()][stage_info][self.index[resource_kind]]*self.expand
    
    
    def get_job_values(self, job, stage_info):
        cpu=self.get_job_value(job,stage_info, "cpu")
        mem=self.get_job_value(job,stage_info, "mem")
        gpu=self.get_job_value(job,stage_info, "gpu")
        gmem=self.get_job_value(job,stage_info, "gmem")
        time=self.get_job_value(job,stage_info, "time")
        return [cpu, mem, gpu, gmem, time]
    
    def get_job_pack_resource(self, job):
        [cpu0, mem0, gpu0, gmem0,time0]=self.get_job_values(job,"stage_init")
        [cpu1, mem1, gpu1, gmem1,time1]=self.get_job_values(job,"stage_sample")
        [cpu2, mem2, gpu2, gmem2,time2]=self.get_job_values(job,"stage_train")
        pack_resource=[max(cpu0,cpu1,cpu2), max(mem0, mem1,mem2), max(gpu0, gpu1, gpu2), max(gmem0, gmem1, gmem2)]
        return pack_resource

        
    
if __name__=="__main__":
    
    offline_analyze("Muri")
    # analyze_data=AnalyzeDataLoader("Analyzer-NVIDIA_GeForce_RTX_2080.csv")
    
    
    
    
    
    