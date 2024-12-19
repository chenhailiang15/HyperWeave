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
from models.Model_ResNet_etal import ResNet_etal_class
from models.Model_Bert import Bert_class
from models.Model_GCN import GCN_class
from models.Model_GraphSage import GraphSage_class
from util import *





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
    

def single_training(local_rank,args,model):
    """
       Main training function for distributed data parallel (DDP) setup.
    """

    #设置
    os.environ["MASTER_ADDR"]=args.MASTER_ADDR
    os.environ["MASTER_PORT"]=args.MASTER_PORT
    # os.environ["NCCL_SOCKET_IFNAME"]=args.net_card
    
    # Set up the distributed environment, including setting the master address, port, and backend.
    ddp_setup(local_rank, args)
    #set device for model and data
    model.set_local_rank(local_rank)
    
    #判断要不要启动同步器
    
    model.set_shm_name( args.shm_name)
    # Load the necessary training objects - dataset, model, and optimizer.
    model.load_mode_data()
    # Train the model for the specified number of epochs.
    model.run()
    # Cleanup the distributed environment after training is complete.
    destroy_process_group()

    
def Record_resource(args, gpu_id, out_dir, out_file_name,event,queue):
    record=Record(gpu_id=gpu_id,net_card="", sample_interval=args.sample_interval,out_dir=out_dir, out_file_name=out_file_name,event=event,print_flage=args.print_flage)
    record.run_analyze(queue,args.shm_name)


def analyze_one_task(args_t,dataset_dir):
    if args_t.model_name == "ResNet18" or args_t.model_name == "ResNet50" or args_t.model_name =="AlexNet"\
        or args_t.model_name =="VGG16" or args_t.model_name =="MobileNetv2":
        model=ResNet_etal_class(args_t,dataset_dir,"analyze")
    elif args_t.model_name == "Bert":
        model=Bert_class(args_t,dataset_dir)
    elif args_t.model_name == "GCN":
        model=GCN_class(args_t,dataset_dir)
    elif args_t.model_name == "GraphSage":
        model=GraphSage_class(args_t,dataset_dir)
    else:
        print("model_name wrong!")
        exit(-1)
    mp.spawn(single_training, args=(args_t,model), nprocs=args_t.nprocs_per_node)
    return


def analyze_tasks(args,dataset_dir,queue):
    args.total_epochs=2
    # args.gpu_id_list=[0,1,2,3]
    args.node_rank=0
    model_name_list=["AlexNet","ResNet18","ResNet50","MobileNetv2","VGG16"]#"AlexNet","ResNet18","ResNet50",,"MobileNetv2"
    batch_size_list=[8,16,32,64,128,256]
    max_parrallel=4
    for model_name in model_name_list:
        for batch_size in batch_size_list:
            for parrallel in range(1,max_parrallel+1):
                # try:
                print("start analyze: ", model_name+"-"+batch_size.__str__()+"-"+parrallel.__str__())
                queue.put(model_name+"-"+batch_size.__str__()+"-"+parrallel.__str__())
                args.model_name=model_name
                args.batch_size=batch_size
                args.nprocs_per_node=parrallel
                
                analyze_one_task(args,dataset_dir)
                # except Exception:
                #     print("wrong:",model_name+"-"+batch_size.__str__()+"-"+parrallel.__str__()) 
    return


def offline_analyze():

    args=args_weave()
    
    shm_name=generate_shm_name()
    args.set_shm_name(shm_name)
    my_queue=queue.Queue()
    # args.set_queue(my_queue)
    dataset_dir=get_dataset_dir()
    output_dir=get_output_dir()
    record_file_name=generate_file_name_for_analyze()
    event=threading.Event()
    subthread_record=threading.Thread(target=Record_resource,args=(args,-1,output_dir,record_file_name,event,my_queue))
    subthread_record.start()
    analyze_tasks(args,dataset_dir,my_queue)
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
        self.expand=1.2
        
        self.data={}
        self.load_csv(file_name)
        if print_level>=1:
            print(f"AnalyzeDataLoader init over: {file_name}")
        
    def load_csv(self, file_name,header=None):
        dataset_dir=get_dataset_dir()
        file=open(dataset_dir+"cluster_exp_data"+"/"+file_name,"r")
        for line in file.readlines():
            model_info=line.split("-[(")[0]
            base_cost=np.array(ast.literal_eval(line.split("-[(")[1].split("), (")[0]))
            stage_init_cost=np.array(ast.literal_eval(line.split("-[(")[1].split("), (")[1]))-base_cost
            stage_sample_cost=np.array(ast.literal_eval(line.split("-[(")[1].split("), (")[2]))-base_cost
            stage_train_cost=np.array(ast.literal_eval(line.split("-[(")[1].split("), (")[3].split(")]")[0]))-base_cost
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

        
    
if __name__=="__main__":
    offline_analyze()
    # analyze_data=AnalyzeDataLoader("Analyzer-NVIDIA_GeForce_RTX_2080.csv")
    
    
    
    
    
    