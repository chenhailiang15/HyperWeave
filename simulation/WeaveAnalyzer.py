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
from util import *

    
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
        # if print_level>=1:
        #     print(f"AnalyzeDataLoader init over: {file_name}")
        
    def load_csv(self, file_name,header=None):
        dataset_dir=get_dataset_dir()
        file=open(dataset_dir+"exp_data"+"/"+file_name,"r")
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

        
    def load_time_csv(self, file_name):
        self.time_data = {}
        dataset_dir=get_dataset_dir()
        file=open(dataset_dir+"exp_data"+"/"+file_name,"r")
        for line in file.readlines():
            model_info=line.split("-[")[0]
            # base_cost=np.array(ast.literal_eval(line.split("-[(")[1].split("), (")[0]))
            stage0_time=float(line.split("-[")[1].split(",")[0])
            stage1_time=float(line.split("-[")[1].split(",")[1])
            stage2_time=float(line.split("-[")[1].split(",")[2])
            stage3_time=float(line.split("-[")[1].split(",")[3].split("]")[0])
            
            
            self.time_data[model_info]=[stage0_time, stage1_time, stage2_time, stage3_time]
    
    def get_time_value(self, model_info, index):
        try:
            value=self.time_data[model_info][index]
        except:
            print("model info is wrong!")
            exit(-1)
            
        return value
    
    def get_time_all(self, model_info):
        return [self.get_time_value(model_info,0), self.get_time_value(model_info,1), self.get_time_value(model_info,2), self.get_time_value(model_info,3)]
    
if __name__=="__main__":
    
    # offline_analyze("Muri")
    analyze_data=AnalyzeDataLoader("Analyzer-NVIDIA_GeForce_RTX_2080.csv")
    
    
    
    
    
    