from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import torch
import copy
import os
import torchvision
import psutil
import time
import threading
from WeaveSynchronizer import Synchronizer
import numpy as np
from models.Framework_model_CV import CVModel
from models.Framework_model_Bert import BertModel
from models.Framework_model_Transformer import TransformerModel
from models.Framework_model_GraphSage import GraphSageModel
from models.Framework_model_GCN import GCNModel





class model_framework:
    def __init__(self, local_rank, args):
        self.system=args.system
        self.mode=args.mode
        self.job_idx=args.job_idx
        self.idx_on_gpu=args.idx_on_gpu
        
        self.model_name=args.model_name
        self.local_rank=local_rank
        self.args=args
        
        if torch.cuda.is_available():
            if len(self.args.gpu_id_list)>=self.args.node_rank+1 and len(self.args.gpu_id_list[self.args.node_rank]) != 0:
                self.device = "cuda:"+self.args.gpu_id_list[self.args.node_rank][self.local_rank].__str__()
            else:
                self.device = "cuda:"+self.local_rank.__str__()
        else:
            self.device="cpu"
        print(f"load model {self.args.model_name} and data with device {self.device} ...")
        
        
        args.device=self.device
        
        
        if args.model_name == "ResNet18" or args.model_name == "ResNet50" or args.model_name =="AlexNet"\
            or args.model_name =="VGG16" or args.model_name =="MobileNetv2":
            self.model=CVModel(args)
        elif args.model_name == "Bert":
            self.model=BertModel(args)
        elif args.model_name == "Transformer":
            self.model=TransformerModel(args)
        elif args.model_name == "GraphSage":
            self.model=GraphSageModel(args)
        elif args.model_name == "GCN":
            self.model=GCNModel(args)
        
        
        else:
            print("model_name wrong!")
            exit(-1)
        
        
        
    def set_shm_name(self,shm_name, prior=True, enable_flage=True):
        if self.system == "Weave":
            if self.args.mode=="train":
                self.sync_er=Synchronizer(shm_name,prior=prior, enable_flage=enable_flage)
            elif self.args.mode == "analyze":
                self.sync_er=Synchronizer(shm_name, shm_size=4)
        elif self.system == "Muri":
            return  #等待完善
            # if self.args.mode=="train":
            #     self.sync_er=Synchronizer(shm_name,prior=prior, enable_flage=enable_flage)
            # elif self.args.mode == "analyze":
            #     self.sync_er=Synchronizer(shm_name, shm_size=4)
    
    def load_mode_data(self):
        if self.system=="Weave" and self.mode=="analyze":
            if self.local_rank==0:
                self.sync_er.set_value(0,True)
                time.sleep(10)
                self.sync_er.set_value(0,False)
                self.sync_er.set_value(1,True)
            else:
                time.sleep(10)
        
        self.model.prepare()
        
        if self.system=="Weave" and self.mode == "analyze":
            if self.local_rank==0:
                self.sync_er.set_value(1,False)
            
    def run(self):
        if self.system=="Weave":
            for epoch in range(self.args.total_epochs):
                print(f"job_idx: {self.job_idx} idx on gpu: {self.idx_on_gpu} epoch: {epoch+1}/{self.args.total_epochs}...")
                
                if self.mode == "train":
                    self.sync_er.sync_in_start_epoch(epoch==0)
                    print("model name:",self.args.model_name,"\tepoch:",epoch,"/",self.args.total_epochs-1)
                elif self.mode == "analyze":
                    if self.local_rank==0 and epoch==1:
                        self.sync_er.set_value(2,True)
                else:
                    print("model mode wrong!")
                    exit(-1)
                #**********************************************************
                self.model.sample()
                #**********************************************************
                #进行同步操作 等待信号，方可继续执行，后方代码主要利用GPU
                if self.mode == "train":
                    self.sync_er.sync_in_batch()
                elif self.mode == "analyze":
                    if self.local_rank==0 and epoch==1:
                        self.sync_er.set_value(2,False)
                        self.sync_er.set_value(3,True)
                else:
                    print("model mode wrong!")
                    exit(-1)
                    
                #**********************************************************
                self.model.train()
                #**********************************************************
                if self.mode == "train":
                    #进行同步操作（）
                    self.sync_er.sync_in_end_epoch()
                elif self.mode == "analyze":
                    if self.local_rank==0 and epoch==1:
                        self.sync_er.set_value(3,False)
                else:
                    print("model mode wrong!")
                    exit(-1)
                
        elif self.system == "Muri":
            return #等待完善
                    # self.model.prepare_sub()
                    # while not self.model.is_epoch_end():
                    #     if self.model.batch_idx%500 == 0:
                    #         print(f"split mode batch:{self.model.batch_idx}/{self.model.total_batch_num}")
                    #     data_tuple=self.model.get_data()
                    #     self.model.forward_backward(data_tuple)
                    #     self.model.comm()
        
        
        print("train over!")
        