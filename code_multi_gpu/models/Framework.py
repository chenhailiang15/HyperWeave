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
from models.Framework_model_cv import CVModel









class model_framework:
    def __init__(self, local_rank, args):
        
        self.local_rank=local_rank
        self.args=args
        
        if torch.cuda.is_available():
            if len(self.args.gpu_id_list[self.args.node_rank]) != 0:
                self.device = "cuda:"+self.args.gpu_id_list[self.args.node_rank][self.local_rank].__str__()
            else:
                self.device = "cuda:"+self.local_rank.__str__()
        else:
            self.device="cpu"
        print(f"load model and data with device {self.device} ...")
        
        
        args.device=self.device
        
        
        if args.model_name == "ResNet18" or args.model_name == "ResNet50" or args.model_name =="alexnet"\
            or args.model_name =="VGG16" or args.model_name =="MobileNetv2":
            self.model=CVModel(args)
        # elif args_t.model_name == "Bert":
        #     model=Bert_class(args_t,dataset_dir)
        # # elif args_t.model_name == "GCN":
        # #     model=GCN_class(args_t,dataset_dir)
        # elif args_t.model_name == "GraphSage":
        #     model=GraphSage_class(args_t,dataset_dir)
        # elif args_t.model_name == "Transformer":
        #     model=Transformer_class(args_t,dataset_dir)
        else:
            print("model_name wrong!")
            exit(-1)
        
        
        
    def set_shm_name(self,shm_name, prior=True, enable_flage=True):
        if self.args.mode=="train":
            self.sync_er=Synchronizer(shm_name,prior=prior, enable_flage=enable_flage)
        elif self.args.mode == "analyze":
            self.sync_er=Synchronizer(shm_name, shm_size=4)
    
    def load_mode_data(self):
        
        self.model.prepare()
            
    def run(self):
        for epoch in range(self.args.total_epochs):
            print(f"epoch num: {epoch}...")
            time_start=time.time()
            self.model.sample()
            time_end=time.time()
            print(f"sample time:{time_end-time_start}")
            print(f"start train...")
            self.model.train()
        
        