import torch
import torch.nn.functional as F
import torch.optim as optim
# import torch.utils.data.distributed
from torchvision import datasets, transforms, models
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import torch
import copy
import os
import torchvision
import psutil
import time
import threading
from WeaveSynchronizer import Synchronizer
import numpy as np




class CVModel:
    def __init__(self,  args):
        self.model_name=args.model_name
        self.args = args 

    def prepare(self):
        '''
        prepare dataloader, model, optimizer for training
        '''
        self.device=self.args.device
        data_dir = self.args.dataset_dir + "/tiny-ImageNet"
        
        train_dataset = \
            datasets.ImageFolder(os.path.join(data_dir, "train"),
                            transform= transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ])
                            )

        # self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size, pin_memory=True, shuffle=False,
             num_workers=self.args.worker_num)

        num_classes=len(train_dataset.classes)
        self.model = getattr(models, "alexnet")(num_classes=num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        self.model = self.model.to(self.device)

        # print("start ddp model..." )
        # self.model = DDP(self.model, device_ids=[self.device], output_device=self.device)
        # print("end ddp model...")
        
        self.model.train()
        self.cur_epoch = 0
        self.total_batch_num=len(self.train_loader)
    
    
    def prepare_sub(self):  
        self.dataloader_iter = iter(self.train_loader)
        self.batch_idx = 0

    # def is_epoch_end(self):
    #     if self.batch_idx==self.total_batch_num-1:
    #         return True
    #     else:
    #         return False
        
    def is_end(self):
        if self.cur_epoch==self.args.total_epochs-1 and self.batch_idx==self.total_batch_num:
            return True
        else:
            return False
        
    def get_data(self):
        '''
        get data
        '''
        try:
            data,target = next(self.dataloader_iter)
        except StopIteration:
            self.cur_epoch+=1
            # self.train_sampler.set_epoch(self.cur_epoch)
            self.dataloader_iter = iter(self.train_loader)
            data,target = next(self.dataloader_iter)
            self.batch_idx=0
            
        self.batch_idx +=1
        
        return (data,target)
    
    def forward_backward(self, data_tuple):
        '''
        forward, calculate loss and backward
        '''
        
        (data, target)=data_tuple
        data = data.to(self.device)
        target = target.to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

    def comm(self):
        '''
        sync for communication
        '''
        self.optimizer.step()
    
    
    
    def sample(self):
        self.cur_epoch +=1
        # self.train_sampler.set_epoch(self.cur_epoch)
        self.dataloader_iter = iter(self.train_loader)
        
        
        
    def train(self):

        batch_idx=0
        while True:
            try:
                if batch_idx%500 == 0:
                    print(f"job_idx: {self.args.job_idx} batch_idx: {batch_idx}/{self.total_batch_num}...")
                
                data,target = next(self.dataloader_iter)
                data=data.to(self.device)
                target=target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss =  self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                batch_idx+=1
            except StopIteration:
                break
            
        
