import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
# import torch.profiler
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
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




class CVModel:
    def __init__(self,  args):
        self.idx = args.idx
        self.args = args 
    
    def prepare(self):
        '''
        prepare dataloader, model, optimizer for training
        '''
        
            
        self.device=self.args.device
        print(f"CVModel device {self.device}")
        
        data_dir = self.args.dataset_dir + "tiny-ImageNet"
        
        # train_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
        #                         transform=transforms.Compose([
        #                         transforms.RandomResizedCrop(224),
        #                         transforms.RandomHorizontalFlip(),
        #                         transforms.ToTensor(),
        #                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                             std=[0.229, 0.224, 0.225])
        #                     ]))
        #                   for x in ['train', 'val']}

        # self.dataloaders = {x: DataLoader(train_dataset[x], batch_size=self.args.batch_size, pin_memory=True, shuffle=False, \
        #     sampler=DistributedSampler(train_dataset[x]) , num_workers=self.args.worker_num)
        #                     for x in ['train', 'val']}
        
        
        train_dataset = \
            datasets.ImageFolder(os.path.join(data_dir, "train"),
                            transform=transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ]))

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size, pin_memory=True, shuffle=False,
            sampler=self.train_sampler, num_workers=self.args.worker_num)

        num_classes=len(train_dataset.classes)
        self.model = getattr(models, "alexnet")(num_classes=num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        self.model = self.model.to(self.device)

        print("start ddp model..." )
        self.model = DDP(self.model, device_ids=[self.device], output_device=self.device)
        print("end ddp model...")
        
        self.model.train()
        self.cur_epoch = 0
    
    
    def prepare_sub(self):  
        self.dataloader_iter = iter(self.train_loader)
        self.batch_idx = -1


    def get_data(self):
        '''
        get data
        '''
        try:
            data,target = next(self.dataloader_iter)
        except StopIteration:
            self.cur_epoch += 1
            self.train_sampler.set_epoch(self.cur_epoch)
            self.dataloader_iter = iter(self.train_loader)
            data,target = next(self.dataloader_iter)
            self.batch_idx = -1
        self.batch_idx +=1
        
        return data,target
    
    def forward_backward(self, thread):
        '''
        forward, calculate loss and backward
        '''
        data, target = thread.get_result()
        if self.args.cuda:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
        
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
        self.dataloader_iter = iter(self.train_loader)
        self.cur_epoch +=1
        
        
    def train(self):
        batch_num=len(self.train_loader)
        print(f"batch num:{batch_num}")
        while True:
            try:
                data,target = next(self.dataloader_iter)
                data=data.to(self.device)
                target=target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss =  self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
            except StopIteration:
                break
            
        
        
        
        
    
    def print_info(self):
        print("Model ", self.idx, ": ", self.sargs["model_name"], "; batch size: ", self.sargs["batch_size"])

    def data_size(self):
        # each image is 108.6kb on average
        return self.sargs["batch_size"] * 108.6