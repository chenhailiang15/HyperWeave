import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.optim as optim
import torch
import os


cv_model_dict={"AlexNet": "alexnet",
               "ResNet18": "resnet18",
               "ResNet50": "resnet50",
               "VGG16": "vgg16",
               "MobileNetv2": "mobilenet_v2"
               }


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

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size, pin_memory=True, shuffle=False,
            sampler=self.train_sampler, num_workers=self.args.worker_num)
        
        # num_classes=len(train_dataset.classes)
        class_names=train_dataset.classes
        # self.model = getattr(models, cv_model_dict[self.model_name])(num_classes=num_classes)
        
        # 加载预训练的ResNet-18,50模型
        if self.args.model_name == "ResNet18":
            self.model = models.resnet18()
            num_ftrs = self.model.fc.in_features
            # 替换最后一层以适应ImageNet的类别数（1000类）
            self.model.fc = nn.Linear(num_ftrs, len(class_names))
        elif self.args.model_name == "ResNet50":
            self.model = models.resnet50()
            num_ftrs = self.model.fc.in_features
            # 替换最后一层以适应ImageNet的类别数（1000类）
            self.model.fc = nn.Linear(num_ftrs, len(class_names))
        elif self.args.model_name == "AlexNet":
            self.model = models.alexnet()
            num_fc = self.model.classifier[6].in_features
            self.model.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=len(class_names))
        elif self.args.model_name =="MobileNetv2":
            self.model = models.mobilenet_v2()
            # 替换最后一层以适应ImageNet的类别数（1000类）
            self.model.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(class_names))
            )
        elif self.args.model_name =="VGG16":
            self.model = models.vgg16()
            # 替换最后一层以适应ImageNet的类别数（1000类）
            num_fc = self.model.classifier[6].in_features  # 获取最后一层的输入维度
            self.model.classifier[6] = torch.nn.Linear(num_fc, len(class_names))  # 修改最后一层的输出维度，即分类
        else:
            print("model name wrong!")
            exit(-1)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        self.model = self.model.to(self.device)

        print("start ddp model..." )
        self.model = DDP(self.model, device_ids=[self.device], output_device=self.device)
        print("end ddp model...")
        
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
            self.train_sampler.set_epoch(self.cur_epoch)
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
        return
        # self.cur_epoch +=1
        # self.train_sampler.set_epoch(self.cur_epoch)
        # self.dataloader_iter = iter(self.train_loader)
        
        
        
    def train(self):
        self.cur_epoch +=1
        batch_idx=0
        for data, target in self.train_loader:
            if batch_idx%500 == 0:
                print(f"job_idx: {self.args.job_idx} batch_idx: {batch_idx}/{self.total_batch_num}...")
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output = self.model(data)
                loss =  self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            batch_idx+=1
            
            
            
        # batch_idx=0
        # while True:
        #     try:
        #         if batch_idx%500 == 0:
        #             print(f"job_idx: {self.args.job_idx} batch_idx: {batch_idx}/{self.total_batch_num}...")
                
        #         data,target = next(self.dataloader_iter)
        #         data=data.to(self.device)
        #         target=target.to(self.device)
                
        #         self.optimizer.zero_grad()
        #         output = self.model(data)
        #         loss =  self.criterion(output, target)
        #         loss.backward()
        #         self.optimizer.step()
        #         batch_idx+=1
        #     except StopIteration:
                # break
            
        
