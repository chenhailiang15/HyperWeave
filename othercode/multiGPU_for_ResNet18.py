import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms,models
import torch.optim as optim
import copy



def ddp_setup(rank, world_size):
    """
       Set up the distributed environment.

       Args:
           rank: The rank of the current process. Unique identifier for each process in the distributed training.
           world_size: Total number of processes participating in the distributed training.
       """

    # Address of the main node. Since we are doing single-node training, it's set to localhost.
    os.environ["MASTER_ADDR"] = "localhost"

    # Port on which the master node is expected to listen for communications from workers.
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group.
    # 'backend' specifies the communication backend to be used, "nccl" is optimized for GPU training.
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Set the current CUDA device to the specified device (identified by rank).
    # This ensures that each process uses a different GPU in a multi-GPU setup.
    torch.cuda.set_device(rank)





class Trainer():
    def __init__(self, model, train_data, optimizer,criterion,dataset_size, gpu_id, save_every):
        self.model = model.to(gpu_id)
        self.dataloaders = train_data
        self.optimizer = optimizer
        self.criterion=criterion
        self.dataset_sizes=dataset_size
        self.gpu_id = gpu_id
        self.save_every = save_every
        self.losses = []
        self.model = DDP(self.model, device_ids=[gpu_id])
        

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        total_loss = 0.0
        num_batches = len(self.train_data)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            total_loss += loss

        avg_loss = total_loss / num_batches
        self.losses.append(avg_loss)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    def _save_checkpoint(self, epoch):
        checkpoint = self.model.state_dict()
        PATH = f"model_{epoch}.pt"
        torch.save(checkpoint, PATH)
        print(f"Epoch {epoch} | Model saved to {PATH}")

    def train(self, max_epochs):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        for epoch in range(max_epochs):
            
            print(f'Epoch {epoch}/{max_epochs - 1}')
            print('-' * 10)
            # 每个epoch都有训练和验证阶段
            for phase in ['train']:
                if phase == 'train':
                    self.model.train()  # 设置模型为训练模式
                else:
                    self.model.eval()  # 设置模型为评估模式
                running_loss = 0.0
                running_corrects = 0

                # 迭代数据
                for inputs, labels in self.dataloaders[phase]:
                    
                    inputs = inputs.to(self.gpu_id)
                    labels = labels.to(self.gpu_id)
                    # 清除梯度
                    self.optimizer.zero_grad()

                    # 跟踪历史中的操作
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # 仅在训练阶段进行反向传播和优化
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # 统计
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # 深度拷贝模型
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            print()
        print(f'Best val Acc: {best_acc:4f}')

        # 加载最佳模型权重
        self.model.load_state_dict(best_model_wts)
        return




        # best_model_wts = copy.deepcopy(self.model.state_dict())
        # best_acc = 0.0
        # for epoch in range(max_epochs):
            
        #     print(f'Epoch {epoch}/{max_epochs - 1}')
        #     print('-' * 10)
            
        #     self.model.train()  # 设置模型为训练模式
            
        #     running_loss = 0.0
        #     running_corrects = 0

        #     # 迭代数据
        #     for inputs, labels in self.dataloaders:
        #         inputs = inputs.to(self.gpu_id)
        #         labels = labels.to(self.gpu_id)
        #         # 清除梯度
        #         self.optimizer.zero_grad()

        #         # 跟踪历史中的操作
        #         with torch.set_grad_enabled(True):
        #             outputs = self.model(inputs)
        #             _, preds = torch.max(outputs, 1)
        #             loss = self.criterion(outputs, labels)

        #             # 仅在训练阶段进行反向传播和优化
        #             loss.backward()
        #             self.optimizer.step()
        #         # 统计
        #         running_loss += loss.item() * inputs.size(0)
        #         running_corrects += torch.sum(preds == labels.data)
        #     epoch_loss = running_loss / self.dataset_sizes["train"]
        #     epoch_acc = running_corrects.double() / self.dataset_sizes["train"]
        #     print(f'{"train"} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        #     print()
        # print(f'Best val Acc: {best_acc:4f}')

        # # 加载最佳模型权重
        # self.model.load_state_dict(best_model_wts)
        # return

def load_train_objs(batch_size):
    worker_num=4
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }



    # 数据加载
    data_dir = dataset_dir+"tiny-ImageNet" # 替换为你的ImageNet数据集路径

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                        for x in ['train', 'val']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, pin_memory=True, shuffle=False,sampler=DistributedSampler(image_datasets[x]), num_workers=worker_num )
                    for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # 加载预训练的ResNet-50模型
    model = models.resnet18()
    num_ftrs = model.fc.in_features

    # 替换最后一层以适应ImageNet的类别数（1000类）
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    return dataloaders, model, optimizer,criterion,dataset_sizes

def prepare_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False,sampler=DistributedSampler(dataset))


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    """
       Main training function for distributed data parallel (DDP) setup.

       Args:
           rank (int): The rank of the current process (0 <= rank < world_size). Each process is assigned a unique rank.
           world_size (int): Total number of processes involved in the distributed training.
           save_every (int): Frequency of model checkpoint saving, in terms of epochs.
           total_epochs (int): Total number of epochs for training.
           batch_size (int): Number of samples processed in one iteration (forward and backward pass).
       """

    # Set up the distributed environment, including setting the master address, port, and backend.
    ddp_setup(rank, world_size)

    # Load the necessary training objects - dataset, model, and optimizer.
    train_data, model, optimizer,criterion,dataset_sizes = load_train_objs(batch_size)

    # Prepare the data loader for distributed training. It partitions the dataset across the processes and handles shuffling.
    # train_data = prepare_dataloader(dataset, batch_size)

    # Initialize the trainer instance with the loaded model, data, and other configurations.
    trainer = Trainer(model, train_data, optimizer,criterion,dataset_sizes, rank, save_every)

    # Train the model for the specified number of epochs.
    trainer.train(total_epochs)

    # Cleanup the distributed environment after training is complete.
    destroy_process_group()

import platform
# 创建条件变量
training_over_flage=False
dataset_dir=""
out_dir=""
sys = platform.system()
# 数据集路径
# 获取当前文件的绝对路径
current_path = os.path.abspath(os.curdir)
# 获取当前文件所在目录的上级目录
parent_dir = os.path.dirname(current_path)
default_worker_num=1
# global current_epoch_num
# current_epoch_num=0

import pynvml

if sys == "Windows":
    dataset_dir=parent_dir + '\\dataset\\'
    out_dir=parent_dir+"\\output\\"
    print("OS is Windows!!!")
elif sys == "Linux":
    dataset_dir = parent_dir + '/dataset/'
    out_dir = parent_dir + "/output/"
    default_worker_num=4
    print("OS is Linux!!!")
    pass
else:
    print("wrong!")


class Record:

    def __init__(self,gpu_id,sample_interval,out_file_name):
        print("start a record object**********************")
        self.gpu_id=gpu_id
        self.sample_interval=sample_interval
        self.out_file_name=out_file_name
        self.pynvml=pynvml.nvmlInit()


    def run(self):
        file=open(out_dir+self.out_file_name,"w")

        while training_over_flage==False:
            cpu_util=self.get_cpu_util()
            mem_util=self.get_mem_util()
            file.write(cpu_util.__str__()+","+mem_util.__str__())
            print("cpu:",cpu_util,"\tmem:",mem_util,end="")
            if self.gpu_id==-1:
                for i in range(torch.cuda.device_count()):
                    gpu_util=self.get_gpu_util_1(i)
                    gpu_mem_util=self.get_gpu_mem_util(i)
                    file.write(","+gpu_util.__str__()+","+gpu_mem_util.__str__())
                    print("\tgpu:"+i.__str__(),"-",gpu_util,"\tgmem:"+i.__str__(),"-",gpu_mem_util,end="")
            
            else:
                gpu_util=self.get_gpu_util_1(self.gpu_id)
                gpu_mem_util=self.get_gpu_mem_util(self.gpu_id)
                file.write(","+gpu_util.__str__()+","+gpu_mem_util.__str__())
                print("\tgpu:"+i.__str__(),"-",gpu_util,"\tgmem:"+i.__str__(),"-",gpu_mem_util,end="")
            file.write("\n")
            print()
            # global current_epoch_num
            file.flush()
        file.close()

    def get_cpu_util(self):
        cpu_usage=psutil.cpu_percent(interval=self.sample_interval)
        return cpu_usage


    def get_gpu_util_1(self,gpu_id):
        gpus = gpustat.GPUStatCollection.new_query()
        gpu = gpus.gpus[gpu_id]  # 获取第一个GPU的信息
        utilization = gpu.utilization  # GPU利用率
        return utilization
        # print(f"GPU Utilization 1:", utilization)

    def get_gpu_util_2(self):

        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        utilization = result.stdout.strip().split('\n')

        print(f"GPU Utilization 2:",utilization[0])

    def get_mem_util(self):
        memory=psutil.virtual_memory()
        memory_usage=memory.percent
        # print("Mem utilization:", memory_usage)
        return memory_usage
    
    
    def get_gpu_mem_util(self,gpu_id):

        gpu_device = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        # get GPU memory total
        totalMemory = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).total
        # get GPU memory used
        usedMemory = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).used
        # UtilizationRates = pynvml.nvmlDeviceGetUtilizationRates(gpu_device)

        # print(gpu_id.__str__()+"  - 总显存: {:.2f} MB".format(totalMemory / (1024 ** 2)))
        # print(gpu_id.__str__()+"  - 已分配的显存: {:.2f} MB".format(usedMemory / (1024 ** 2)))
        # print(gpu_id.__str__()+"  - 利用率: {:.2f} ".format(UtilizationRates.gpu))
        return round(usedMemory/totalMemory*100,2)
        # if torch.cuda.is_available():
        #     # 获取CUDA设备数量
        #     device_count = torch.cuda.device_count()
        #     # print("CUDA可用，共有 {} 个CUDA设备可用:".format(device_count))

        #     device = torch.device("cuda:{}".format(gpu_id))
        #     # print("CUDA 设备 {}: {}".format(i, torch.cuda.get_device_name(i)))

        #     # 获取当前设备的显存使用情况
        #     total_memory = torch.cuda.get_device_properties(device).total_memory
        #     allocated_memory = torch.cuda.memory_allocated(device)  # 已分配的显存
        #     reserved_memory = torch.cuda.memory_reserved(device)  # 已保留的显存
        #     free_memory = total_memory - allocated_memory - reserved_memory  # 剩余可用显存
        #     print(gpu_id.__str__()+"  - 总显存: {:.2f} MB".format(total_memory / (1024 ** 2)))
        #     print(gpu_id.__str__()+"  - 已分配的显存: {:.2f} MB".format(allocated_memory / (1024 ** 2)))
        #     print(gpu_id.__str__()+"  - 已保留的显存: {:.2f} MB".format(reserved_memory / (1024 ** 2)))
        #     print(gpu_id.__str__()+"  - 剩余可用显存: {:.2f} MB".format(free_memory / (1024 ** 2)))

            # return round(allocated_memory/total_memory*100,2)
        return -1

        #     reserved_memory = torch.cuda.memory_reserved(device)  # 已保留的显存
        #     free_memory = total_memory - allocated_memory - reserved_memory  # 剩余可用显存
        #     print("  - 总显存: {:.2f} GB".format(total_memory / (1024 ** 3)))
        #     print("  - 已分配的显存: {:.2f} GB".format(allocated_memory / (1024 ** 3)))
        #     print("  - 已保留的显存: {:.2f} GB".format(reserved_memory / (1024 ** 3)))
        #     print("  - 剩余可用显存: {:.2f} GB".format(free_memory / (1024 ** 3)))
        #
        # return


def Record_func(gpu_id, sample_interval,out_file_name):
    record=Record(gpu_id=gpu_id,sample_interval=sample_interval,out_file_name=out_file_name)
    record.run()

def run_multi_gpu():
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)


import argparse
import subprocess
import threading
import time
import gpustat
import psutil

parser = argparse.ArgumentParser(description='simple distributed training job')
parser.add_argument('--total_epochs', default= 10,type=int, help='Total epochs to train the model')
parser.add_argument('--save_every',default= 100, type=int, help='How often to save a snapshot')
parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 32)')
parser.add_argument("--sample_interval", default=1, type=float)
args = parser.parse_args()


if __name__=="__main__":




    import datetime
    # 获取当前时间
    now = datetime.datetime.now()
    # 格式化输出
    formatted_time = now.strftime('%m_%d_%H_%M_%S')
    device_name=''
    if torch.cuda.is_available():
        device_name=torch.cuda.get_device_name(0).replace(" ","_")
    else:
        device_name="CPU"
    model_name="multiGPU"
    batch_size=args.batch_size
    epoch=args.total_epochs
    sample_interval=args.sample_interval

    out_file_name=model_name+"-"+device_name+"-"+batch_size.__str__() +"-"+epoch.__str__() +"-"+\
        sample_interval.__str__()+"-"+formatted_time+".txt"
    print(out_file_name)

    subTread1=threading.Thread(target=Record_func,args=(-1,sample_interval,out_file_name))
    # subTread2=threading.Thread(target=run_multi_gpu)
    subTread1.start()
    time.sleep(1)
    run_multi_gpu()
    training_over_flage=True
    subTread1.join()
    print("process end!")
    # subTread2.start()