from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import copy
import os


class ResNet_etal_class:
    def __init__(self, args_t, dataset_dir):
        self.args = args_t
        self.dataset_dir = dataset_dir

    def set_local_rank(self, local_rank):
        self.local_rank = local_rank

    def load_mode_data(self):
        print("start load_mode_data")
        if torch.cuda.is_available():
            if len(self.args.gpu_id_list) != 0:
                self.device = "cuda:"+self.args.gpu_id_list[self.local_rank].__str__()
            else:
                self.device = "cuda:"+self.local_rank.__str__()
        else:
            self.device="cpu"
        print("设备是：",self.device)

        worker_num = self.args.worker_num
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # 数据加载
        data_dir = self.dataset_dir + "tiny-ImageNet"  # 替换为你的ImageNet数据集路径

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val']}

        self.dataloaders = {x: DataLoader(image_datasets[x], batch_size=self.args.batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(image_datasets[x]), num_workers=worker_num)
                            for x in ['train', 'val']}

        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

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

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)

        
        self.model = self.model.to(self.device)

        print("device :" + self.device)
        self.model = DDP(self.model, device_ids=[self.device], output_device=self.device)
        print("model init end")

    def run(self):
        max_epochs = self.args.total_epochs
        print("in trainning")
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
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
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