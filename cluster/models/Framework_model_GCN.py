import torch.nn.functional as F
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv  # 从PyTorch几何库中导入图卷积网络层（GCNConv）
import math





class GCNModel:
    def __init__(self,  args):
        self.model_name=args.model_name
        self.args = args 

    def prepare(self):
        '''
        prepare dataloader, model, optimizer for training
        '''
        self.device=self.args.device
        
        dataset = Planetoid(root=self.args.dataset_dir+"/Cora_Planetoid", name='Cora')  # 加载Cora数据集

        self.model = GCN(dataset.num_node_features, dataset.num_classes, self.args.layer_num, self.args.layer_feature, self.device).to(self.device)  # 实例化GNN模型，并移动到对应设备
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)  # 定义Adam优化器

        self.train_loader=DataLoader(dataset , sampler=DistributedSampler(dataset), batch_size=self.args.batch_size)
        self.model=DDP(self.model, device_ids=[self.device],output_device=self.device)
        
        self.model.train()
        self.cur_epoch = 0
        self.total_batch_num=math.ceil(self.args.batch_num/self.args.world_size)
    
    
    def prepare_sub(self):  
        self.batch_idx = 0
        # time.sleep(2)

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
        if self.batch_idx==self.total_batch_num:
            self.cur_epoch+=1
            self.batch_idx=0
            
        data=self.train_loader.dataset[0].to(self.device)            
        self.batch_idx += 1
        
        # time.sleep(2)
        return data
    
    def forward_backward(self, data):
        '''
        forward, calculate loss and backward
        '''
        out = self.model(data)  # 前向传播，得到模型输出
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  # 计算损失，使用负对数似然损失函数
        loss.backward()  # 反向传播计算梯度
        # time.sleep(2)
        
    def comm(self):
        '''
        sync for communication
        '''
        self.optimizer.step()
        # time.sleep(2)
    
    
    def sample(self):
        self.data=self.train_loader.dataset[0].to(self.device)
        
    def train(self):
            for batch_idx in range(self.total_batch_num):
                if batch_idx%500 == 0:
                    print(f"job_idx: {self.args.job_idx} batch_idx: {batch_idx}/{self.total_batch_num}...")
                
                self.optimizer.zero_grad()  # 梯度清零
                out = self.model(self.data)  # 前向传播，得到模型输出
                loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])  # 计算损失，使用负对数似然损失函数
                loss.backward()  # 反向传播计算梯度
                self.optimizer.step()  # 更新模型参数
        
        
        
        
        
        
        
            
        
class GCN(torch.nn.Module):  # 定义一个GCN类，继承自PyTorch的Module类
    def __init__(self,num_node_features, num_classes,layer_num, layer_feature,device):  # 定义GNN类的初始化函数
        if layer_num<2:
            print("layer_num should no less than 2")
            exit(-1)
        super().__init__()  # 调用父类（Module类）的初始化函数
        # 创建第一个图卷积层，输入特征维度为数据集节点特征维度，输出特征维度为16
        self.conv1 = GCNConv(num_node_features, layer_feature)
        self.middel_conv=[]
        for i in range(layer_num-2):
            self.middel_conv.append(GCNConv(layer_feature,layer_feature).to(device))
        # 创建第二个图卷积层，输入特征维度为16，输出特征维度为数据集类别数量
        self.conv2 = GCNConv(layer_feature, num_classes)
        self.layer_num=layer_num
        self.layer_feature=layer_feature

    def forward(self, data):  # 定义前向传播函数，接受一个数据对象作为输入
        x, edge_index = data.x, data.edge_index  # 从数据对象中获取节点特征和边索引
        x = self.conv1(x, edge_index)  # 通过第一个图卷积层处理节点特征
        x = F.relu(x)  # 对输出进行ReLU激活函数操作
        for i in range(self.layer_num-2):
            x=self.middel_conv[i](x,edge_index)
            x=F.relu(x)
        x = F.dropout(x, training=self.training)  # 对输出进行Dropout操作，用于防止过拟合
        x = self.conv2(x, edge_index)  # 通过第二个图卷积层处理节点特征
        return F.log_softmax(x, dim=1)  # 对输出进行LogSoftmax操作，得到预测结果