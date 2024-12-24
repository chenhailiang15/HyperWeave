from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch.utils.data import TensorDataset,DataLoader, RandomSampler
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv  # 从PyTorch几何库中导入图卷积网络层（GCNConv）
import torch.nn.functional as F  # 导入PyTorch中的函数模块，通常用于激活函数、损失函数等操作
from WeaveSynchronizer import Synchronizer


class GCN_class:
    
    def __init__(self, args_t,dataset_dir):
        self.args=args_t
        self.dataset_dir=dataset_dir
        
    def set_local_rank(self,local_rank):
        self.local_rank=local_rank
    def set_shm_name(self,prior,shm_name,enable_flage=True):
        self.sync_er=Synchronizer(shm_name,prior=prior, enable_flage=enable_flage)

    def load_mode_data(self):
        if torch.cuda.is_available():
            if len(self.args.gpu_id_list) != 0:
                self.device = "cuda:"+self.args.gpu_id_list[self.args.node_rank][self.local_rank].__str__()
            else:
                self.device = "cuda:"+self.local_rank.__str__()
        else:
            self.device="cpu"
        print("设备是：",self.device)
        

        # 数据加载和模型训练部分：
        dataset = Planetoid(root=self.dataset_dir+"Corakk", name='Cora')  # 加载Cora数据集

        self.model = GCN(dataset.num_node_features, dataset.num_classes, self.args.layer_num, self.args.layer_feature, self.device).to(self.device)  # 实例化GNN模型，并移动到对应设备
        self.data = dataset[0].to(self.device)  # 获取数据集的第一个图数据，并移动到对应设备
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)  # 定义Adam优化器

        self.data=DataLoader(self.data , sampler=DistributedSampler(self.data ), batch_size=self.args.batch_size)
        self.model=DDP(self.model, device_ids=[self.device],output_device=self.device)

        

    def run(self):
        self.model.train()  # 将模型设置为训练模式
        for epoch in range(self.args.total_epochs):  # 进行训练循环，共200个epoch
            print("\tepoch num:", epoch)
            data =self.data.dataset#Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask
            # print("labels:",labels)
            self.optimizer.zero_grad()  # 梯度清零
            out = self.model(data)  # 前向传播，得到模型输出
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  # 计算损失，使用负对数似然损失函数
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