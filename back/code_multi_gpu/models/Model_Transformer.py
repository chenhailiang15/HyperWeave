import torch.nn as nn
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
import torch
import torch.nn as nn
import torch.optim as optim
# from torchtext.data import Field, BucketIterator
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
from torchtext.datasets import IMDB
from torchtext import data
# 导入经典文本相关数据集的工具包
import torchtext
# 导入专门用于英文分词的工具
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import copy
from torch.utils.data import dataset
from torch import nn, Tensor
import math



class Transformer_class:
    
    def __init__(self, args_t, dataset_dir):
        self.args=args_t
        self.dataset_dir=dataset_dir
    
    def set_local_rank(self, local_rank):
        self.local_rank = local_rank
        
    def set_shm_name(self,prior,shm_name,enable_flage=True):
        self.sync_er=Synchronizer(shm_name,prior=prior, enable_flage=enable_flage)
        
    def set_shm_name_analyze(self,shm_name):
        self.sync_er=Synchronizer(shm_name, shm_size=4)
        
        
    def data_process(self,raw_text_iter: dataset.IterableDataset) -> Tensor:
        """将原始文本转换成扁平的张量"""
        data = [torch.tensor(self.vocab(self.tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
        
    def load_mode_data(self):
        if torch.cuda.is_available():
            if len(self.args.gpu_id_list[self.args.node_rank]) != 0:
                self.device = "cuda:"+self.args.gpu_id_list[self.args.node_rank][self.local_rank].__str__()
            else:
                self.device = "cuda:"+self.local_rank.__str__()
        else:
            self.device="cpu"
        print(f"load model and data with device {self.device} ...")
        
        
    

        # 导入wikiText-2数据集并作基本处理
        # self.TEXT = torchtext.legacy.data.Field(tokenize=get_tokenizer("basic_english"),
        #                             init_token='<sos>',
        #                             eos_token = '<eos>',
        #                             lower=True)
        # 最终获得了一个Field对象，即TEXT是一个Field对象
        # 使用torchtext的数据集方法导入WikiText2数据
        # 并切分为对应训练文本，验证文本，测试文本，并对这些文本施加刚刚创建的预料阈
        train_iter = torchtext.datasets.WikiText2(root="../dataset", split='train') # splits切分
        
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, train_iter), specials=['<unk>'])
        
        train_iter = torchtext.datasets.WikiText2(root="../dataset", split='train')
        
        self.train_data = self.data_process( train_iter)
        self.train_data = self.batchify(self.train_data, self.args.batch_size)
        
        self.dataloaders = DataLoader(self.train_data, batch_size=self.args.batch_size,  shuffle=False, sampler=DistributedSampler(self.train_data))
        
        
        self.ntokens = len(self.vocab) # 词汇表的大小
        emsize = 200 # 嵌入维度
        d_hid = 200 # TransformerEncoder中前馈网络模型的维度
        nlayers = 2 # TransformerEncoder中EncoderLayer层数
        nhead = 2 # Transformer中的头数
        dropout = 0.2 # 丢弃概率

        self.model = TransformerModel( self.ntokens, emsize, nhead, d_hid, nlayers, dropout).to(self.device)

        # self.ntokens=len(self.vocab)
        # INPUT_DIM = len(self.vocab)
        # EMBEDDING_DIM = 300
        # HIDDEN_DIM = 256
        # OUTPUT_DIM = 2
        # N_LAYERS = 2
        # N_HEADS = 4
        # PF_DIM = 512
        # DROPOUT = 0.5
        # # 令句子长度允许的最大值是bptt为35
        self.bptt = 35 #即一个句子里，最多包含35个单词
        
        
        

        
        # self.model = TransformerModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, N_HEADS, PF_DIM, DROPOUT).to(self.device)
        
        self.optimizer =torch.optim.SGD(self.model.parameters(), lr=5.0)
        self.criterion = nn.CrossEntropyLoss()
        
        
        print("start ddp model..." )
        self.model = DDP(self.model, device_ids=[self.device], output_device=self.device)
        print("end ddp model...")
        
        
        
        
        
    def run(self):
       
        for epoch in range(1, self.args.total_epochs):
            self.model.train()
            print(f"epoch:{epoch}")
            # 开始遍历批次数据
            # for batch, i in enumerate(range(0, self.train_data.size(0)-1, self.bptt)):
            #     # 通过get_batch获得源数据和目标数据
            #     data, targets = self.get_batch(self.train_data, i)
            i=0
            print(f"batch num:{len(self.dataloaders)}")
            for data_all in self.dataloaders:
                data, targets = self.get_batch(data_all, 0)
                # 设置优化器初始采样梯度为0梯度
                self.optimizer.zero_grad()
                # 将数据装入model得到输出
                output = self.model(data)
                # 将输出和目标数据传入损失函数对象
                output_flat=output.view(-1,self.ntokens)
                loss = self.criterion(output_flat, targets)
                # 损失进行反向传播已获得总损失
                loss.backward()
                # 用nn自带的clip_grad_norm_方法进行梯度规范化，防止出现梯度消失或爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                # 模型参数进行更新
                self.optimizer.step()
                i+=1
                
                


    # ============================构建用于模型输入的批次化数据============================
    def batchify(self, data, bsz):
        """batchify函数用于将文本数据映射成连续数字，并转换成指定的样式，指定的样式可参考下图
        batchify函数有两个参数，data是我们之前得到的文本数据(train_txt, val_txt, test_txt)，
        bsz是就是batch_size，每次模型更新参数的数据量"""
        seq_len = data.size(0) // bsz
        data = data[:seq_len*bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data.to(self.device)



    
    def get_batch(self,source, i):
        """获取批次数据
        参数：
            source: Tensor, 形状为 ``[full_seq_len, batch_size]``
            i: int, 当前批次索引
        返回：
            tuple(data, target),
            - data形状为[seq_len, batch_size]
            - target形状为[seq_len * batch_size]
        """
        # 计算当前批次的序列长度，最大为bptt，确保不超过source的长度
        seq_len = min(self.bptt, len(source) - 1 - i)
        # 获取data，从i开始，长度为seq_len
        data = source[i:i+seq_len]
        # 获取target，从i+1开始，长度为seq_len，并将其形状转换为一维张量
        target = source[i+1:i+1+seq_len].reshape(-1)

        return data, target

    


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # 生成位置编码的位置张量
        position = torch.arange(max_len).unsqueeze(1)
        # 计算位置编码的除数项
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # 创建位置编码张量
        pe = torch.zeros(max_len, 1, d_model)
        # 使用正弦函数计算位置编码中的奇数维度部分
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # 使用余弦函数计算位置编码中的偶数维度部分
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """Arguments:
            x: Tensor, 形状为 [seq_len, batch_size, embedding_dim]
        """
        # 将位置编码添加到输入张量
        x = x + self.pe[:x.size(0)]
        # 应用dropout
        return self.dropout(x)
        
# Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()

		# 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 定义编码器层
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)

        # 定义编码器
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """Arguments
            src: Tensor, 形状为 [seq_len, batch_size]
            src_mask: Tensor, 形状为 [seq_len, seq_len]

        Returns:
            输出的Tensor, 形状为 [seq_len, batch_size, ntoken]
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


# class TransformerModel(nn.Module):
#     def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, n_heads, pf_dim, dropout):
#         super().__init__()

#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.n_layers = n_layers
#         self.n_heads = n_heads
#         self.pf_dim = pf_dim
#         self.dropout = dropout

#         self.embedding = nn.Embedding(input_dim, embedding_dim)

#         self.encoder_layer = nn.TransformerEncoderLayer(embedding_dim, n_heads, pf_dim, dropout)
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)

#         self.fc = nn.Linear(embedding_dim, output_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, src):
#         embedded = self.dropout(self.embedding(src))
#         embedded = self.encoder(embedded)
#         embedded = embedded.mean(dim=0)
#         embedded = self.dropout(embedded)
#         output = self.fc(embedded)
#         return output
