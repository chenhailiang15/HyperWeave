from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import pickle
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW


class BertModel:
    def __init__(self, args):
        self.idx=args.idx
        self.args=args

    def prepare(self):
        self.device=self.args.device
        
        with open(self.dataset_dir+'SQuAD_train_features.pkl', 'rb') as f:
            train_features = pickle.load(f)
            
        # 将特征转换为PyTorch张量
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

        # train_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_start_positions, all_end_positions)
        if self.args.squad_data_size == 0:
            train_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_start_positions, all_end_positions)
        else:
            num_samples = self.args.squad_data_size
            train_dataset = TensorDataset(
                all_input_ids[:num_samples],
                all_attention_mask[:num_samples],
                all_token_type_ids[:num_samples],
                all_start_positions[:num_samples],
                all_end_positions[:num_samples])
        train_sampler = RandomSampler(train_dataset)
        self.train_dataloader = DataLoader(train_dataset, sampler=DistributedSampler(train_sampler), batch_size=self.args.batch_size)
        #num_workers = worker_num,
        # 加载BERT模型和优化器
        # 下载未经微调的BERT
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(self.device)
        para_path="../model_para_data/bert_para/"
        self.model = BertForQuestionAnswering.from_pretrained(para_path).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        print("start ddp model..." )
        self.model = DDP(self.model, device_ids=[self.device],output_device=self.device)
        print("end ddp model...")
        self.model.train()
        self.cur_epoch = 0


    def prepare_sub(self):  
        self.dataloader_iter = iter(self.train_dataloader)
        self.batch_idx = -1

    
    def get_data(self):
        '''
        get data
        '''
        try:
            step, batch = next(self.dataloader_iter)
        except StopIteration:
            self.cur_epoch += 1
            self.train_sampler.set_epoch(self.cur_epoch)
            self.dataloader_iter = iter(self.train_dataloader)
            step, batch = next(self.dataloader_iter)
            self.batch_idx = -1
        self.batch_idx +=1
        
        return step, batch
    
    
    def forward_backward(self, thread):
        '''
        forward, calculate loss and backward
        '''
        
    
    
    
    
    def run(self):
        # 微调BERT
        for epoch in range(self.args.total_epochs):
            print("epoch:",epoch)
            
            for step, batch in enumerate(self.train_dataloader):
                
                self.model.train()
                self.optimizer.zero_grad()
                input_ids, attention_mask, token_type_ids, start_positions, end_positions = tuple(t.to(self.device) for t in batch)
                outputs = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                start_positions=start_positions,
                                end_positions=end_positions)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                print(f"Epoch [{epoch + 1}/{self.args.total_epochs}], Step [{step + 1}/{len(self.train_dataloader)}], Loss: {loss.item():.4f}")