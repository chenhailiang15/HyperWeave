import time
from WeaveSynchronizer import Synchronizer
from models.Framework_model_CV import CVModel
from models.Framework_model_Bert import BertModel
from models.Framework_model_Transformer import TransformerModel
from models.Framework_model_GraphSage import GraphSageModel
from models.Framework_model_GCN import GCNModel
import torch, gc


class model_framework:
    def __init__(self, local_rank, args):
        self.system=args.system
        self.mode=args.mode
        self.job_idx=args.job_idx
        self.idx_on_gpu=args.idx_on_gpu
        
        self.model_name=args.model_name
        self.local_rank=local_rank
        self.args=args
        
        if torch.cuda.is_available():
            if len(self.args.gpu_id_list)>=self.args.node_rank+1 and len(self.args.gpu_id_list[self.args.node_rank]) != 0:
                self.device = "cuda:"+self.args.gpu_id_list[self.args.node_rank][self.local_rank].__str__()
            else:
                self.device = "cuda:"+self.local_rank.__str__()
        else:
            self.device="cpu"
        print(f"load model {self.args.model_name} and data with device {self.device} ...")
        
        
        args.device=self.device
        
        
        if args.model_name == "ResNet18" or args.model_name == "ResNet50" or args.model_name =="AlexNet"\
            or args.model_name =="VGG16" or args.model_name =="MobileNetv2":
            self.model=CVModel(args)
        elif args.model_name == "Bert":
            self.model=BertModel(args)
        elif args.model_name == "Transformer":
            self.model=TransformerModel(args)
        elif args.model_name == "GraphSage":
            self.model=GraphSageModel(args)
        elif args.model_name == "GCN":
            self.model=GCNModel(args)
        
        
        else:
            print("model_name wrong!")
            exit(-1)
        
        
        
    def init_sync_er(self):
        try:
            try:
                gpu_id=self.args.gpu_id_list[self.args.node_rank][self.local_rank]
            except:
                gpu_id=self.local_rank
                
            shm_name=self.args.shm_name_list[self.args.node_rank][gpu_id]
            enable_flage=True
        except:
            shm_name=""
            enable_flage=False
        
        if self.system == "Weave":
            if self.args.mode=="train":
                self.sync_er=Synchronizer(shm_name, shm_size=3, prior=self.args.prior, enable_flage=enable_flage)
            elif self.args.mode == "analyze" and self.local_rank==0 :
                self.sync_er=Synchronizer(shm_name, shm_size=4)
        elif self.system == "Muri":
            if self.args.mode=="train":
                self.sync_er=Synchronizer(shm_name, shm_size=12, max_sync_num=self.args.max_sync_num,enable_flage=enable_flage)
            elif self.args.mode == "analyze" and self.local_rank==0:
                self.sync_er=Synchronizer(shm_name, shm_size=16)  #用来将测试出来的耗时传递出去
            
        return  

    
    def load_mode_data(self):
        if self.system=="Weave" and self.mode=="analyze":
            if self.local_rank==0:
                self.sync_er.set_value(0,True)
                time.sleep(10)
                self.sync_er.set_value(0,False)
                self.sync_er.set_value(1,True)
            else:
                time.sleep(10)
        if self.system == "Muri":
            if self.mode == "train":
                
                self.sync_er.muri_sync_start(self.idx_on_gpu, 0)
                print(f"job {self.job_idx} start stage 0...")
                # print(f"job {self.job_idx} start stage 0...")
            elif self.mode == "analyze" and self.local_rank==0:
                stage0_start_time=time.time()
        #************************
        self.model.prepare()
        #************************
        if self.system=="Weave" and self.mode == "analyze":
            if self.local_rank==0:
                self.sync_er.set_value(1,False)
        
        if self.system == "Muri":
            if self.mode == "analyze" and self.local_rank==0:
                stage0_end_time=time.time()
                self.sync_er.set_value(0, stage0_end_time-stage0_start_time)
            
            #************************
            self.model.prepare_sub()
            #************************
            if self.mode == "train":
                # print(f"job {self.job_idx} end stage 0...")
                self.sync_er.muri_sync_end(self.idx_on_gpu, 0)
            
                
                
            
    def run(self):
        if self.system=="Weave":
            for epoch in range(self.args.total_epochs):
                print(f"job:{self.job_idx} epoch:{epoch+1}/{self.args.total_epochs} gpu:{self.device} ...")
                 
                if self.mode == "train":
                    self.sync_er.sync_in_start_epoch(epoch==0)
                    # print("model name:",self.args.model_name,"\tepoch:",epoch,"/",self.args.total_epochs-1)
                elif self.mode == "analyze":
                    if self.local_rank==0 and epoch==1:
                        self.sync_er.set_value(2,True)
                else:
                    print("model mode wrong!")
                    exit(-1)
                #**********************************************************
                self.model.sample()
                #**********************************************************
                #进行同步操作 等待信号，方可继续执行，后方代码主要利用GPU
                if self.mode == "train":
                    self.sync_er.sync_in_batch()
                elif self.mode == "analyze":
                    if self.local_rank==0 and epoch==1:
                        self.sync_er.set_value(2,False)
                        self.sync_er.set_value(3,True)
                else:
                    print("model mode wrong!")
                    exit(-1)
                    
                #**********************************************************
                self.model.train()
                #**********************************************************
                if self.mode == "train":
                    #进行同步操作（）
                    self.sync_er.sync_in_end_epoch()
                elif self.mode == "analyze":
                    if self.local_rank==0 and epoch==1:
                        self.sync_er.set_value(3,False)
                else:
                    print("model mode wrong!")
                    exit(-1)
                
                # gc.collect()
                # torch.cuda.empty_cache()
                
        elif self.system == "Muri":
            if self.mode=="analyze" and self.local_rank==0:
                stage1_time_all=0
                stage2_time_all=0
                stage3_time_all=0
                record_num=0
            elif self.mode=="analyze":
                record_num=0
                
            while not self.model.is_end():
                if self.model.batch_idx==0 :
                    print(f"job:{self.job_idx} tepoch:{self.args.total_epochs} tbatch:{self.model.total_batch_num} gpu:{self.device} ...")
                
                # print(f"batch id: {self.model.batch_idx}")
                if self.mode == "train":
                    self.sync_er.muri_sync_start(self.idx_on_gpu,1)
                    # print(f"job {self.job_idx} start stage 1...")
                elif self.mode == "analyze" and self.local_rank==0:
                    stage1_start_time=time.time()
                #************************
                data=self.model.get_data()
                #************************
                if self.model.batch_idx%500 == 1:
                    print(f"job:{self.job_idx} batch:{self.model.batch_idx}/{self.model.total_batch_num} ({self.model.cur_epoch+1}/{self.args.total_epochs}) ...")
                
                
                
                if self.mode == "train":
                    self.sync_er.muri_sync_end(self.idx_on_gpu,1)
                    
                    self.sync_er.muri_sync_start(self.idx_on_gpu,2)
                    # print(f"job {self.job_idx} start stage 2...")
                elif self.mode == "analyze" and self.local_rank==0:
                    stage2_start_time=time.time()
                #************************
                self.model.forward_backward(data)
                #************************
                if self.mode == "train":
                    self.sync_er.muri_sync_end(self.idx_on_gpu,2)
                    
                    self.sync_er.muri_sync_start(self.idx_on_gpu,3)
                    # print(f"job {self.job_idx} start stage 3...")
                elif self.mode == "analyze" and self.local_rank==0:
                    stage3_start_time=time.time()
                #************************
                self.model.comm()
                #************************
                
                if self.mode == "train":
                    self.sync_er.muri_sync_end(self.idx_on_gpu,3)
                elif self.mode == "analyze" and self.local_rank==0:
                    stage_end_time=time.time()
                    stage1_time_all+=stage2_start_time-stage1_start_time
                    stage2_time_all+=stage3_start_time-stage2_start_time
                    stage3_time_all+=stage_end_time-stage3_start_time
                    record_num+=1
                    if record_num>=100:
                        break
                    
                elif self.mode == "analyze":
                    record_num+=1
                    if record_num>=100:
                        break
            
            
            if self.mode == "train":
                self.sync_er.muri_job_end(self.idx_on_gpu)
            elif self.mode == "analyze" and self.local_rank==0:
                self.sync_er.set_value(1, stage1_time_all/record_num)
                self.sync_er.set_value(2, stage2_time_all/record_num)
                self.sync_er.set_value(3, stage3_time_all/record_num)
                        
        elif self.system=="Normal":
            for epoch in range(self.args.total_epochs):
                print(f"job_idx: {self.job_idx} idx on gpu: {self.idx_on_gpu} epoch: {epoch+1}/{self.args.total_epochs}...")
                self.model.sample()
                self.model.train()
            
        else:
            print(f"system wrong! {self.system}")
            
            
        return #等待完善
                    # self.model.prepare_sub()
                    # while not self.model.is_epoch_end():
                    #     if self.model.batch_idx%500 == 0:
                    #         print(f"split mode batch:{self.model.batch_idx}/{self.model.total_batch_num}")
                    #     data_tuple=self.model.get_data()
                    #     self.model.forward_backward(data_tuple)
                    #     self.model.comm()
        
        
        print("train over!")
        