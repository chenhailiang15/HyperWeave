import json
import time





class Job:
    def __init__(self):
        self.arrive_time=0
        self.start_time=0
        self.end_time=0
        self.succeed_flage=None
            
            
    #模型信息        
    def set_model_info(self,job_name, model_name,total_epochs, batch_size, worker_num=4, layer_num=10, layer_feature=10 ):
        self.job_name=job_name
        self.model_name=model_name
        self.total_epochs=total_epochs
        self.batch_size=batch_size
        self.worker_num=worker_num
        self.layer_num=layer_num
        self.layer_feature=layer_feature
    
    #计划资源
    def set_plan_resource(self, plan_cpu, plan_mem, plan_gpu):
        self.plan_cpu=plan_cpu
        self.plan_mem=plan_mem
        self.plan_gpu=plan_gpu
    
    #时间信息
    def set_arrive_time(self, arrive_time):
        self.arrive_time=arrive_time
    def set_start_time(self, start_time):
        self.start_time=start_time
    def set_end_time(self, end_time):
        self.end_time=end_time
    
    def succeed(self):
        self.succeed_flage=True
    def failed(self):
        self.succeed_flage=False
    
    def set_execute_info(self,MASTER_ADDR, MASTER_PORT, net_card, node_rank,world_size ,nprocs_list, gpu_id_list, prior=False, max_sync_num=0, shm_name_list=[]):
        self.MASTER_ADDR=MASTER_ADDR
        self.MASTER_PORT=MASTER_PORT
        self.net_card=net_card
        self.node_rank=node_rank
        self.world_size=world_size
        self.nprocs_list=nprocs_list
        self.gpu_id_list=gpu_id_list
        self.prior=prior
        self.max_sync_num=max_sync_num
        self.shm_name_list=shm_name_list
        
        nprocs_list_c=nprocs_list.__str__().replace(" ","")
        gpu_id_list_c=gpu_id_list.__str__().replace(" ","")
        shm_name_list_c=shm_name_list.__str__().replace(" ", "")
        
        self.command=f"python WeaveExecutor.py --MASTER_ADDR {MASTER_ADDR} --MASTER_PORT {MASTER_PORT} --net_card {net_card}  --node_rank {node_rank} \
            --world_size {world_size} --nprocs_list {nprocs_list_c} --gpu_id_list {gpu_id_list_c} --model_name {self.model_name} --batch_size {self.batch_size} \
            --total_epochs {self.total_epochs} --worker_num {self.worker_num}--layer_num {self.layer_num} --layer_feature {self.layer_feature} \
            --squad_data_size {self.squad_data_size} --max_sync_num {max_sync_num} --shm_name_list {shm_name_list_c}"
        if prior:
            self.command=self.command+" --prior"
        
    
    
    def to_string(self):
        json_dict={}
        json_dict["job_name"]=self.job_name
        json_dict["model_name"]=self.model_name
        json_dict["total_epochs"]=self.total_epochs
        json_dict["batch_size"]=self.batch_size
        json_dict["worker_num"]=self.worker_num
        json_dict["layer_num"]=self.layer_num
        json_dict["layer_feature"]=self.layer_feature
        
        json_dict["MASTER_ADDR"]=self.MASTER_ADDR
        json_dict["MASTER_PORT"]=self.MASTER_PORT
        json_dict["net_card"]=self.net_card
        json_dict["node_rank"]=self.node_rank
        json_dict["world_size"]=self.world_size
        json_dict["nprocs_list"]=self.nprocs_list
        json_dict["gpu_id_list"]=self.gpu_id_list
        json_dict["prior"]=self.prior
        json_dict["max_sync_num"]=self.max_sync_num
        json_dict["shm_name_list"]=self.shm_name_list
        json_dict["command"]=self.command
        
        json_dict["plan_cpu"]=self.plan_cpu
        json_dict["plan_mem"]=self.plan_mem
        json_dict["plan_gpu"]=self.plan_gpu
        
        json_dict["arrive_time"]=self.arrive_time
        json_dict["start_time"]=self.start_time
        json_dict["end_time"]=self.end_time
        json_dict["succeed_flage"]=self.succeed_flage
        
        return json.dumps(json_dict)
    
    
    def load_string(self, json_string):
        json_dict=json.loads(json_string)
        
        self.job_name=json_dict["job_name"]
        self.model_name=json_dict["model_name"]
        self.total_epochs=json_dict["total_epochs"]
        self.batch_size=json_dict["batch_size"]
        self.worker_num=json_dict["worker_num"]
        self.layer_num=json_dict["layer_num"]
        self.layer_feature=json_dict["layer_feature"]
        
        self.MASTER_ADDR=json_dict["MASTER_ADDR"]
        self.MASTER_PORT=json_dict["MASTER_PORT"]
        self.net_card=json_dict["net_card"]
        self.node_rank=json_dict["node_rank"]
        self.world_size=json_dict["world_size"]
        self.nprocs_list=json_dict["nprocs_list"]
        self.gpu_id_list=json_dict["gpu_id_list"]
        self.prior=json_dict["prior"]
        self.max_sync_num=json_dict["max_sync_num"]
        self.shm_name_list=json_dict["shm_name_list"]
        self.command=json_dict["command"]
        
        self.plan_cpu=json_dict["plan_cpu"]
        self.plan_mem=json_dict["plan_mem"]
        self.plan_gpu=json_dict["plan_gpu"]
        
        self.arrive_time=json_dict["arrive_time"]
        self.start_time=json_dict["start_time"]
        self.end_time=json_dict["end_time"]
        self.succeed_flage=json_dict["succeed_flage"]
        
        
        
        
    def get_key_job_info(self):
        return f"jn:{self.job_name}-mn:{self.model_name}-tep:{self.total_epochs}-bts:{self.batch_size}-gpu:{self.plan_gpu}"