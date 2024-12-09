import time
from NodeCommunicate import NodeMessageSender
import json
import secrets
import string
 
def generate_shm_name(length=10):
    # 选择字母和数字
    characters = string.ascii_letters + string.digits
    # 使用secrets.choice从characters中随机选择字符，并使用join将它们组合成一个字符串
    secure_random_string = ''.join(secrets.choice(characters) for i in range(length))
    return secure_random_string
 


# if __name__=="__main__":
#     worker_ip="10.26.128.51"
#     worker_port=8000
#     node_message_sender=NodeMessageSender(worker_ip,worker_port)
#     while True:
#         send_info=input("发送数据：")
#         node_message_sender.send(send_info)
        
        

        
''' how to record a shedule
model para:






'''

def get_strategy():
    strategy_all={}
    ####################第一个任务信息
    task_first={}
    task_first["MASTER_ADDR"]="10.26.128.70"
    task_first["MASTER_PORT"]=12355
    task_first["nnodes"]=2
    task_first["nprocs_per_node"]=2
    
    task_first["model_name"]="AlexNet"
    task_first["total_epoch"]=3
    task_first["batch_size"]=16
    task_first["worker_num"]=4
    
    task_first["max_sync_num"]=2
    
    
    #master特定信息
    task_first["master_spec"]={}
    task_first["master_spec"]["net_card"]="eno1"
    task_first["master_spec"]["node_rank"]=1
    task_first["master_spec"]["gpu_id_list"]=[1,2]
    #worker特定信息
    task_first["worker_spec"]={}
    task_first["worker_spec"]["net_card"]="eno1"
    task_first["worker_spec"]["node_rank"]=1
    task_first["worker_spec"]["gpu_id_list"]=[2,3]
    
    ####################第二个任务信息
    task_second={}
    task_second["MASTER_ADDR"]="10.26.128.70"
    task_second["MASTER_PORT"]=12345
    task_second["nnodes"]=2
    task_second["nprocs_per_node"]=2
    
    task_second["model_name"]="ResNet18"
    task_second["total_epoch"]=3
    task_second["batch_size"]=16
    task_second["worker_num"]=4
    
    task_second["max_sync_num"]=2
    #master特定信息
    task_second["master_spec"]={}
    task_second["master_spec"]["net_card"]="eno1"
    task_second["master_spec"]["node_rank"]=1
    task_second["master_spec"]["gpu_id_list"]=[1,2]
    #worker特定信息
    task_second["worker_spec"]={}
    task_second["worker_spec"]["net_card"]="eno1"
    task_second["worker_spec"]["node_rank"]=1
    task_second["worker_spec"]["gpu_id_list"]=[2,3]
    
    ##############共同共享内存名
    task_first["shm_name_list"]=[]
    task_second["shm_name_list"]=[]
    for i in range(task_first["max_sync_num"]):
        shm_name=generate_shm_name(16)
        task_first["shm_name_list"].append(shm_name)
        task_second["shm_name_list"].append(shm_name)
        
    strategy_all["task_first"]=task_first
    strategy_all["task_second"]=task_second
    return strategy_all



def execution_local(strategy):
    
    return
    
    
    
worker_ip="10.26.128.51"
worker_port=8000
node_message_sender=NodeMessageSender(worker_ip,worker_port)
if __name__=="__main__":
    strategy_all=get_strategy()
    node_message_sender.send(json.dumps(strategy_all))
    execution_local(strategy_all)
    
    print(strategy_all)
    
    
         
