import sys
sys.path.append("..")
sys.path.append("platform_h")
# 分析模型运行时GPU空闲时间，通过增加并行和batch size来对比
from cluster.platform_h.util import *





def generate_command(model_name, parallel_num, total_epochs, batch_size,gpu_id_list, net_card):
    world_size=parallel_num
    nprocs_list=f"[{world_size}]"
    gpu_id_list=f"[{gpu_id_list}]".replace(" ","")
    
    
    # if model_name == "Bert":
    #     batch_size=8
    # else:
    #     batch_size=16           #8 for Bert (default:16)
    

    if model_name == "GCN":
        layer_num=100        #5000 for GCN (default:10)
        layer_feature=100        #100 for GCN (default:10)
    else:
        layer_num=10       
        layer_feature=10 
              
    global port_id
    port_id+=1
    while is_port_in_use("localhost",port_id):
        port_id+=1
    
    
    command=f"python platform_h/HyperWeaveExecutor.py --model_name {model_name}  --world_size {world_size} --net_card {net_card}  --MASTER_PORT {port_id}\
    --nprocs_list {nprocs_list} --gpu_id_list {gpu_id_list} --layer_num {layer_num} --layer_feature {layer_feature} \
    --batch_size {batch_size} --total_epochs {total_epochs} --record_flage"
    return command

def run_command(command):
        print(f"command: {command}")
        back=os.system(command)
        

port_id=2000


def exp_one_model(model_name):
    # max_parallel_num=3
    # batch_size_list=[256] #
    total_epochs=2
    gpu_id_list=[0,1,2,3]
    net_card="eth0"
    
    
    now_time = datetime.datetime.now()
    formatted_time = now_time.strftime('%m_%d_%H_%M_%S')
    
    # for parallel_num in range(1, max_parallel_num+1):
    #     for batch_size in batch_size_list:
    
    parallel_num=1
    batch_size=512
    # out_file_name="ExpPre__compareIdelTime_"+model_name+"_"+str(batch_size)+"_"+str(parallel_num)+"_"+formatted_time+".txt"
    command=generate_command(model_name, parallel_num, total_epochs, batch_size, gpu_id_list, net_card)
    run_command(command)
    
    # parallel_num=4
    # batch_size=128
    # # out_file_name="ExpPre__compareIdelTime_"+model_name+"_"+str(batch_size)+"_"+str(parallel_num)+"_"+formatted_time+".txt"
    # command=generate_command(model_name, parallel_num, total_epochs, batch_size, gpu_id_list, net_card)
    # run_command(command)
    
if __name__=="__main__":
    
    exp_one_model("ResNet50")
    # exp_one_model("VGG16")
    # exp_one_model("MobileNetv2")
    # exp_one_model("AlexNet")
    # exp_one_model("ResNet18")