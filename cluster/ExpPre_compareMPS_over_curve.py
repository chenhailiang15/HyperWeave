import sys
sys.path.append("..")
sys.path.append("platform_h")
# 生成不同模型在MPS开启和不开启之下，并行时的资源消耗
from platform_h.util import *
from models.Framework import *

lock = threading.Lock()

def generate_command(model_info, job_idx):
    nprocs_list="[1,0]"
    model_name=model_info[0]
    batch_size=model_info[1]
    total_epochs=model_info[2]
    gpu_id_list=f"[[{gpu_id}],[]]"
    net_card="eth0"

    # total_epochs=2
    # if model_name == "Bert":
    #     batch_size=batch_size_list_bert[index]    #16
    # else:
    #     batch_size=batch_size_list[index]   #128           #8 for Bert (default:16)
    

    if model_name == "GCN":
        layer_num=100        #5000 for GCN (default:10)
        layer_feature=100        #100 for GCN (default:10)
    elif model_name=="GraphSage":
        layer_num=10       
        layer_feature=10
    else:
        layer_num=10       
        layer_feature=10 
    with lock:          
        global port_id
        port_id+=1
        while is_port_in_use("localhost",port_id):
            port_id+=1
        
        
        command=f"python platform_h/WeaveExecutor.py --model_name {model_name}  --net_card {net_card}  --MASTER_PORT {port_id}\
        --nprocs_list {nprocs_list} --gpu_id_list {gpu_id_list} --layer_num {layer_num} --layer_feature {layer_feature} \
        --batch_size {batch_size} --total_epochs {total_epochs} --job_idx {job_idx}"
    return command

def is_end(model_group,job_order):
    if job_order>=len(model_group):
        return True
    else:
        return False

def run_command(model_group,index):
    global end_time_list,job_order
    start_time_t=time.time()
    while not is_end(model_group,job_order):
        this_index=job_order
        job_order+=1
        command=generate_command(model_group[this_index], this_index)
        
        print(f"command: {command}")
        back=os.system(command)
        # back=0
        if back==0:
            end_time_list[this_index]=time.time()-start_time_t
            print(f"end_time_list:{end_time_list}")   
        else:
            print("model run wrong!")
            duration_time=-1
            end_time_list[this_index]=duration_time
            return
    
    return  
    

def do_experiment(model_group, mps_state,file_writer,alloc,para_num):
    global end_time_list,job_order
    job_order=0
    thread_hand=[]
    #初始化统计变量
    job_parallel_num=len(model_group)
    end_time_list=[0]*job_parallel_num
    # end_job_num_list=[0]*job_parallel_num
        
    for index in range(para_num):
        
        # command = generate_command(model_info, index)
        sub_thread=threading.Thread(target=run_command,args=(model_group,index, ))
        sub_thread.start()
        thread_hand.append(sub_thread)
        
        

    for thread_t in thread_hand:
        thread_t.join()
    
    
    file_writer.write(f"mps={mps_state},alloc={alloc},||{model_group}||,time_list={end_time_list}\n")
    file_writer.flush()

            
port_id=2000
gpu_id=0
end_time_list=[]
# end_job_num_list=[]
job_order=0
mps_limite=33.3
para_num_low=1
para_num_high=9

model_group=[['ResNet18', 8, 1],['ResNet18', 16, 1],['ResNet18', 32, 1], ['ResNet18', 64, 1],['ResNet18', 128, 1],
                   ['AlexNet', 8, 1], ['AlexNet', 16, 1], ['AlexNet', 32, 1], ['AlexNet', 64, 1],
                   ['Transformer', 8, 1],['Transformer', 16, 1],
                   ['ResNet50', 8, 1], ['ResNet50', 16, 1], ['ResNet50', 32, 1], ['ResNet50', 64, 1],
                   ['MobileNetv2', 8, 1],['MobileNetv2', 16, 1],['MobileNetv2', 32, 1], ['MobileNetv2', 64, 1],['MobileNetv2', 128, 1]
            ]
                  



#记录代码开始时间
now_time = datetime.datetime.now()
formatted_time = now_time.strftime('%m_%d_%H_%M_%S')

out_file_name="ExpPre_compareMPS_"+formatted_time+".txt"
file_writer=open(get_output_dir()+out_file_name,"w")

with_mps=True
alloc_percent=mps_limite

start_MPS(11) 
command_alloc=f"echo set_default_active_thread_percentage {alloc_percent} | nvidia-cuda-mps-control"
os.system(command_alloc)
    
for para_num in range(para_num_low,para_num_high+1):
    do_experiment(model_group, with_mps,file_writer,alloc=True,para_num=para_num )
     


    
file_writer.close()

stop_MPS(11)