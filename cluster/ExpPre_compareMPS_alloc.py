# 生成不同模型在MPS开启和不开启之下，并行时的资源消耗
from util import *
from models.Framework import *


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
        layer_num=50       
        layer_feature=100
    else:
        layer_num=10       
        layer_feature=10 
              
    global port_id
    port_id+=1
    while is_port_in_use("localhost",port_id):
        port_id+=1
    
    
    command=f"python WeaveExecutor.py --model_name {model_name}  --net_card {net_card}  --MASTER_PORT {port_id}\
    --nprocs_list {nprocs_list} --gpu_id_list {gpu_id_list} --layer_num {layer_num} --layer_feature {layer_feature} \
    --batch_size {batch_size} --total_epochs {total_epochs} --job_idx {job_idx}"
    return command

def is_end():
    if all(num > 0 for num in end_job_num_list):
        return True
    else:
        return False

def run_command(command,index):
    global end_time_list, end_job_num_list
    start_time_t=time.time()
    temp_end_time=0
    while not is_end():
        print(f"command: {command}")
        back=os.system(command)
        # back=0
        if back==0:
            end_job_num_list[index]+=1
            temp_end_time=end_time_list[index]
            end_time_list[index]=time.time()-start_time_t
        else:
            print("model run wrong!")
            duration_time=-1
            end_time_list[index]=duration_time
            return
    
    if end_job_num_list[index]!=1:
        end_time_list[index]=temp_end_time/(end_job_num_list[index]-1)   
    return  
    

def do_experiment(model_group, mps_state,file_writer,alloc):
    global end_time_list, end_job_num_list
    thread_hand=[]
    #初始化统计变量
    job_parallel_num=len(model_group)
    end_time_list=[0]*job_parallel_num
    end_job_num_list=[0]*job_parallel_num
        
        
        
    for index, model_info in enumerate(model_group):
        
        command = generate_command(model_info,index)
        sub_thread=threading.Thread(target=run_command,args=(command,index, ))
        sub_thread.start()
        thread_hand.append(sub_thread)
        

    for thread_t in thread_hand:
        thread_t.join()
    
    
    file_writer.write(f"mps={mps_state},alloc={alloc},||{model_group}||,time_list={end_time_list}\n")
    file_writer.flush()

            
port_id=2000
gpu_id=7
end_time_list=[]
end_job_num_list=[]

           

max_parallel_num=5



model_group_list=[[["ResNet50",512,1],["GCN",8,1]],
                    # [["ResNet50",256,5],["MobileNetv2",8,1],["MobileNetv2",8,1],["MobileNetv2",8,1]],
                    # [["ResNet50",256,5],["MobileNetv2",32,1],["MobileNetv2",32,1],["MobileNetv2",32,1]],
                    # [["ResNet50",256,5],["Transformer",64,1],["Transformer",64,1],["Transformer",64,1]],
                    # [["ResNet50",256,10],["VGG16",8,1],["VGG16",8,1],["VGG16",8,1]],
                    # [["VGG16",256,10],["Bert",8,1],["Bert",8,1],["Bert",8,1]],
                    # [["VGG16",256,10],["Transformer",64,1],["Transformer",64,1],["Transformer",64,1]],
                    # [["VGG16",256,10],["GCN",8,1],["GCN",8,1],["GCN",8,1]],
                    # [["VGG16",256,10],["MobileNetv2",8,1],["MobileNetv2",8,1],["MobileNetv2",8,1]],
                    # # [["MobileNetv2",512,20],["GraphSage",8,2],["GraphSage",8,2],["GraphSage",8,2]],
                    # [["ResNet50",512,10],["Transformer",8,1],["Transformer",8,1],["Transformer",8,1]],
                    # [["Transformer",128,2],["MobileNetv2",32,1],["MobileNetv2",32,1],["MobileNetv2",32,1]],
                    # [["MobileNetv2",512,10],["Bert",8,2],["Bert",8,2],["Bert",8,2]],
                    
                    # [["GraphSage",128,5],["GraphSage",128,5]],
                    # [["Bert",16,10],["Bert",16,10]],
                    
                    # [["GCN",8,5],["GCN",8,5]],
                    # [["Transformer",128,2],["Transformer",128,2]],
                    # [["ResNet50",256,2],["ResNet50",256,2]],
                    # [["Transformer",128,2],["Transformer",128,2],["Transformer",128,2],["Transformer",128,2]],
                    # [["ResNet50",256,2],["ResNet50",256,2],["ResNet50",256,2],["ResNet50",256,2]]
                    ]
                #   "ResNet50", "VGG16", "MobileNetv2","AlexNet", "Transformer", "GCN", "Bert", "GraphSage", "ResNet18"]#"ResNet50", "MobileNetv2", "VGG16",  "Transformer", "GCN" 


#记录代码开始时间
now_time = datetime.datetime.now()
formatted_time = now_time.strftime('%m_%d_%H_%M_%S')

out_file_name="ExpPre_compareMPS_"+formatted_time+".txt"
file_writer=open(get_output_dir()+out_file_name,"w")


for one_group_info in model_group_list:
    with_mps=False
    stop_MPS(11) 
    do_experiment(one_group_info, with_mps,file_writer,alloc=False )
    
    with_mps=True
    start_MPS(11) 
    alloc_percent=100/len(one_group_info)
    command_alloc=f"echo set_default_active_thread_percentage {alloc_percent} | nvidia-cuda-mps-control"
    os.system(command_alloc)
    
    do_experiment(one_group_info, with_mps,file_writer,alloc=True )
    
    
    alloc_percent=100
    command_alloc=f"echo set_default_active_thread_percentage {alloc_percent} | nvidia-cuda-mps-control"
    os.system(command_alloc)
    do_experiment(one_group_info, with_mps,file_writer,alloc=False )
    

# for model_name in model_list:
    
    
    
# for model_name in model_list:
    
    
# 
    
    
    
file_writer.close()
# file_writer_true_alloc.close()
# file_writer_false.close()
    
stop_MPS(11)
    
    



    
    
    
    




