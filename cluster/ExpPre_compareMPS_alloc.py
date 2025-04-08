# 生成不同模型在MPS开启和不开启之下，并行时的资源消耗
from util import *
from models.Framework import *

batch_size_list=[16,32,128,64,8]
batch_size_list_bert=[16,8,16,16,8]

def do_experiment(model_group, mps_state, max_parallel_num,file_writer,alloc):
    
    
    for para_num in range(len(model_group),len(model_group)+1):
        if para_num==1:
            repeat=2
        else:
            repeat=1
            
        
            
        for _ in range(repeat):
            run_specific_model_paranum(model_group, para_num)
            file_writer.write(f"mps={mps_state},alloc={alloc},||{model_group}||,para_num={para_num},time_list={end_time_list}\n")
            file_writer.flush()
 
 
def run_specific_model_paranum(model_group,para_num):
    global end_time_list
    end_time_list=[]
    thread_hand=[]
    
    for model_info in range(model_group):
        
        command = generate_command(model_info)
        sub_thread=threading.Thread(target=run_command,args=(command, ))
        sub_thread.start()
        thread_hand.append(sub_thread)
        

    for thread_t in thread_hand:
        thread_t.join()
    


    
    
def generate_command(model_info):
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
    else:
        layer_num=10       
        layer_feature=10 
              
    global port_id
    port_id+=1
    while is_port_in_use("localhost",port_id):
        port_id+=1
    
    
    command=f"python WeaveExecutor.py --model_name {model_name}  --net_card {net_card}  --MASTER_PORT {port_id}\
    --nprocs_list {nprocs_list} --gpu_id_list {gpu_id_list} --layer_num {layer_num} --layer_feature {layer_feature} \
    --batch_size {batch_size} --total_epochs {total_epochs}"
    return command


def run_command(command):
        start_time=time.time()
        print(f"command: {command}")
        back=os.system(command)
        # back=0
        if back==0:
            end_time=time.time()
            duration_time=end_time-start_time
            
        else:
            print("model run wrong!")
            duration_time=-1
        
        end_time_list.append(duration_time)    
        return  
    



            
port_id=2000
end_time_list=[]
gpu_id=0

            
if __name__=="__main__":
    max_parallel_num=5
    
    # "AlexNet", "Transformer", "GCN", "Bert", "GraphSage", "ResNet18", "ResNet50",, "VGG16"
    model_group_list=[[["ResNet50",512,5],["GCN",100,2],["AlexNet",16,2]],
                      [["VGG16",512,5],["Bert",32,2],["ResNet18",16,2]],
                      [["MobileNetv2",512,5],["GraphSage",32,2],["Transformer",128,2]],
                      [["ResNet50",512,5],["VGG16",32,2],["MobileNetv2",16,2]],
                      [["Transformer",128,3],["Transformer",128,3],["Transformer",128,3]],
                      [["ResNet50",256,2],["ResNet50",256,2],["ResNet50",256,2]]]
                    #   "ResNet50", "VGG16", "MobileNetv2","AlexNet", "Transformer", "GCN", "Bert", "GraphSage", "ResNet18"]#"ResNet50", "MobileNetv2", "VGG16",  "Transformer", "GCN" 
    
    
    #记录代码开始时间
    now_time = datetime.datetime.now()
    formatted_time = now_time.strftime('%m_%d_%H_%M_%S')
    
    out_file_name="ExpPre_compareMPS_"+formatted_time+".txt"
    file_writer=open(get_output_dir()+out_file_name,"w")
    
    
    for one_group_info in model_group_list:
        with_mps=False
        stop_MPS(11) 
        do_experiment(one_group_info, with_mps, max_parallel_num,file_writer,alloc=False )
        
        with_mps=True
        start_MPS(11) 
        alloc_percent=100
        command_alloc=f"echo set_default_active_thread_percentage {alloc_percent} | nvidia-cuda-mps-control"
        os.system(command_alloc)
        do_experiment(one_group_info, with_mps, max_parallel_num,file_writer,alloc=False )
        
    
    # for model_name in model_list:
        
        alloc_percent=100/len(one_group_info)
        command_alloc=f"echo set_default_active_thread_percentage {alloc_percent} | nvidia-cuda-mps-control"
        os.system(command_alloc)
        
        do_experiment(one_group_info, with_mps, max_parallel_num,file_writer,alloc=True )
        
    # for model_name in model_list:
        
        
    # 
        
        
        
    file_writer.close()
    # file_writer_true_alloc.close()
    # file_writer_false.close()
     
    stop_MPS(11)
    
    



    
    
    
    




