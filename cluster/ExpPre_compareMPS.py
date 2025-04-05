# 生成不同模型在MPS开启和不开启之下，并行时的资源消耗
from util import *
from models.Framework import *




def do_experiment(mps_state, max_parallel_num,file_writer,alloc):
    if mps_state ==True:
        start_MPS(11) 
    else:
        stop_MPS(11)
    
    
    
    for para_num in range(1, max_parallel_num+1):
        if para_num==1:
            repeat=2
        else:
            repeat=1
            
        if mps_state ==True:
            if alloc==True:
                alloc_percent=100/para_num
            else:
                alloc_percent=100
            command_alloc=f"echo set_default_active_thread_percentage {alloc_percent} | nvidia-cuda-mps-control"
            os.system(command_alloc)
            
        for _ in range(repeat):
            run_specific_model_paranum(model_name, para_num)
            file_writer.write(f"mps={mps_state},alloc={alloc},{model_name},para_num={para_num},time_list={end_time_list}\n")
            file_writer.flush()
 
 
def run_specific_model_paranum(model_name,para_num):
    global end_time_list
    end_time_list=[]
    thread_hand=[]
    
    for index in range(para_num):
        
        command = generate_command(model_name, index)
        sub_thread=threading.Thread(target=run_command,args=(command, ))
        sub_thread.start()
        thread_hand.append(sub_thread)

    for thread_t in thread_hand:
        thread_t.join()
    


    
    
def generate_command(model_name, index):
    nprocs_list="[1,0]"

    gpu_id_list=f"[[{gpu_id}],[]]"
    net_card="eth0"

    total_epochs=2
    if model_name == "Bert":
        batch_size=8
    else:
        batch_size=32           #8 for Bert (default:16)
    

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
       
    max_parallel_num=10
    # "AlexNet", "Transformer", "GCN", "Bert", "GraphSage", "ResNet18", "ResNet50",, "VGG16"
    model_list=["MobileNetv2","AlexNet", "Transformer", "GCN", "Bert", "GraphSage", "ResNet18", "ResNet50", "VGG16"]#"ResNet50", "MobileNetv2", "VGG16",  "Transformer", "GCN" 
    
    
    #记录代码开始时间
    now_time = datetime.datetime.now()
    formatted_time = now_time.strftime('%m_%d_%H_%M_%S')
    
    out_file_name="ExpPre_compareMPS_True_"+formatted_time+".txt"
    file_writer_true=open(get_output_dir()+out_file_name,"w")
    
    out_file_name="ExpPre_compareMPS_True_alloc_"+formatted_time+".txt"
    file_writer_true_alloc=open(get_output_dir()+out_file_name,"w")
    
    out_file_name="ExpPre_compareMPS_False_"+formatted_time+".txt"
    file_writer_false=open(get_output_dir()+out_file_name,"w")
    
    for model_name in model_list:
        with_mps=True
        do_experiment(with_mps, max_parallel_num,file_writer_true,alloc=False )
        do_experiment(with_mps, max_parallel_num,file_writer_true_alloc,alloc=True )
        with_mps=False
        do_experiment(with_mps, max_parallel_num,file_writer_false,alloc=False )
        
    file_writer_true.close()
    file_writer_true_alloc.close()
    file_writer_false.close()
     
    stop_MPS(11)
    
    
    




