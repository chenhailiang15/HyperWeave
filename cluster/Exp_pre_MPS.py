from util import *
from models.Framework import *


def do_experiment(mps_state):
    # if mps_state ==True:
    #     if not start_MPS(password):
    #         print("MPS open wrong")
    #         exit(-1)
    # else:
    #     if not stop_MPS(password):
    #         print("MPS close wrong")
    #         exit(-1)
    
    for _ in range(2):
        for model_name in model_list:
            for para_num in range(1, max_parallel_num+1):
                
                run_model(model_name, para_num)
                file_writer.write(f"{model_name},mps={mps_state},para_num={para_num},time_list={end_time_list}\n")
                file_writer.flush()
 
 
def run_model(model_name,para_num):
    global end_time_list
    end_time_list=[]
    thread_hand=[]
    
    for index in range(para_num):
        
        command = generate_command(model_name, index)
        sub_thread=threading.Thread(target=run_command,args=(command, model_name))
        sub_thread.start()
        thread_hand.append(sub_thread)

    for thread_t in thread_hand:
        thread_t.join()
    


    
    
def generate_command(model_name, index):
    nprocs_list="[1,0]"

    gpu_id_list="[[7],[]]"
    net_card="eno1"

    total_epochs=2
    if model_name == "Bert":
        batch_size=8
    else:
        batch_size=16           #8 for Bert (default:16)
    

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


def run_command( command, model_name):
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
    




with_mps=True   #这个参数需要手动进行
max_parallel_num=10


password="sim2024"
now_time    = datetime.datetime.now()
formatted_time = now_time.strftime('%m_%d_%H_%M_%S')
out_file_name="Exp_pre_MPS_"+str(with_mps)+"_"+formatted_time+".txt"

model_list=["AlexNet", "Transformer", "GCN", "Bert", "GraphSage", "ResNet18", "ResNet50", "MobileNetv2", "VGG16", ]#"ResNet50", "MobileNetv2", "VGG16",  "Transformer", "GCN" 
file_writer=open(get_output_dir()+out_file_name,"w")

end_time_list=[]
port_id=2000


do_experiment(with_mps)
# do_experiment(False)
file_writer.close()




