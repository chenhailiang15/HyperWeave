from util import *
import gpustat
import subprocess


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


def run_command(command):
    
    start_time_t=time.time()
    temp_end_time=0
    
    print(f"command: {command}")
    end_flage=False
    back=os.system(command)
    end_flage=True
    if back==0:
        print("succeed end")   
    else:
        print("model run wrong!")
        exit(-1)  
    return  


def get_nvidia_smi_value(gpu_id_t):
    gpus = gpustat.GPUStatCollection.new_query()
    gpu = gpus.gpus[gpu_id_t]  # 获取第一个GPU的信息
    utilization = gpu.utilization  # GPU利用率
    return utilization
def get_dcgm_sm_ave_value(gpu_id_t):
    command=f"dcgmi dmon -e 1002 -i {gpu_id_t}"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
    stdout, stderr = process.communicate() 
    print(stdout)
    return 1
    
def get_dcgm_gpu_util_value(gpu_id_t):
    command=f"dcgmi dmon -e 203 -i {gpu_id_t}"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
    stdout, stderr = process.communicate() 
    print(stdout)
    return 1
    
    
def recored_tools(file_writer):
    file_writer.write(f"nvidia-smi_gpu_util,dcgm_gpu_util,dcgm_sm\n")
    while end_flage==False:
        nvidias_smi_gpu_util=get_nvidia_smi_value(gpu_id)
        
        dcgm_gpu_util=get_dcgm_gpu_util_value(gpu_id)
        dcgm_sm_gpu_util=get_dcgm_sm_ave_value(gpu_id)
        file_writer.write(f"{nvidias_smi_gpu_util},{dcgm_gpu_util},{dcgm_sm_gpu_util}\n")
        time.slee(1)
        
    
    
 #/usr/local/NVIDIA-Nsight-Compute-2024.1   
    
gpu_id=0
end_flage=False

model_info_list=[['ResNet50', 512, 5],['MobileNetv2', 32, 1]]
now_time = datetime.datetime.now()
formatted_time = now_time.strftime('%m_%d_%H_%M_%S')

for model_info in model_info_list:
    out_file_name=f"ExpPre_compareGPUtools_{model_info[0]}_{model_info[1]}_{model_info[2]}.txt"
    file_writer=open(get_output_dir()+out_file_name,"w")
    recored_tools(file_writer)
    command=generate_command(model_info, 0)
    run_command(command)


