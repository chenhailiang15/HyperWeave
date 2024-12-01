import pynvml
import torch
import subprocess
import psutil
import gpustat
import time



class Record:

    def __init__(self,gpu_id,sample_interval,out_dir,out_file_name,event):
        print("start a record object...")
        self.gpu_id=gpu_id
        self.sample_interval=sample_interval
        self.out_dir=out_dir
        self.out_file_name=out_file_name
        self.event=event
        self.pynvml=pynvml.nvmlInit()


    def run(self):
        file=open(self.out_dir+self.out_file_name,"w")
        start_time=time.time()
        while not self.event.is_set():
            cpu_util=self.get_cpu_util()
            mem_util=self.get_mem_util()
            file.write(cpu_util.__str__()+","+mem_util.__str__())
            #print("cpu:",cpu_util,"\tmem:",mem_util,end="")
            if self.gpu_id==-1:
                for i in range(torch.cuda.device_count()):
                    gpu_util=self.get_gpu_util_1(i)
                    gpu_mem_util=self.get_gpu_mem_util(i)
                    file.write(","+gpu_util.__str__()+","+gpu_mem_util.__str__())
                    #print("\tgpu:"+i.__str__(),"-",gpu_util,"\tgmem:"+i.__str__(),"-",gpu_mem_util,end="")
            
            else:
                gpu_util=self.get_gpu_util_1(self.gpu_id)
                gpu_mem_util=self.get_gpu_mem_util(self.gpu_id)
                file.write(","+gpu_util.__str__()+","+gpu_mem_util.__str__())
                #print("\tgpu:"+i.__str__(),"-",gpu_util,"\tgmem:"+i.__str__(),"-",gpu_mem_util,end="")
            file.write("\n")
            # print()
            # global current_epoch_num
            file.flush()
        end_time=time.time()
        file.write((round(end_time-start_time,2)).__str__())
        file.close()

    def get_cpu_util(self):
        cpu_usage=psutil.cpu_percent(interval=self.sample_interval)
        return cpu_usage


    def get_gpu_util_1(self,gpu_id):
        gpus = gpustat.GPUStatCollection.new_query()
        gpu = gpus.gpus[gpu_id]  # 获取第一个GPU的信息
        utilization = gpu.utilization  # GPU利用率
        return utilization
        # print(f"GPU Utilization 1:", utilization)

    def get_gpu_util_2(self):

        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        utilization = result.stdout.strip().split('\n')

        print(f"GPU Utilization 2:",utilization[0])

    def get_mem_util(self):
        memory=psutil.virtual_memory()
        memory_usage=memory.percent
        # print("Mem utilization:", memory_usage)
        return memory_usage
    
    
    def get_gpu_mem_util(self,gpu_id):

        gpu_device = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        # get GPU memory total
        totalMemory = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).total
        # get GPU memory used
        usedMemory = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).used
        # UtilizationRates = pynvml.nvmlDeviceGetUtilizationRates(gpu_device)

        # print(gpu_id.__str__()+"  - 总显存: {:.2f} MB".format(totalMemory / (1024 ** 2)))
        # print(gpu_id.__str__()+"  - 已分配的显存: {:.2f} MB".format(usedMemory / (1024 ** 2)))
        # print(gpu_id.__str__()+"  - 利用率: {:.2f} ".format(UtilizationRates.gpu))
        return round(usedMemory/totalMemory*100,2)
        # if torch.cuda.is_available():
        #     # 获取CUDA设备数量
        #     device_count = torch.cuda.device_count()
        #     # print("CUDA可用，共有 {} 个CUDA设备可用:".format(device_count))

        #     device = torch.device("cuda:{}".format(gpu_id))
        #     # print("CUDA 设备 {}: {}".format(i, torch.cuda.get_device_name(i)))

        #     # 获取当前设备的显存使用情况
        #     total_memory = torch.cuda.get_device_properties(device).total_memory
        #     allocated_memory = torch.cuda.memory_allocated(device)  # 已分配的显存
        #     reserved_memory = torch.cuda.memory_reserved(device)  # 已保留的显存
        #     free_memory = total_memory - allocated_memory - reserved_memory  # 剩余可用显存
        #     print(gpu_id.__str__()+"  - 总显存: {:.2f} MB".format(total_memory / (1024 ** 2)))
        #     print(gpu_id.__str__()+"  - 已分配的显存: {:.2f} MB".format(allocated_memory / (1024 ** 2)))
        #     print(gpu_id.__str__()+"  - 已保留的显存: {:.2f} MB".format(reserved_memory / (1024 ** 2)))
        #     print(gpu_id.__str__()+"  - 剩余可用显存: {:.2f} MB".format(free_memory / (1024 ** 2)))

            # return round(allocated_memory/total_memory*100,2)
        
