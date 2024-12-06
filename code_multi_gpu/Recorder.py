import pynvml
import torch
import subprocess
import psutil
import gpustat
import time
import threading 

netIn='0.00'
netOut='0.00'

class Record:

    def __init__(self,gpu_id,net_card,sample_interval,out_dir,out_file_name,event,print_flage=False):
        print("start a record object...")
        self.gpu_id=gpu_id
        self.net_card=net_card
        self.sample_interval=sample_interval
        self.out_dir=out_dir
        self.out_file_name=out_file_name
        self.event=event
        self.pynvml=pynvml.nvmlInit()
        self.print_flage=print_flage
        self.unit="M"


    def run(self):
        global netIn, netOut
        
        subTread_record=threading.Thread(target=self.get_netIO,args=(self.net_card, self.sample_interval*2, self.unit, self.event))
        subTread_record.start()

        file=open(self.out_dir+self.out_file_name,"w")
        start_time=time.time()
        while not self.event.is_set():
            cpu_util=self.get_cpu_util()
            mem_util=self.get_mem_util()
            file.write(cpu_util.__str__()+","+mem_util.__str__())
            file.write(","+netIn+","+netOut)
            if self.print_flage:
                print("cpu:",cpu_util,"\tmem:",mem_util,end="")
                print("\tnetIn:"+netIn+"\tnetOut:"+netOut,end="")
                
            if self.gpu_id==-1:
                for i in range(torch.cuda.device_count()):
                    gpu_util=self.get_gpu_util_1(i)
                    gpu_mem_util=self.get_gpu_mem_util(i)
                    file.write(","+gpu_util.__str__()+","+gpu_mem_util.__str__())
                    if self.print_flage:
                        print("\tgpu:"+i.__str__(),"-",gpu_util,"\tgmem:"+i.__str__(),"-",gpu_mem_util,end="")
            
            else:
                gpu_util=self.get_gpu_util_1(self.gpu_id)
                gpu_mem_util=self.get_gpu_mem_util(self.gpu_id)
                file.write(","+gpu_util.__str__()+","+gpu_mem_util.__str__())
                if self.print_flage:
                    print("\tgpu:"+i.__str__(),"-",gpu_util,"\tgmem:"+i.__str__(),"-",gpu_mem_util,end="")
            if self.print_flage:
                print()
            file.write("\n")
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
    
    def get_netIO(self, interface_name, interval_time, unit, event):
        global netIn, netOut
        
        # 初始化
        interfaces, _, _ = self.getNetworkData()
        #开始循环监控网卡流量
        while not event.is_set():
            _, networkIn, networkOut = self.getNetworkRate(interval_time)
            for interface in interfaces:
                # if interface != "lo" and bool(1 - interface.startswith("veth")) and bool(
                #         1 - interface.startswith("蓝牙")) and bool(1 - interface.startswith("VMware")):
                if interface == interface_name:
                    if unit == "K" or unit == "k":
                        netIn = "%.2f" % (networkIn.get(interface) / 1024)#KB/s
                        netOut = "%.2f" % (networkOut.get(interface) / 1024)
                    elif unit == "M" or unit == "m":
                        netIn = "%.2f" % (networkIn.get(interface) / 1024 / 1024)#MB/s
                        netOut = "%.2f" % (networkOut.get(interface) / 1024 / 1024)
                    elif unit == "G" or unit == "g":
                        netIn = "%.3f" % (networkIn.get(interface) / 1024 / 1024 / 1024)#GB/s
                        netOut = "%.3f" % (networkOut.get(interface) / 1024 / 1024 / 1024)
                    else:
                        netIn = "%.1f" % networkIn.get(interface)#B/s
                        netOut = "%.1f" % networkOut.get(interface)
        # print("sub sub theading out")

    def getNetworkData(self):
        # 获取网卡流量信息
        recv = {}
        sent = {}
        data = psutil.net_io_counters(pernic=True)
        interfaces = data.keys()
        for interface in interfaces:
            recv.setdefault(interface, data.get(interface).bytes_recv)
            sent.setdefault(interface, data.get(interface).bytes_sent)
        return interfaces, recv, sent


    def getNetworkRate(self,interval_time):
        # 计算网卡流量速率
        interfaces, oldRecv, oldSent = self.getNetworkData()
        time.sleep(interval_time)
        interfaces, newRecv, newSent = self.getNetworkData()
        networkIn = {}
        networkOut = {}
        for interface in interfaces:
            networkIn.setdefault(interface, float("%.3f" % ((newRecv.get(interface) - oldRecv.get(interface)) / interval_time)))
            networkOut.setdefault(interface, float("%.3f" % ((newSent.get(interface) - oldSent.get(interface)) / interval_time)))
        return interfaces, networkIn, networkOut





if __name__=="__main__":

    event=threading.Event()
    # event.set()
    out_dir="../output/"
    out_file_name="Recorder_test.csv"
    print_flage=False
    recorder=Record(gpu_id=-1,net_card="eno1",sample_interval=1,out_dir=out_dir, out_file_name=out_file_name,event=event,print_flage=print_flage)
    recorder.run()
    print("end")
    


