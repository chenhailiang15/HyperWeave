#!/usr/bin/env python
## -*- coding: utf-8 -*-
import pynvml
import torch
import subprocess
import psutil
import gpustat
import time
import threading 
from WeaveSynchronizer import Synchronizer
import numpy as np



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
        

    #正常运行，记录CPU，GPU，跨机器IO等
    def run(self):
        global netIn, netOut
        
        subTread_record=threading.Thread(target=self.get_netIO,args=(self.net_card, self.sample_interval*10, self.unit, self.event))
        subTread_record.start()

        file=open(self.out_dir+self.out_file_name,"w")
        start_time=time.time()
        while not self.event.is_set():
            cpu_util=self.get_cpu_util()
            mem_util=self.get_mem_util()
            file.write(cpu_util.__str__()+","+mem_util.__str__())
            file.write(","+netIn+","+netOut)
            if self.print_flage:
                print("cpu:"+cpu_util.__str__()+"\tmem:"+mem_util.__str__(), end="")
                print("\tnetIn:"+netIn.__str__()+"\tnetOut:"+netOut.__str__(), end="")
                
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

    #用于分析运行，记录各个阶段的资源数据
    def run_analyze(self,queue, shm_name):
        self.gpu_id_analyze=0
        self.sync_er=Synchronizer(shm_name,shm_size=4)
        file=open(self.out_dir+self.out_file_name,"w")
        
        self.cpu=[]
        self.mem=[]
        self.gpu_util=[]
        self.gpu_mem=[]
        
        self.start_flage=True
        self.record_flage=False
        subthread=threading.Thread(target=self.run_analyze_sub_threading,args=())
        subthread.start()
        
        while not self.event.is_set():
            if queue.qsize()>0:
                label=queue.get()
                out_list=[0]*4
                #表明需要开始记录
                while True:
                    #这一个一定能捕获到
                    if self.sync_er.get_value(0) == True:
                        out_list[0]=self.run_analyze_wait_record(0)
                    if self.sync_er.get_value(1) == True:
                        out_list[1]=self.run_analyze_wait_record(1)
                    if self.sync_er.get_value(2) == True:
                        out_list[2]=self.run_analyze_wait_record(2)
                    if self.sync_er.get_value(3) == True:
                        out_list[3]=self.run_analyze_wait_record(3)
                        break
                file.write(label+"-"+out_list.__str__()+"\n")  
                file.flush()
        self.start_flage=False
        file.close()
        self.sync_er.delete_shm()


    def run_analyze_sub_threading(self):
        while self.start_flage:
            cpu_temp=self.get_cpu_use_abs()
            mem_temp=self.get_mem_use_abs()
            gpu_temp=self.get_gpu_util_1(self.gpu_id_analyze)
            gmem_temp=self.get_gmem_use_abs(self.gpu_id_analyze)
            if self.record_flage:
                self.cpu.append(cpu_temp)
                self.mem.append(mem_temp)
                self.gpu_util.append(gpu_temp)
                self.gpu_mem.append(gmem_temp)
        return

    def run_analyze_wait_record(self,index):
        
        start_time=time.time()
        self.record_flage = True
        while self.sync_er.get_value(index) == True:
            a=1
        self.record_flage=False
        end_time=time.time()
        time_t=round(end_time-start_time,2)
        (cpu,mem,gpu,gmem)=self.run_analyze_get_ave_value()
        return (cpu,mem,gpu,gmem,time_t)
    
    def run_analyze_get_ave_value(self):
        cpu_ave=round(np.mean(self.cpu),2) if len(self.cpu)>0 else 0
        mem_ave=round(np.mean(self.mem),2) if len(self.mem)>0 else 0
        gpu_ave=round(np.mean(self.gpu_util),2) if len(self.gpu_util)>0 else 0
        gmem_ave=round(np.mean(self.gpu_mem),2) if len(self.gpu_mem)>0 else 0
        self.cpu=[]
        self.mem=[]
        self.gpu_util=[]
        self.gpu_mem=[]
        return (cpu_ave,mem_ave, gpu_ave,gmem_ave)
        
    
        
        
    
        
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
                #     print(interface+":"+netIn+","+netOut,end="\t")
                # print("")
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

    #单位核
    def get_cpu_use_abs(self):
        cpu_util=self.get_cpu_util()
        return round(cpu_util*cpu_count()/100,2)
    #单位MB
    def get_mem_use_abs(self):
        memory=psutil.virtual_memory()
        return round(memory.used/1024/1024,2)
    
    #单位MB
    def get_gmem_use_abs(self,gpu_id):
        gpu_device = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        usedMemory = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).used
        return round(usedMemory/1024/1024, 2)
        
    def run_abs(self):
        while True:
            cpu=self.get_cpu_use_abs()
            mem=self.get_mem_use_abs()
            gmem=self.get_gmem_use_abs(0)
            gpu=self.get_gpu_util_1(0)
            print(f"cpu use(core):{cpu}\tmem use(MB):{mem}\tgmem use(MB):{gmem}\tgpu(per):{gpu}")
            

from multiprocessing import cpu_count

if __name__=="__main__":

    event=threading.Event()
    # event.set()
    out_dir="../output/"
    out_file_name="Recorder_test.csv"
    print_flage=True
    recorder=Record(gpu_id=-1,net_card="eno1",sample_interval=1,out_dir=out_dir, out_file_name=out_file_name,event=event,print_flage=print_flage)
    recorder.run_abs()
    print("end")
    


