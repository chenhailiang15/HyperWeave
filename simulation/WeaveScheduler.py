import math
import random
import os
import threading
from util import *
import copy
import time
from blossom import Blossom_Same


class WeaveSchedulor:
    def __init__(self,master, print_level=0):
        
        self.master=master
        self.strategy=master.schedule_strategy
        self.gpu_list=[i for i in range(7)]
        self.print_level=print_level
        self.paired_instance_list=[]
        self.paired_instance_name=[]
        
    
    def do_schedule(self,job_list):
        # if self.print_level>2:
        print(f"start schedule ({self.strategy})...")
            
        if self.master.system=="Weave":
            rest_job=self.schedule_weave(job_list)
        elif self.master.system=="Muri":
            rest_job=self.schedule_muri(job_list)
        elif self.master.system == "Normal":
            rest_job=self.schedule_normal(job_list)
        else:
            print(f"system name wrong! {self.master.system} (should be Weave, Muri or Normal)")
        print(f"end schedule ({self.strategy})!!!")
        return rest_job
    
    #Weave 调度主线
    def schedule_weave(self,job_list):
        # if self.master.env.now>3038583:
        #     a=0
        if self.strategy=="BN-SRSF":
            all_bucket={}
            for job in job_list:
                bucket_id =math.floor(job.arrive_time/self.master.bucket_length)
                if bucket_id not in all_bucket:
                    all_bucket[bucket_id]=[job]
                else:
                    all_bucket[bucket_id].append(job)





            # self.schedule_weave_BN_SRSF(all_matched_instance_list, job_list)
        else:
            instance_group=self.schedule_weave_classify_instance_based_on_gpu_num(job_list)
            all_matched_instance_list=[]
            print("divide to group")
            for parallel_num in instance_group:
                sub_instance_list=instance_group[parallel_num]
                #将已经匹配过的，进行剔除
                for i in range(len(sub_instance_list)-1, -1, -1):
                    if sub_instance_list[i].instance_name in self.paired_instance_name:
                        sub_instance_list.remove(sub_instance_list[i])
                print(f"match one group parallel:{parallel_num}, list num: {len(sub_instance_list)}")
                matched_instance_list=self.schedule_weave_match_instances(sub_instance_list)
                #将新匹配成功的，加入调度器记录(进行记录的需要从原始列表中删除)
                for i in range(len(matched_instance_list) - 1, -1, -1):
                    [instance1, instance2, pack_resource] = matched_instance_list[i]
                    if instance2 != None:
                        self.paired_instance_list.append([instance1, instance2, pack_resource])
                        self.paired_instance_name.append(instance1.instance_name)
                        self.paired_instance_name.append(instance2.instance_name)
                        matched_instance_list.remove([instance1, instance2, pack_resource])
                #合并得到新的list
                all_matched_instance_list.extend(matched_instance_list)
            all_matched_instance_list.extend(self.paired_instance_list)
            print("start strategy...")
            if self.strategy=="FIFO":
                self.schedule_weave_FIFO(all_matched_instance_list, job_list)
            elif self.strategy=="SRTF":
                self.schedule_weave_SRTF(all_matched_instance_list, job_list)
            elif self.strategy=="SRSF":
                self.schedule_weave_SRSF(all_matched_instance_list, job_list)

            else:
                print("strategy wrong!")
                exit(-1)

        return job_list
    
    def schedule_weave_FIFO(self, matched_instances, job_list):
        order_matched_jobs=[]
        for [instance1,instance2,pack_resource] in matched_instances:
            if instance2 != None:
                arrive_time=min(instance1.job.arrive_time, instance2.job.arrive_time)
            else:
                arrive_time=instance1.job.arrive_time
            order_matched_jobs.append([instance1, instance2, pack_resource, arrive_time])
        #matched_jobs排序
        order_matched_jobs.sort(key=lambda x : x[3])
        print("order over...")
        self.schedule_weave_ordered_matched_job_list(order_matched_jobs, job_list)

    
    
    
    def schedule_weave_SRTF(self, matched_instances, job_list):
        order_matched_jobs=[]
        time_now=time.time()

        for [instance1,instance2,pack_resource] in matched_instances:
            if instance2 != None:
                rest_time1=instance1.job.ddl_time-time_now-instance1.duration_time
                rest_time2=instance2.job.ddl_time-time_now-instance2.duration_time
                rest_time=min(rest_time1, rest_time2)
            else:
                rest_time=instance1.job.ddl_time-time_now-instance1.duration_time
            order_matched_jobs.append([instance1, instance2, pack_resource, rest_time])
        
        #match_jobs排序
        order_matched_jobs.sort(key=lambda x : x[3])
        self.schedule_weave_ordered_matched_job_list(order_matched_jobs, job_list)

        
        
    def schedule_weave_SRSF(self, matched_instances, job_list):
        order_matched_jobs=[]
        time_now=time.time()

        for [instance1,instance2,pack_resource] in matched_instances:
            if instance2 != None:
                rest_time1=(instance1.job.ddl_time-time_now-instance1.duration_time)*instance1.job.parallel_num
                rest_time2=(instance2.job.ddl_time-time_now-instance2.duration_time)*instance2.job.parallel_num
                rest_time=min(rest_time1, rest_time2)
            else:
                rest_time=instance1.job.ddl_time-time_now-instance1.duration_time
            order_matched_jobs.append([instance1, instance2, pack_resource, rest_time])
        
        #match_jobs排序
        order_matched_jobs.sort(key=lambda x : x[3])
        self.schedule_weave_ordered_matched_job_list(order_matched_jobs, job_list)

    def schedule_weave_BN_SRSF(self, matched_instances, job_list):
        order_matched_jobs = []
        time_now = time.time()

        for [instance1, instance2, pack_resource] in matched_instances:
            if instance2 != None:
                rest_time1 = (instance1.job.ddl_time - time_now - instance1.duration_time) * instance1.job.parallel_num
                rest_time2 = (instance2.job.ddl_time - time_now - instance2.duration_time) * instance2.job.parallel_num
                rest_time = min(rest_time1, rest_time2)
            else:
                rest_time = instance1.job.ddl_time - time_now - instance1.duration_time
            order_matched_jobs.append([instance1, instance2, pack_resource, rest_time])

        # match_jobs排序
        order_matched_jobs.sort(key=lambda x: x[3])
        self.schedule_weave_ordered_matched_job_list(order_matched_jobs, job_list)


    def schedule_weave_classify_instance_based_on_gpu_num(self, job_list):
        instance_group = {}
        # 将Job根据GPU使用数量打包
        for job in job_list:
            for instance_t in job.instance_list:
                if job.parallel_num in instance_group:
                    instance_group[job.parallel_num].append(instance_t)
                else:
                    instance_group[job.parallel_num] = [instance_t]
        return instance_group

    # # 任务分类 单GPU和多GPU任务
    # def schedule_weave_classify_jobs(self, job_list):
    #     multi_gpu_instances = []
    #     single_gpu_instances = []
    #     for job in job_list:
    #         for instance in job.instance_list:
    #             if job.is_multi_gpu():
    #                 multi_gpu_instances.append(instance)
    #             else:
    #                 single_gpu_instances.append(instance)
    #     return multi_gpu_instances, single_gpu_instances

    # 任务匹配
    def schedule_weave_match_instances(self, instances_list):
        complete_match_list = []
        out_matched_instances_list = []
        if len(instances_list) == 0:
            return out_matched_instances_list
        if len(instances_list) == 1:
            instance1 = instances_list[0]
            pack_resource = [max(instance1.job.used_resource_cpu), max(instance1.job.used_resource_mem), max(instance1.job.used_resource_gpu), max(instance1.job.used_resource_gmem)]
            out_matched_instances_list.append([instance1, None, pack_resource])
            return out_matched_instances_list

        # 计算两两的匹配值
        for i_index in range(len(instances_list)):
            for j_index in range(i_index + 1, len(instances_list)):
                instance1 = instances_list[i_index]
                instance2 = instances_list[j_index]
                epoch1 = instance1.job.total_epochs
                parallel1 = instance1.job.parallel_num
                [cpu10, cpu11, cpu12] = instance1.job.used_resource_cpu
                [mem10, mem11, mem12] = instance1.job.used_resource_mem
                [gpu10, gpu11, gpu12] = instance1.job.used_resource_gpu
                [gmem10, gmem11, gmem12] = instance1.job.used_resource_gmem
                [time10, time11, time12] = instance1.job.get_three_stage_time()

                epoch2 = instance2.job.total_epochs
                parallel2 = instance2.job.parallel_num
                [cpu20, cpu21, cpu22] = instance2.job.used_resource_cpu
                [mem20, mem21, mem22] = instance2.job.used_resource_mem
                [gpu20, gpu21, gpu22] = instance2.job.used_resource_gpu
                [gmem20, gmem21, gmem22] = instance2.job.used_resource_gmem
                [time20, time21, time22] = instance2.job.get_three_stage_time()

                epoch_factor = self.schedule_weave_cal_similarity(epoch1, epoch2)
                parallel_factor = self.schedule_weave_cal_similarity(parallel1, parallel2)
                cpu_factor = self.schedule_weave_cal_similarity(cpu11 + cpu22, cpu12 + cpu21)
                mem_factor = self.schedule_weave_cal_similarity(mem11 + mem22, mem12 + mem21)
                gpu_factor = self.schedule_weave_cal_similarity(gpu11 + gpu22, gpu12 + gpu21)
                gmem_factor = self.schedule_weave_cal_similarity(gmem11 + gmem22, gmem12 + gmem21)
                time_factor = self.schedule_weave_cal_similarity(time11 + time22, time12 + time21)

                pack_resource = [max(cpu10 + cpu20, cpu11 + cpu22, cpu12 + cpu21), max(mem10 + mem20, mem11 + mem22, mem12 + mem21), max(gpu10 + gpu20, gpu11 + gpu22, gpu12 + gpu21), max(gmem10 + gmem20, gmem11 + gmem22, gmem12 + gmem21)]
                if instance2.job.job_idx==75:
                    a=1
                result = self.master.monitor.judge_runable_with_resource(pack_resource, instance1.job.parallel_num, plan_flage=False, init=False)
                if result:
                    similarity = epoch_factor + parallel_factor + cpu_factor + mem_factor + gpu_factor + gmem_factor + time_factor
                else:
                    similarity = 0
                complete_match_list.append([similarity, instance1, instance2, pack_resource])  # 存储匹配值，后续用于匹配

        # 按照匹配值高低进行提取
        matched_instance_name = set()
        single_instance = None
        single_instance_pack_resource = None
        complete_match_list.sort(key=lambda x: x[0], reverse=True)
        for [similarity, instance1, instance2, pack_resource] in complete_match_list:

            if instance1.instance_name not in matched_instance_name and instance2.instance_name not in matched_instance_name:
                if similarity == 0:  # 如果为0，表示当前匹配已经超出机器最大资源容量， 放弃匹配
                    single_instance1_pack_resource = instance1.job.get_max_used_resource()
                    out_matched_instances_list.append([instance1, None, single_instance1_pack_resource])
                    single_instance2_pack_resource = instance2.job.get_max_used_resource()
                    out_matched_instances_list.append([instance2, None, single_instance2_pack_resource])
                else:
                    out_matched_instances_list.append([instance1, instance2, pack_resource])
                matched_instance_name.add(instance1.instance_name)
                matched_instance_name.add(instance2.instance_name)
                continue

            elif instance1.instance_name not in matched_instance_name:
                single_instance = instance1
                single_instance_pack_resource = instance1.job.get_max_used_resource()
            elif instance2.instance_name not in matched_instance_name:
                single_instance = instance2
                single_instance_pack_resource = instance2.job.get_max_used_resource()

        # 判断输出是否包含所有jobs
        if len(matched_instance_name) != len(instances_list):
            if single_instance.instance_name not in matched_instance_name:
                out_matched_instances_list.append([single_instance, None, single_instance_pack_resource])
            else:
                print("schedule_weave_match_instances wrong!")
                exit(256)
        return out_matched_instances_list

    # 匹配值计算的子函数，匹配效果越好，值越接近1
    def schedule_weave_cal_similarity(self, var1, var2):
        if max(var1, var2) == 0:
            return 1
        return 1 - (abs(var1 - var2) / max(var1, var2))



    def schedule_weave_ordered_matched_job_list(self, match_instances, job_list):
        print("start execute...")
        # 是否继续调度的标志，当遇到一个无法调度的任务时，停止调度等待下一轮调度，将剩余的job返回
        continue_schedule_flage = True
        # 循环调度
        for [instance1, instance2, pack_resource, order_value] in match_instances:
            # if len(job1.instance_list)==0 or (job2 != None and len(job2.instance_list)==0):
            #
            #     continue

            if continue_schedule_flage:
                # 正常调度
                # [[node_index, score, [[gpu_id, score],...]],...]
                adjust_pack_resource=[x/instance1.job.parallel_num for x in pack_resource]
                satisfy_gpu_list = self.master.monitor.get_satisfy_gpu_for_sim(adjust_pack_resource)
                # 对优先级进行排序
                # satisfy_gpu_list_new = []
                all_satisfy_gpu_num = 0
                for [node_index, score, temp_gpu_list] in satisfy_gpu_list:
                    # score = score + len(temp_gpu_list) * 10 - self.master.nodes[node_index].get_over_corss_num() * 10
                    # # 调整优先级，score原始是GPU剩余量百分比的和
                    # satisfy_gpu_list_new.append([node_index, score, temp_gpu_list])
                    all_satisfy_gpu_num += len(temp_gpu_list)
                # satisfy_gpu_list_new.sort(key=lambda x: x[1], reverse=True)

                if instance2 != None:
                    need_parallel = instance1.job.parallel_num
                    if all_satisfy_gpu_num < need_parallel:
                        continue_schedule_flage = False
                        continue



                    selected_gpu_id_list, shm_name_dict = self.schedule_weave_select_gpu(need_parallel, satisfy_gpu_list)
                    # scheduled_jobs.append([job1, job1_gpu_id_list, shm_name_dict])
                    # scheduled_jobs.append([job2, job2_gpu_id_list, shm_name_dict])
                    [cpu, mem, gpu, gmem] = pack_resource
                    instance1.set_pack_resource(math.ceil(cpu), math.ceil(mem), math.ceil(gpu), math.ceil(gmem), instance2.instance_name)
                    instance2.set_pack_resource(math.ceil(cpu), math.ceil(mem), math.ceil(gpu), math.ceil(gmem),instance1.instance_name)

                    #更新持续时间
                    self.schedule_weave_updata_duration_time(instance1, instance2)
                    self.master.monitor.alloc_resource(instance1, instance2, selected_gpu_id_list)
                    #将两个instance从记录中删除
                    for i in range(len(self.paired_instance_list)-1,-1,-1):
                        if self.paired_instance_list[i][0].instance_name == instance1.instance_name:
                            self.paired_instance_list.remove(self.paired_instance_list[i])
                            self.paired_instance_name.remove(instance1.instance_name)
                            self.paired_instance_name.remove(instance2.instance_name)
                            break

                    instance1.job.instance_list.remove(instance1)
                    if len(instance1.job.instance_list) == 0:
                        # print(f"remove:{instance1.job.job_idx}with instance2 {instance2.instance_name} ins1: {instance1.instance_name}")
                        job_list.remove(instance1.job)

                    self.execute_schedule(instance1, selected_gpu_id_list, True, shm_name_dict)

                    instance2.job.instance_list.remove(instance2)
                    if len(instance2.job.instance_list) == 0:
                        # print(f"remove:{instance2.job.job_idx} with instance1 {instance1.instance_name} ins1: {instance2.instance_name}")
                        job_list.remove(instance2.job)

                    self.execute_schedule(instance2, selected_gpu_id_list, False, shm_name_dict)

                    print(f"instance:{instance1.instance_name} couple with instance: {instance2.instance_name}")
                    # if self.master.weave_sync_mode == False: 仿真系统难以准确建模同步不同步，以及是否启用MPS对系统的影响，因此，凡是Weave，都默认启动
                    #     shm_name_dict = {}



                else:
                    need_parallel = instance1.job.parallel_num
                    if all_satisfy_gpu_num < need_parallel:
                        continue_schedule_flage = False
                        continue

                    selected_gpu_id_list,_ = self.schedule_weave_select_gpu(need_parallel, satisfy_gpu_list)
                    # scheduled_jobs.append([job1, job1_gpu_id_list, {}])
                    [cpu, mem, gpu, gmem] = pack_resource
                    instance1.set_pack_resource(math.ceil(cpu), math.ceil(mem), math.ceil(gpu), math.ceil(gmem))

                    instance1.job.instance_list.remove(instance1)
                    if len(instance1.job.instance_list) == 0:
                        # print(f"remove{instance1.job.job_idx}")
                        job_list.remove(instance1.job)


                    self.master.monitor.alloc_resource(instance1, None, selected_gpu_id_list)
                    self.execute_schedule(instance1, selected_gpu_id_list, False, {})

            else:
                return

    def schedule_weave_updata_duration_time(self, instance1, instance2):
        # 初始化rest_batch_num
        one_epoch_time=max(instance1.job.time_init_iter, instance2.job.time_epoch_no_init_iter)+ \
                       max(instance2.job.time_init_iter, instance1.job.time_epoch_no_init_iter)
        start_time = instance1.job.time_init + max(instance2.job.time_init, instance1.job.time_init_iter) + \
                     max(instance2.job.time_init_iter, instance1.job.time_epoch_no_init_iter)
        cur_time=0
        if instance1.job.total_epochs<=instance2.job.total_epochs:

            cur_time+=start_time
            cur_time+= one_epoch_time*(instance1.job.total_epochs-1)
            instance1.duration_time=cur_time
            cur_time+=instance2.job.time_epoch_no_init_iter
            cur_time+=(instance2.job.time_init_iter+instance2.job.time_epoch_no_init_iter)*(instance2.job.total_epochs-instance1.job.total_epochs)
            instance2.duration_time=cur_time
        else:
            cur_time += start_time
            cur_time += one_epoch_time * (instance2.job.total_epochs - 1)
            instance2.duration_time = cur_time+ instance2.job.time_epoch_no_init_iter
            cur_time += (instance1.job.time_init_iter + instance1.job.time_epoch_no_init_iter) * (instance1.job.total_epochs - instance2.job.total_epochs)
            instance1.duration_time = cur_time


        print(f"instance {instance1.instance_name}, init duration {instance1.job.duration_time}, new duration {instance1.duration_time}")
        print(f"instance {instance2.instance_name}, init duration {instance2.job.duration_time}, new duration {instance2.duration_time}")
        return
    
    def schedule_weave_select_gpu(self, rest_gpu,  satisfy_gpu_list):

        selected_gpu_id_list = []
        shm_name_dict = {}
        for [node_index, score, temp_gpu_list] in satisfy_gpu_list:

            temp_gpu_id_list = []
            shm_name_dict_temp = {}
            for [gpu_index, ave_per] in temp_gpu_list:
                if rest_gpu > 0 :
                    shm_name = generate_shm_name()
                    shm_name_dict_temp[gpu_index] = shm_name
                    temp_gpu_id_list.append(gpu_index)
                    rest_gpu -= 1
                elif rest_gpu == 0 :
                    break
                else:
                    print("(monitor) wrong!")
                    exit(256)
            selected_gpu_id_list.append([node_index, temp_gpu_id_list])
            shm_name_dict[node_index] = shm_name_dict_temp
            if rest_gpu == 0 :
                break
        return selected_gpu_id_list, shm_name_dict
    
    #muri 调度主线
    def schedule_muri(self,job_list):
        print("start schedule_muri... ")
        self.global_idx_to_instance={}
        all_matched_instance_list=[]
        instance_group={}
        match_instance_num=0
        need_match_instance_num=0
        #将Job根据GPU使用数量打包
        for job in job_list:
            for instance in job.instance_list:
                if instance.instance_name not in self.paired_instance_name:
                    need_match_instance_num+=1
                    self.global_idx_to_instance[instance.instance_global_idx]=instance
                    instance_mini={}
                    instance_mini["num_gpu"]=job.parallel_num
                    instance_mini['resource_time']=job.get_four_stage_time()
                    instance_mini['job_idx']=instance.instance_global_idx
                    instance_mini['iteration_time']=job.total_epochs

                    if job.parallel_num in instance_group:
                        instance_group[job.parallel_num].append(instance_mini)
                    else:
                        instance_group[job.parallel_num]=[instance_mini]
        print("start Blossom_Same... ")
        packings=Blossom_Same.run(instance_group, self.master.monitor.get_idle_gpu_num())
        print("end Blossom_Same... ")
        succeed_matched_instance_idx=set()
        faile_matched_instance_idx=set()
        for gpu_num in packings:
            # print(f"blossom gpu num:{gpu_num} ... ")
            
            for _pack in packings[gpu_num]:    #obj _pack : _Packing  这里面每个循环是一个匹配
                matched_instance=[]
                matched_flage=True
                # print(f"one pack:\t", end="")
                for instance_mini_t in _pack.best_permutation: #这里总共是一个匹配
                    # print(job_t.job_idx,end="\t")
                    matched_instance.append(self.global_idx_to_instance[instance_mini_t.job_idx])
                    #如果有一个已经属于被匹配了的，本次匹配失败，后续单独处理
                    if instance_mini_t.job_idx in succeed_matched_instance_idx:
                        matched_flage=False
                # print("")
            
            
                #处理一个匹配
                if matched_flage==True:

                    if self.schedule_muri_is_max_resource_satisfy(matched_instance): #判断资源是否足够，足够才能算匹配成功

                        # 将匹配后，两个及以上成功匹配的，放入缓存
                        # assert len(matched_instance)>=2
                        if len(matched_instance)<4:
                            all_matched_instance_list.append(matched_instance)
                        else:
                            self.paired_instance_list.append(matched_instance)
                            for instance in matched_instance:
                                self.paired_instance_name.append(instance.instance_name)

                        match_instance_num+=len(matched_instance)
                        # print(f"0000000000000000000000000000000000000000000000000 match num:  {len(matched_job)}")
                        for instance in matched_instance:
                            succeed_matched_instance_idx.add(instance.instance_global_idx)
                    else:
                        for instance in matched_instance:
                            faile_matched_instance_idx.add(instance.instance_global_idx)
                else:
                    for instance in matched_instance:
                        faile_matched_instance_idx.add(instance.instance_global_idx)
        print("end match... ")
        #将失败的单独调度
        for instance_global_idx in faile_matched_instance_idx:
            if instance_global_idx not in succeed_matched_instance_idx:
                all_matched_instance_list.append([self.global_idx_to_instance[instance_global_idx]])
                match_instance_num+=1


        # 这里是个判断，判断上述匹配是否已经完成所有匹配
        assert match_instance_num == need_match_instance_num
        all_matched_instance_list.extend(self.paired_instance_list)
        # try:
        #     assert match_instance_num == len(job_list)
        # except:
        #     for job in job_list:
        #         if job.job_idx not in faile_matched_job_idx and job.job_idx not in succeed_matched_job_idx:
        #             print(f"******************fix blossom with add job:{job.job_idx}")
        #             all_matched_job_list.append([job])
        print("start strategy... ")
        if self.strategy=="FIFO":
            self.schedule_muri_FIFO(all_matched_instance_list, job_list)
        elif self.strategy=="SRTF":
            self.schedule_muri_SRTF(all_matched_instance_list, job_list)
        elif self.strategy=="SRSF":
            self.schedule_muri_SRSF(all_matched_instance_list, job_list)
        else:
            print("strategy wrong!")
            exit(-1)

        return job_list
    
    def schedule_muri_is_max_resource_satisfy(self, matched_instance):
        max_cpu=0
        all_need_mem=0
        max_gpu=0
        parallel_num=matched_instance[0].job.parallel_num
        for instance in matched_instance:
            [plan_cpu, plan_mem, plan_gpu]=instance.job.get_plan_resource()
            max_cpu=max(plan_cpu,max_cpu)
            all_need_mem+=plan_mem
            max_gpu=max(plan_gpu,max_gpu)

        pack_resource=[max_cpu, all_need_mem, max_gpu, 0]
        flage= self.master.monitor.judge_runable_with_resource(pack_resource, parallel_num, plan_flage=True, init=False)
        if flage:
            print(f"匹配好 {len(matched_instance)}，资源够")
        else:
            print(f"匹配好 {len(matched_instance)}，资源不够")
        return flage
        
    
    def schedule_muri_FIFO(self, all_matched_instance_list, job_list):
        order_matched_instances=[]
        for instance_list in all_matched_instance_list:
            arrive_time=float('inf')
            for instance in instance_list:
                arrive_time=min(arrive_time, instance.job.arrive_time)
            order_matched_instances.append([arrive_time, instance_list])
        #matched_jobs排序
        order_matched_instances.sort(key=lambda x : x[0])
        print("start schedule ordered_matched_job_list... ")
        self.schedule_muri_ordered_matched_job_list(order_matched_instances, job_list)

    
    def schedule_muri_SRTF(self, all_matched_instance_list, job_list):
        order_matched_instances=[]
        time_now=time.time()
        for instance_list in all_matched_instance_list:
            rest_time=float('inf')
            for instance in instance_list:
                rest_time_t=instance.job.ddl_time-time_now-instance.job.duration_time
                rest_time=min(rest_time, rest_time_t)
            order_matched_instances.append([rest_time, instance_list])
        #matched_jobs排序
        order_matched_instances.sort(key=lambda x : x[0])
        self.schedule_muri_ordered_matched_job_list(order_matched_instances, job_list)

    
    def schedule_muri_SRSF(self, all_matched_instance_list, job_list):
        order_matched_instances=[]
        time_now=time.time()
        for instance_list in all_matched_instance_list:
            rest_time=float('inf')
            for instance in instance_list:
                rest_time_t=(instance.job.ddl_time-time_now-instance.job.duration_time)**instance.job.parallel_num
                rest_time=min(rest_time, rest_time_t)
            order_matched_instances.append([rest_time, instance_list])
        #matched_jobs排序
        order_matched_instances.sort(key=lambda x : x[0])
        self.schedule_muri_ordered_matched_job_list(order_matched_instances, job_list)





    def schedule_muri_ordered_matched_job_list(self, match_instances, job_list):
        #是否继续调度的标志，当遇到一个无法调度的任务时，停止调度等待下一轮调度，将剩余的job返回
        continue_schedule_flage=True
        #循环调度
        print(f"schedule_muri_ordered_matched_job_list cycle-{len(match_instances)}")
        temp_cycle=0
        for [order_value, instance_list] in match_instances:
            # print(f"cycle - {temp_cycle}")
            temp_cycle+=1
            if continue_schedule_flage:
                #正常调度
                need_gpu_num = instance_list[0].job.parallel_num
                all_need_cpu = 0
                all_need_mem = 0

                for instance in instance_list:
                    [plan_cpu, plan_mem, plan_gpu] = instance.job.get_plan_resource()
                    all_need_cpu += plan_cpu
                    all_need_mem += plan_mem

                pack_resource = [all_need_cpu/need_gpu_num, all_need_mem/need_gpu_num, 100, 0]
                #[[node_index, score, [[gpu_id, score],...]],...]
                satisfy_gpu_list=self.master.monitor.get_satisfy_gpu_for_sim(pack_resource)
                #对优先级进行排序
                
                all_satisfy_gpu_num=0
                for [node_index, score, temp_gpu_list] in satisfy_gpu_list:
                    all_satisfy_gpu_num+=len(temp_gpu_list)
                
                if all_satisfy_gpu_num< need_gpu_num:
                    continue_schedule_flage=False
                    continue
                
                instance_gpu_id_list=[]
                shm_name_dict={}
                
                for [node_index, score, temp_gpu_list] in satisfy_gpu_list:
                    instance_temp_gpu_id_list=[]
                    shm_name_dict_temp={}
                    for [gpu_id,value] in temp_gpu_list:
                        
                        if need_gpu_num>0:
                            instance_temp_gpu_id_list.append(gpu_id)
                            shm_name=generate_shm_name()
                            shm_name_dict_temp[gpu_id]=shm_name
                            
                            need_gpu_num-=1
                        else:
                            break
                    if len(instance_temp_gpu_id_list)>0:
                        instance_gpu_id_list.append([node_index, instance_temp_gpu_id_list])
                        shm_name_dict[node_index]=shm_name_dict_temp
                
                idx_on_gou=0
                
                if len(instance_list)==1:
                    shm_name_dict={}

                self.schedule_muri_update_duration_time(instance_list)
                # 将两个instance从记录中删除
                if len(instance_list)>=2:
                    for i in range(len(self.paired_instance_list) - 1, -1, -1):
                        if self.paired_instance_list[i][0].instance_name == instance_list[0].instance_name:
                            self.paired_instance_list.remove(self.paired_instance_list[i])
                            for instance in instance_list:
                                self.paired_instance_name.remove(instance.instance_name)
                            break

                for instance in instance_list:
                    instance.idx_on_gou=idx_on_gou
                    instance.max_sync_num=len(instance_list)
                    idx_on_gou+=1
                    print(f"start do instance:{instance.instance_name}")


                    self.master.monitor.alloc_resource(instance, None, instance_gpu_id_list, plan=True)

                    instance.job.instance_list.remove(instance)
                    if len(instance.job.instance_list) == 0:
                        print(f"remove: job- {instance.job.job_idx}")
                        job_list.remove(instance.job)

                    self.execute_schedule(instance, instance_gpu_id_list, False, shm_name_dict)

    def schedule_muri_update_duration_time(self, instance_list):
        if len(instance_list) == 1:
            return
        temp_instance_list=instance_list[:]
        #初始化rest_batch_num
        for instance in temp_instance_list:
            instance.rest_batch_num = instance.job.total_epochs * instance.job.batch_num-1
            instance.init_iter_num=1

        cur_time=self.schedule_muri_update_duration_time_get_start_time(instance_list)
        acc_batch_num=1
        while len(temp_instance_list)!=0:
            #找到最先结束的instance
            min_batch_num=float("inf")
            early_end_instance=None
            for instance in temp_instance_list:
                if instance.rest_batch_num<min_batch_num:
                    min_batch_num=instance.rest_batch_num
                    early_end_instance=instance
            assert min_batch_num>=0

            update_batch_time=self.schedule_muri_update_duration_time_get_batch_time(temp_instance_list)
            cur_time+=update_batch_time*min_batch_num
            #计算init iter 次数和时间。直接相加


            for instance in temp_instance_list:
                extern_init_iter_num=math.floor((min_batch_num+acc_batch_num)/instance.job.batch_num)-instance.init_iter_num
                cur_time+=extern_init_iter_num*instance.job.time_init_iter
                instance.init_iter_num+=extern_init_iter_num
                instance.rest_batch_num-=min_batch_num

            early_end_instance.duration_time=cur_time
            temp_instance_list.remove(early_end_instance)
            acc_batch_num+=min_batch_num
            print(f"instance {early_end_instance.instance_name}, init duration {early_end_instance.job.duration_time}, new duration {early_end_instance.duration_time}")
        return

    def schedule_muri_update_duration_time_get_batch_time(self,instance_list):
        if len(instance_list) ==1:
            return instance_list[0].job.four_times[1]+instance_list[0].job.four_times[2]+instance_list[0].job.four_times[3]
        if len(instance_list)==2:
            batch_time=0
            for j in range(1,4):
                i=j%3+1
                batch_time+=max(instance_list[0].job.four_times[i], instance_list[1].job.four_times[j])
            return batch_time
        if len(instance_list)==3:
            batch_time=0
            for k in range(1,4):
                j=k%3+1
                i=j%3+1
                batch_time+=max(instance_list[0].job.four_times[i], instance_list[1].job.four_times[j], instance_list[2].job.four_times[k])
            return batch_time
        if len(instance_list)==4:
            batch_time=0
            for k in range(0,4):
                j=(k+1)%4
                i=(j+1)%4
                batch_time+=max(instance_list[i].job.four_times[1], instance_list[j].job.four_times[2], instance_list[k].job.four_times[3])
            return batch_time

    def schedule_muri_update_duration_time_get_start_time(self, instance_list):
        if len(instance_list)==2:
            time0=instance_list[0].job.four_times[0]
            time1=max(instance_list[0].job.time_init_iter, instance_list[1].job.four_times[0])
            time2=max(instance_list[0].job.four_times[2], instance_list[1].job.time_init_iter)
            time3=max(instance_list[0].job.four_times[3], instance_list[1].job.four_times[2])
            return time0+time1+time2+time3
        if len(instance_list)==3:
            time0 = instance_list[0].job.four_times[0]
            time1 = max(instance_list[0].job.time_init_iter, instance_list[1].job.four_times[0])
            time2 = max(instance_list[0].job.four_times[2], instance_list[1].job.time_init_iter, instance_list[2].job.four_times[0])
            time3 = max(instance_list[0].job.four_times[3], instance_list[1].job.four_times[2], instance_list[2].job.time_init_iter)
            return time0 + time1 + time2 + time3
        if len(instance_list)==4:
            time0 = instance_list[0].job.four_times[0]
            time1 = max(instance_list[0].job.time_init_iter, instance_list[1].job.four_times[0])
            time2 = max(instance_list[0].job.four_times[2], instance_list[1].job.time_init_iter, instance_list[2].job.four_times[0])
            time3 = max(instance_list[0].job.four_times[3], instance_list[1].job.four_times[2], instance_list[2].job.time_init_iter, instance_list[3].job.four_times[0])
            time4 = max(instance_list[1].job.four_times[3], instance_list[2].job.four_times[2], instance_list[3].job.time_init_iter)

            return time0 + time1 + time2 + time3 + time4




            
    def execute_schedule(self, instance, select_gpu_list, prior, shm_name_dict):
        world_size=0
        nprocs_list=[0]*self.master.node_num
        gpu_id_list=[ [] for i in range(self.master.node_num)]  #需要有顺序

        min_node_index=float("inf")    #选取最小的node index作为
        for [node_index, gpu_list] in select_gpu_list:
            
            world_size+=len(gpu_list)
            nprocs_list[node_index]=len(gpu_list)
            gpu_id_list[node_index]=gpu_list
            if len(gpu_list)>0:
                min_node_index=node_index if node_index<min_node_index else min_node_index
            
        main_ip=self.master.nodes[min_node_index].ip
        main_temp_port=self.master.nodes[min_node_index].get_idle_port()

        for [node_index, gpu_list] in select_gpu_list:
            if len(gpu_list)==0:    #如果对应GPU list没有被选择，则不用将Job发送到Node，不然，会导致任务重复
                continue
            instance_t=copy.deepcopy(instance)
            instance_t.job=instance.job
            if node_index == min_node_index:
                instance_t.set_is_main(True)
            else:
                instance_t.set_is_main(False)
        #
            net_card=self.master.nodes[node_index].net_card
            instance_t.set_execute_info(main_ip, main_temp_port, net_card, node_index, world_size ,nprocs_list, gpu_id_list, prior=prior, shm_name_list=shm_name_dict)
            self.master.send_instance_to_execution(instance_t)
    
    
    
    #********************************************************************************Normal***********************************************************************************************
    #normal 调度主线
    def schedule_normal(self,job_list):
        if self.strategy=="FIFO":
            rest_job=self.schedule_normal_FIFO(job_list)
        elif self.strategy=="SRTF":
            rest_job=self.schedule_normal_SRTF(job_list)
        elif self.strategy=="SRSF":
            rest_job=self.schedule_normal_SRSF(job_list)
        else:
            print("strategy wrong!")
            exit(-1)

        return rest_job
    
    
    
    def schedule_normal_FIFO(self, job_list):
        #按照到来的先后顺序排序
        job_list.sort(key=lambda x: x.arrive_time)
        rest_job=self.schedule_normal_single_job_list(job_list)
            
        return rest_job
    
    def schedule_normal_SRTF(self, job_list):
        time_now=time.time()
        #按照到来的先后顺序排序
        job_list.sort(key=lambda x: x.ddl_time-time_now-x.duration_time)
        rest_job=self.schedule_normal_single_job_list(job_list)
            
        return rest_job
    
    def schedule_normal_SRSF(self, job_list):
        
        time_now=time.time()
        # for job in job_list:
        #     job.set_schedule_order(time_now)
            
        #按照到来的先后顺序排序
        job_list.sort(key=lambda x: (x.ddl_time-time_now-x.duration_time)*x.parallel_num)
        rest_job=self.schedule_normal_single_job_list(job_list)
        return rest_job


    def schedule_normal_single_job_list(self, job_list):
        #初始化未被调度的job
        rest_job=[]
        #是否继续调度的标志，当遇到一个无法调度的任务时，停止调度等待下一轮调度，将剩余的job返回
        continue_schedule_flage=True
        #循环调度
        for job in job_list:
            if continue_schedule_flage:
                #正常调度
                for index in range(len(job.instance_list)):

                    if job.plan_gpu<=100:
                        plan_resource=[job.plan_cpu, job.plan_mem, job.plan_gpu, 0]
                    else:
                        factor=job.plan_gpu / 100
                        plan_resource = [job.plan_cpu / factor, job.plan_mem / factor, 100, 0]

                    satisfy_gpu_list=self.master.monitor.get_satisfy_gpu_for_sim(plan_resource)
                    all_satisfy_gpu_num=0

                    for [node_index, score, temp_gpu_list] in satisfy_gpu_list:
                        all_satisfy_gpu_num+=len(temp_gpu_list)
                    if all_satisfy_gpu_num>=job.parallel_num:
                        selected_gpu_id_list,_= self.schedule_weave_select_gpu(job.parallel_num, satisfy_gpu_list)
                        instance=job.instance_list.pop(0)
                        job.dealing_instance_num+=1
                        self.master.monitor.alloc_resource(instance, None, selected_gpu_id_list, plan=True)
                        self.execute_schedule(instance, selected_gpu_id_list, False, {})
                    else:
                        rest_job.append(job)
                        continue_schedule_flage=False
                        break
            else:
                rest_job.append(job)
            assert len(rest_job)<= len(job_list)
        return rest_job

    # def place_jobs_to_nodes(self, job_list, mode="plan") :
    #
    #
    #
    #     job1_gpu_id_list=[]
    #     job2_gpu_id_list=[]
    #     shm_name_dict={}
    #     for [node_index, score, temp_gpu_list] in satisfy_gpu_list_new:
    #         job1_temp_gpu_id_list=[]
    #         job2_temp_gpu_id_list=[]
    #         shm_name_dict_temp={}
    #         for [gpu_index, ave_per] in temp_gpu_list:
    #             if job1_rest_gpu>0 and job2_rest_gpu>0:
    #                 shm_name=generate_shm_name()
    #                 shm_name_dict_temp[gpu_index]=shm_name
    #                 job1_temp_gpu_id_list.append(gpu_index)
    #                 job2_temp_gpu_id_list.append(gpu_index)
    #                 job1_rest_gpu-=1
    #                 job2_rest_gpu-=1
    #             elif job1_rest_gpu>0 and job2_rest_gpu==0:
    #                 job1_temp_gpu_id_list.append(gpu_index)
    #                 job1_rest_gpu-=1
    #             elif job1_rest_gpu==0 and job2_rest_gpu>0:
    #                 job2_temp_gpu_id_list.append(gpu_index)
    #                 job2_rest_gpu-=1
    #             elif job1_rest_gpu==0 and job2_rest_gpu==0:
    #                 break
    #             else:
    #                 print("(monitor) wrong!")
    #                 exit(256)
    #         job1_gpu_id_list.append([node_index, job1_temp_gpu_id_list])
    #         job2_gpu_id_list.append([node_index, job2_temp_gpu_id_list])
    #         shm_name_dict[node_index]=shm_name_dict_temp
    #         if job1_rest_gpu==0 and job2_rest_gpu==0:
    #             break
    #     return job1_gpu_id_list,job2_gpu_id_list,shm_name_dict
        
        
    