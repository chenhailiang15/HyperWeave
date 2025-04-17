import sys
sys.path.append("..")
sys.path.append("platform_h")
from platform_h.HyperWeaveMaster import *
from platform_h.util import *
import datetime
import multiprocessing

def generate_default_args():
    args=args_hyperweave()
    # args.system="HyperWeave"
    # args.strategy="BN-SRSF"
    args.mps_flage="True"
    args.sync_flage="True"
    # args.job_together_flage="False"
    args.print_level=0
    args.node_kind="cluster"
    args.model_kind="all_model"
    args.gpu_mem_percent=0.9
    args.node_num=100
    args.job_num=10000
    args.validation="False"
    args.overshared_factor=2
    args.write_sum=True
    args.write_trace=True
    args.couple_init_iter_percent=0.2
    args.job_come_time_factor=2
    args.bucket_length=100000000
    # args.trace_id=0
    
    return args


def run_once(system, strategy, job_together_flage, trace_id, mps_flage):
    print(f"start once {system}, {strategy}, {job_together_flage}, {trace_id}, {mps_flage}")
    args=generate_default_args()
    args.system=system
    args.strategy=strategy
    args.job_together_flage=job_together_flage
    args.trace_id=trace_id
    args.mps_flage=mps_flage
    args_copy=copy.deepcopy(args)
    run_system = Runsystem()
    run_system.run(args_copy)
    print(f"End once {system}, {strategy}, {job_together_flage}, {trace_id}, {mps_flage}")
    
    



task_args_list=[]

for trace_id in [5,6,7,8]:
    for job_together_flage in ["True", "False"]:#
        for system in ["Normal","Muri", "HyperWeave"]: #"Normal","Muri", "HyperWeave"
            if system=="Normal":
                for strategy in ["FIFO","SRTF", "SRSF"]:
                    task_args_list.append((system, strategy, job_together_flage, trace_id,"False"))
                    if strategy == "SRSF":
                        task_args_list.append((system, strategy, job_together_flage, trace_id,"True"))
            
            elif system == "HyperWeave":
                strategy="BN-SRSF"
                task_args_list.append((system, strategy, job_together_flage, trace_id,"True"))
                
            elif system =="Muri":
                strategy = "SRSF"
                if job_together_flage=="True":
                        continue
                task_args_list.append((system, strategy, job_together_flage, trace_id,"True"))

with multiprocessing.Pool(processes=10) as pool:
        # 使用 starmap 方法将任务分配给进程池中的进程执行
        pool.starmap(run_once, task_args_list)


    
print("End all sub threadings")
             
