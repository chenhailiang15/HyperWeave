import sys
sys.path.append("..")
sys.path.append("platform_h")
from platform_h.HyperWeaveMaster import *
from platform_h.util import *
import datetime
import multiprocessing

def generate_default_args():
    args=args_hyperweave()
    args.system="HyperWeave"
    args.strategy="BN-SRSF"
    args.mps_flage="True"
    args.sync_flage="True"
    args.job_together_flage="False"
    args.print_level=0
    args.node_kind="cluster"
    args.model_kind="all_model"
    args.gpu_mem_percent=0.9
    args.node_num=4
    args.job_num=10000
    args.validation="False"
    args.overshared_factor=2
    args.write_sum=True
    args.write_trace=True
    args.couple_init_iter_percent=0.2
    args.job_come_time_factor=2
    args.bucket_length=100000000
    args.gpu_kind="V100M32"
    return args


def run_once(strategy, trace_id):
    print(f"start once {strategy}, {trace_id}")
    args=generate_default_args()
    args.strategy=strategy
    args.trace_id=trace_id
    args_copy=copy.deepcopy(args)
    run_system = Runsystem()
    run_system.run(args_copy)
    print(f"End once {strategy}, {trace_id}")
    

task_args_list=[]
for strategy in ["FIFO", "SRTF", "SRSF", "BN-SRSF"]:
    for trace_id in range(10):
        task_args_list.append((strategy, trace_id))
    
    
    
with multiprocessing.Pool(processes=5) as pool:
        # 使用 starmap 方法将任务分配给进程池中的进程执行
        result=pool.starmap(run_once, task_args_list)
        
print("End all sub threadings")