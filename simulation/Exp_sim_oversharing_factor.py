from WeaveMaster import *
from util import *
import datetime
import multiprocessing

def generate_default_args():
    args=args_weave()
    args.system="Weave"
    args.strategy="BN-SRSF"
    args.mps_flage="True"
    args.sync_flage="True"
    args.job_together_flage="False"
    args.print_level=0
    args.node_kind="cluster"
    args.model_kind="all_model"
    args.gpu_mem_percent=0.9
    args.node_num=100
    args.job_num=10000
    args.validation="False"
    args.overshared_factor=3
    args.write_sum=True
    args.write_trace=True
    args.couple_init_iter_percent=0.2
    args.job_come_time_factor=2
    args.bucket_length=100000000
    return args




def run_once(overshared_factor, trace_id, gpu_kind):
    print(f"start once {overshared_factor}, {trace_id}, {gpu_kind}")
    args=generate_default_args()
    args.overshared_factor=overshared_factor
    args.trace_id=trace_id
    args.gpu_kind=gpu_kind
    args_copy=copy.deepcopy(args)
    run_system = Runsystem()
    run_system.run(args_copy)
    print(f"End once {overshared_factor}, {trace_id}, {gpu_kind}")
    

task_args_list=[]
for gpu_kind in ["P100","V100M32","A100M80"]:
    for overshared_factor in np.arange(1,  5.01, 0.1):
        overshared_factor=round(overshared_factor,1)
        for trace_id in range(10):
            print((overshared_factor, trace_id, gpu_kind))
            task_args_list.append((overshared_factor, trace_id, gpu_kind))
    
    
    
with multiprocessing.Pool(processes=40) as pool:
        # 使用 starmap 方法将任务分配给进程池中的进程执行
        result=pool.starmap(run_once, task_args_list)
        
print("End all sub threadings")