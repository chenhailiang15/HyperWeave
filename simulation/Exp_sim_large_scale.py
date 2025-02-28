from WeaveMaster import *
from util import *
import datetime
import threading

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
    args.node_num=200
    args.job_num=1000
    args.validation="False"
    args.overshared_factor=3
    args.write_sum=True
    args.write_trace=True
    args.couple_init_iter_percent=0.2
    args.job_come_time_factor=2
    args.trace_id=0
    
    return args


# def get_out_file_writer():
#     now_time= datetime.datetime.now()
#     formatted_time = now_time.strftime('%m_%d_%H_%M_%S')
#     out_file_name="Sim_bucket_length-"+version+"_"+formatted_time+".txt"
#     cur_dir=os.path.dirname(os.path.abspath(__file__))
#     parent_dir= os.path.dirname(os.path.abspath(cur_dir))
#     file_writer=open(parent_dir+"/output/"+out_file_name,"w")
#     return file_writer

def run_once(system, strategy, job_together_flage, trace_id):
    print(f"start once {system}, {strategy}, {job_together_flage}, {trace_id}")
    args=generate_default_args()
    args.system=system
    args.strategy=strategy
    args.job_together_flage=job_together_flage
    args.trace_id=trace_id
    
    run_system(args)
    print(f"End once {system}, {strategy}, {job_together_flage}, {trace_id}")
    
    

version="v1.0.0"
# file_writer=get_out_file_writer()
# file_writer.write("JCT:")
thread_list=[]
for trace_id in [0,1,2,3]:
    for job_together_flage in ["True", "False"]:
        for system in [ "Normal","Muri", "Weave"]:
            if system == "Weave":
                strategy="BN-SRSF"
                thread_hand=threading.Thread(target=run_once, args=(system, strategy, job_together_flage, trace_id))
                thread_list.append(thread_hand)
                thread_hand.start()
            else:
                for strategy in ["FIFO","SRTF", "SRSF"]:
                    if system=="Muri" and job_together_flage=="True":
                        continue
                    thread_hand=threading.Thread(target=run_once, args=(system, strategy, job_together_flage, trace_id))
                    thread_list.append(thread_hand)
                    thread_hand.start()

for thread_hand in thread_list:
    thread_hand.join()
    
print("End all sub threadings")
             
