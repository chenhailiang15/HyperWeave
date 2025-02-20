from WeaveMaster import *
from util import *
import datetime


def generate_default_args():
    args=args_weave()
    args.system="Weave"
    args.strategy="BN-SRSF"
    args.mps_flage="True"
    args.sync_flage="True"
    args.job_together_flage="False"
    args.print_level=1
    args.node_kind="cluster"
    args.model_kind="all_model"
    args.gpu_mem_percent=0.9
    args.node_num=200
    args.job_num=2000
    args.validation="False"
    args.overshared_factor=3
    args.write_sum=False
    args.write_trace=False
    args.couple_init_iter_percent=0.2
    return args


def get_out_file_writer():
    now_time= datetime.datetime.now()
    formatted_time = now_time.strftime('%m_%d_%H_%M_%S')
    out_file_name="Sim_bucket_length-"+version+"_"+formatted_time+".txt"
    cur_dir=os.path.dirname(os.path.abspath(__file__))
    parent_dir= os.path.dirname(os.path.abspath(cur_dir))
    file_writer=open(parent_dir+"/output/"+out_file_name,"w")
    return file_writer


version="v1.0.0"
file_writer=get_out_file_writer()
file_writer.write("JCT:")
for bucket_length in [100,1000,10000,100000]:
    
    
    print(f"matching_factor:{bucket_length}",end="")
    
    args=generate_default_args()
    args.bucket_length=bucket_length
    result_dict=run_system(args)
    jct_value=result_dict["JCT"]
    print(f"\tJCT:{jct_value}")
    file_writer.write(f"({bucket_length},{jct_value}),")
    file_writer.flush()
    
    
sum_info=result_dict["sum_info"]
file_writer.write(f"\n{sum_info}")
file_writer.close()