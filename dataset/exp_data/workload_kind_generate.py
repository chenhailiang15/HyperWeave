import sys
sys.path.append("../..")
from cluster.util import *
import datetime
import random


def generate_model_with_full_info(spec_string, number, is_full_mode=False):
    if is_full_mode:
        file_name="Full_model_info_"+spec_string+".txt"
        temp_model_list=model_list_g
    else:
        file_name="CV_model_info_"+spec_string+".txt"
        temp_model_list=cv_model_list_g
    file_writer=open(file_name,"w")
    for _ in range(number):
        model_name=random.choice(temp_model_list)
        batch_size=random.choice(model_to_batch_size_g[model_name])
        file_writer.write(f"{model_name}-{batch_size}\n")
    file_writer.close()
    return
    
    
    
    
    
now_time = datetime.datetime.now()
formatted_time = now_time.strftime('%m_%d_%H_%M_%S')  
info_num=1000

generate_model_with_full_info(formatted_time, info_num, is_full_mode=False)
print("generate over!")
# print(model_to_batch_size_g)