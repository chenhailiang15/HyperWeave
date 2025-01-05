import sys
sys.path.append("../..")
from cluster.util import *
import datetime
import random


def generate_model_with_full_info(spec_string, number):
    file_name="Full_model_info_"+spec_string+".txt"
    file_writer=open(file_name,"w")
    for _ in range(number):
        model_name=random.choice(model_list_g)
        batch_size=random.choice(model_to_batch_size_g[model_name])
        file_writer.write(f"{model_name}-{batch_size}\n")
    file_writer.close()
    return
    
    
    
    
    
now_time = datetime.datetime.now()
formatted_time = now_time.strftime('%m_%d_%H_%M_%S')  
info_num=1000

generate_model_with_full_info(formatted_time, info_num)
print("generate over!")
# print(model_to_batch_size_g)