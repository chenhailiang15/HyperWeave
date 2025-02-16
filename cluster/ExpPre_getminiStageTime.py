# 分析生成每个模型小阶段耗时
from WeaveAnalyzer import offline_analyze





if __name__=="__main__":
    max_parrallel=4
    #"AlexNet", "GCN", 
    model_name_list=["ResNet18","AlexNet", "GraphSage","Transformer", "Bert", "ResNet50","MobileNetv2","VGG16", "GCN"]
    gpu_id_list=[[4,5,6,7]]
    #s4*3090 -> net_card = "enp3s0"
    #others -> net_card = "en01"
    offline_analyze(max_parrallel=max_parrallel, model_name_list=model_name_list,net_card="eno1", system="Muri", total_epochs=2, gpu_id_list=gpu_id_list)