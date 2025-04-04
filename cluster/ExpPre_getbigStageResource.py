# 分析生成每个模型大阶段资源消耗和耗时

from WeaveAnalyzer import offline_analyze


if __name__=="__main__":
    max_parrallel=4
    #"AlexNet","GraphSage","Transformer", "Bert", "ResNet18","ResNet50","MobileNetv2","VGG16", "GCN"
    model_name_list=["ResNet18","AlexNet","GraphSage","Transformer", "Bert", "ResNet50","MobileNetv2","VGG16", "GCN"]
    gpu_id_list=[[0,1,2,3]]
    #s4*3090 -> net_card = "enp3s0"
    #others -> net_card = "en01"
    offline_analyze(max_parrallel=max_parrallel,model_name_list=model_name_list,net_card="eno1", system="Weave", total_epochs=1, gpu_id_list=gpu_id_list)