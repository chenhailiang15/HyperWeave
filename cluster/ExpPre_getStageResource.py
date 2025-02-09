from WeaveAnalyzer import offline_analyze


if __name__=="__main__":
    max_parrallel=2
    #"AlexNet","GraphSage","Transformer", "Bert", "ResNet18","ResNet50","MobileNetv2","VGG16", "GCN"
    model_name_list=[ "GCN"]
    gpu_id_list=[[4,5,6,7]]
    
    offline_analyze(max_parrallel=max_parrallel,model_name_list=model_name_list,net_card="eno1", system="Weave", total_epochs=2, gpu_id_list=gpu_id_list)