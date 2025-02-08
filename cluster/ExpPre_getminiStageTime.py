from WeaveAnalyzer import offline_analyze





if __name__=="__main__":
    max_parrallel=4
    model_name_list=["AlexNet", "GCN", "GraphSage","Transformer", "Bert", "ResNet18","ResNet50","MobileNetv2","VGG16"]
    gpu_id_list=[[4,5,6,7]]

    offline_analyze(max_parrallel=max_parrallel,model_name_list=model_name_list,system="Muri", total_epochs=2, gpu_id_list=gpu_id_list)