from WeaveAnalyzer import offline_analyze





if __name__=="__main__":
    model_name_list=["AlexNet"]#"AlexNet","ResNet18","ResNet50","MobileNetv2","VGG16"
        #"ResNet18","ResNet50","MobileNetv2","VGG16", "GCN", "GraphSage","Transformer", "Bert"
    offline_analyze(max_parrallel=4,model_name_list=model_name_list,system="Weave", net_card="eno1", total_epochs=2, gpu_id_list=[[0,1,2,3]])