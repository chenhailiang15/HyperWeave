import kagglehub
import os
import pickle
from transformers.data.processors.squad import SquadV2Processor, squad_convert_examples_to_features
from transformers import BertTokenizer

cur_dir=os.path.dirname(os.path.abspath(__file__))

def tiny_ImageNet_download():
    
    
    if os.path.exists(cur_dir+"/tiny-ImageNet"):
        print("ImageNet exits, return!")
        return
    # Download latest version
    print("start download ImageNet...")
    path = kagglehub.dataset_download("akash2sharma/tiny-imagenet")
    try:
        print("move imagenet...")
        os.system(f"mv {path}/* {cur_dir}")
        os.system(f"mv {cur_dir}/tiny-imagenet-200 {cur_dir}/tiny-ImageNet")
        tiny_imagenet_dir=path.split("/tiny-imagenet/")[0]
        os.system(f"rm -rf {tiny_imagenet_dir}")
    except:
        print("wrong!")
    print("Path to dataset files:", path)

tiny_ImageNet_download()


def squad_convert_to_fit_bert_features():
    if os.path.exists(cur_dir+"/SQuAD_train_features.pkl"):
        print("SQuAD_train_features exits, return!")
        return
    # 初始化SQuAD Processor, 数据集, 和分词器
    processor = SquadV2Processor()
    # cur_dir_path = os.path.abspath(os.curdir)
    # root_path=os.path.dirname(current_path)
    # path = os.path.join(root_path,"dataset")
    train_examples = processor.get_train_examples(cur_dir)
    tokenizer = BertTokenizer(vocab_file=cur_dir+"/vocab.txt")


    # 将SQuAD 2.0示例转换为BERT输入特征
    train_features = squad_convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=True,
        return_dataset=False,
        threads=1
    )

    # 将特征保存到磁盘上
    with open(cur_dir+'/SQuAD_train_features.pkl', 'wb') as f:
        pickle.dump(train_features, f)

print("start deal squad...")
squad_convert_to_fit_bert_features()
