#!/bin/bash
pip_install_data_pre()
{
    pip install transformers==4.29.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install kagglehub==0.2.9 -i https://pypi.tuna.tsinghua.edu.cn/simple
}

pip_install_requirement()
{
    pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install torch_scatter-2.0.7-cp38-cp38-linux_x86_64.whl
    pip install torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl
    echo "waiting for data init..."
}

# source环境
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

my_conda_env="torch_mp"

if conda env list | grep $my_conda_env; then
    echo $my_conda_env "existed!"
else
    echo $my_conda_env "not existed!"
    echo y| conda create -n $my_conda_env python=3.8
fi
# 激活虚拟环境
conda activate $my_conda_env 
pip_install_data_pre
pip_install_requirement & python ../dataset/prepare_data.py

echo "current environment:" | conda info | grep "active environment"
echo "environment and data init over!"




