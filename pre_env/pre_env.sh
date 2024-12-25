#!/bin/bash

pip_install_requirement()
{
    echo "function chl"
    pip install -r requirement.txt
    pip install torch_scatter-2.0.7-cp38-cp38-linux_x86_64.whl
    pip install torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl
}


my_conda_env="torch_chl"

if conda env list | grep $my_conda_env; then
    echo $my_conda_env "existed!"
    source activate 
    conda deactivate
    conda activate $my_conda_env
    pip_install_requirement
else
    echo $my_conda_env "not existed!"
    echo y| conda create -n $my_conda_env python=3.8
    source activate 
    conda deactivate
    conda activate $my_conda_env
    pip_install_requirement
fi

echo | conda info | grep "active environment"
echo "environment init over!"




    # pip install torch==1.9.1+cu111 -f  https://download.pytorch.org/whl/cu111/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple