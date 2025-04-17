# Describtion
This is the master version, which will be updated continuously.
The code is for Paper: HyperWeave: Enabling GPU Overcommitment for Maximizing Resource Utilization of Deep Learning Training Jobs



# How to use for cluster
## preparing environment
- run:
cd pre_env/
source pre_env.sh


## run code
The system needs to run in the created conda virtual environment named: torch_mp

- run:
conda activate torch_mp
sh start_weave



***
# Note
- The following files need to be added manually ( contached with chenhailiang_15@163.com )
    - SQuAD_train_features.pkl
    - model.safetensors