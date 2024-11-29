import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.net2 = nn.Sequential(nn.Linear(10, 10), nn.LayerNorm(10))
        self.net3 = nn.Linear(10, 10)
        self.layer_norm = nn.LayerNorm(10)

    def forward(self, x):
        return self.layer_norm(self.net3(self.net2(self.net1(x))))


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank=rank-2
    # if rank == 0:
    print(f"local rank: {local_rank}, world size: {dist.get_world_size()}")
    torch.cuda.set_device(local_rank)
    model = DummyModel().to(local_rank)
    print("start DDP...")
    ddp_model = DDP(model, device_ids=[local_rank])
    print("end DDP...")
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    for i in range(1000):
        print(i)
        outputs = ddp_model(torch.randn(20, 10).to(local_rank))
        labels = torch.randn(20, 10).to(local_rank)
        loss_fn(outputs, labels).backward()
        optimizer.step()
        if rank == 0 and i % 100 == 0:
            print(f"Iteration: {i/1000 * 100} %")
    if rank == 0:
        print("Training completed.")


if __name__ == '__main__':
    main()
