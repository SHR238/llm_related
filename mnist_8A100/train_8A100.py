import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp


def setup_ddp(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost' # 主节点地址
    os.environ['MASTER_PORT'] = '12355'     # 主节点端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    

def build_dataloaders(per_gpu_batch_size: int, 
                      root: str = "data", 
                      rank: int = None, world_size: int = None,
                      per_gpu_num_workers: int = 0,
                      ):
    """构造 FashionMNIST 训练集与测试集的 DataLoader"""
    training_data = datasets.FashionMNIST(
        root=root, train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root=root, train=False, download=True, transform=ToTensor()
    )
    
    train_sampler = DistributedSampler(
        training_data, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    test_sampler = DistributedSampler(
        test_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = DataLoader(
        training_data, 
        batch_size=per_gpu_batch_size, # 更新一次模型时放到一个GPU上的数据量
        sampler=train_sampler,
        num_workers=per_gpu_num_workers, # workers是负责从磁盘读取数据到CPU内存的
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=per_gpu_batch_size, 
        sampler=test_sampler,
        num_workers=per_gpu_num_workers,
    )
    
    return train_loader, test_loader


class NeuralNetwork(nn.Module):
    """简单三层 MLP"""
    def __init__(self, in_features: int = 28 * 28, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


def train_one_epoch(dataloader, model, loss_fn, optimizer, device, rank):
    """训练一个 epoch"""
    model.train()
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward() # 只同步梯度，不同步loss标量值
        optimizer.step()
        optimizer.zero_grad()
        
        # 只在rank 0上打印
        if batch % 100 == 0:
            # 创建loss的副本用于通信，避免影响计算图
            loss_tensor = loss.clone().detach()

            # 对所有GPU的loss求平均
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        
            if rank == 0:
                current = batch * len(X) * dist.get_world_size()
                print(f"loss: {loss_tensor.item():>7f}  [{current:>5d}/{size:>5d}]")
    
    loss_tensor = loss.clone().detach()
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    if rank == 0:
        print(f"loss: {loss_tensor.item():>7f}  [{size:>5d}/{size:>5d}]")


def evaluate(dataloader, model, loss_fn, device, rank):
    """在测试集上评估"""
    model.eval()
    size = len(dataloader.dataset)
    test_loss, correct = 0.0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() * len(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    # 汇总所有GPU的结果
    test_loss = torch.tensor(test_loss).to(device)
    correct = torch.tensor(correct).to(device)
    
    dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    
    test_loss = test_loss.item() / size
    correct = correct.item() / size
    
    # 只在rank 0上打印
    if rank == 0:
        print(f"Evaluate: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_ddp(rank, world_size, per_gpu_batch_size, epochs, lr, save_path, per_gpu_num_workers):
    """分布式训练主函数，每个进程都会执行这个函数"""
    # 初始化DDP
    setup_ddp(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"Training on {world_size} GPUs")
    
    # 构建数据加载器(使用DistributedSampler)
    train_dataloader, test_dataloader = build_dataloaders(
        per_gpu_batch_size, 
        rank=rank, 
        world_size=world_size,
        per_gpu_num_workers=per_gpu_num_workers,
    )
    
    # 打印一次样本形状(只在rank 0)
    if rank == 0:
        for X, y in test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break
    
    # 创建模型并用DDP包装
    model = NeuralNetwork().to(device)
    model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        print(model)
    
    loss_fn = nn.CrossEntropyLoss() # 默认 reduction='mean'
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # 训练
    for t in range(epochs):
        if rank == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
        
        # 设置epoch以确保每个epoch的数据打乱方式不同
        train_dataloader.sampler.set_epoch(t)
        
        train_one_epoch(train_dataloader, model, loss_fn, optimizer, device, rank)
        
        evaluate(test_dataloader, model, loss_fn, device, rank)
    
    if rank == 0:
        print("Done!")
        
        # 只在rank 0上保存模型
        # 保存时需要使用model.module来获取原始模型
        torch.save(model.module.state_dict(), save_path)
        print(f"Saved PyTorch Model State to {save_path}")
    
    # 清理DDP
    dist.destroy_process_group()


def inference_single(model, device, classes):
    """对测试集第一张图片做推理(单卡推理)"""
    model.eval()
    test_data = datasets.FashionMNIST(root="data", train=False, transform=ToTensor(), download=False)
    x, y = test_data[0][0], test_data[0][1]
    
    with torch.no_grad():
        x = x.unsqueeze(0).to(device)
        pred = model(x)
        predicted = classes[pred[0].argmax(0).item()]
        actual = classes[y]
    
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


def inference():
    """单卡推理函数"""
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Inference on {device}")
    
    # 加载模型
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    
    classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    
    inference_single(model, device, classes)


def main():
    # ---------------- 超参数 ----------------
    world_size = 8    # 设置为8卡训练
    batch_size = 256
    epochs = 10
    lr = 1e-3
    save_path = "model.pth"
    num_workers = 32  # 总worker数
    # ---------------------------------------
    
    # 启动多进程训练
    print("Starting distributed training...")
    assert batch_size % world_size == 0, f"batch_size无法平均分配到每一片GPU上"
    mp.spawn(
        train_ddp, 
        args=(
            world_size, 
            batch_size//world_size, 
            epochs, 
            lr, 
            save_path,
            num_workers//world_size,
        ), 
        nprocs=world_size,
        join=True, # 等待所有进程完成
    )
    # 解释spawn
    # def spawn(fn, args=(), nprocs=1, join=True, daemon=False, ...):
    #     """
    #     fn: 要在每个进程中执行的函数
    #     args: 传递给fn的额外参数（不包括rank）
    #     nprocs: 要创建的进程数
    #     """
    # for i in range(nprocs):
    #     # 创建新进程并执行：
    #     # fn(rank=i, *args)
    #     # 第一个参数rank由spawn自动提供！
    #     process = Process(target=fn, args=(i,) + args)
    # train_ddp 第一个参数默认是rank，需要传入的参数是从第二个开始

    
    # 训练完成后,在单卡上进行推理
    print("\nStarting inference on single GPU...")
    inference()


if __name__ == "__main__":
    main()