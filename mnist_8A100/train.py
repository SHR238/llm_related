import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def build_dataloaders(batch_size: int, root: str = "data"):
    """构造 FashionMNIST 训练集与测试集的 DataLoader"""
    training_data = datasets.FashionMNIST(
        root=root, train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root=root, train=False, download=True, transform=ToTensor()
    )
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
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


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    """训练一个 epoch"""
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            current = batch * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")


def evaluate(dataloader, model, loss_fn, device):
    """在测试集上评估"""
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Evaluate: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def inference_single(model, device, classes):
    """对测试集第一张图片做推理"""
    model.eval()
    test_data = datasets.FashionMNIST(root="data", train=False, transform=ToTensor(), download=False)
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.unsqueeze(0).to(device)          # 增加 batch 维度
        pred = model(x)
        predicted = classes[pred[0].argmax(0).item()]
        actual = classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


def main():
    # ---------------- 超参数 ----------------
    batch_size = 128
    epochs = 10
    lr = 1e-3
    save_path = "model.pth"
    # ---------------------------------------

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using {device} device")

    train_dataloader, test_dataloader = build_dataloaders(batch_size)

    # 打印一次样本形状
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 训练
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_one_epoch(train_dataloader, model, loss_fn, optimizer, device)
        evaluate(test_dataloader, model, loss_fn, device)
    print("Done!")

    # 保存
    torch.save(model.state_dict(), save_path)
    print(f"Saved PyTorch Model State to {save_path}")

    # 加载并推理
    model_loaded = NeuralNetwork().to(device)
    model_loaded.load_state_dict(torch.load(save_path, weights_only=True))

    classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    inference_single(model_loaded, device, classes)


if __name__ == "__main__":
    main()