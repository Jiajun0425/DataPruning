from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from data_loader import load_data
from model import ConvNet
from utils import *


def train(model, device, train_loader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        # 使用tqdm显示训练进度
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='mean')
            # print(loss)
            loss.backward()
            optimizer.step()
            # 更新进度条的显示信息
            progress_bar.set_postfix(loss=loss.item())


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # 使用tqdm显示测试进度
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # 更新进度条的显示信息
            progress_bar.set_postfix(loss=test_loss / len(test_loader.dataset), accuracy=100. * correct / len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)')


if __name__ == "__main__":
    seed = 0
    pruning_rate = 0.1
    data_dir = "./data"
    model_path = f"./checkpoints/pruning_rate={pruning_rate}/seed={seed}"
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, _ = load_data(data_dir, pruning_rate)

    model = ConvNet().to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, device, train_loader, optimizer)
    test(model, device, test_loader)

    save_model(model_path, model)