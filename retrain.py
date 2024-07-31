from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from data_loader import load_data
from model import ConvNet
from utils import *


def compute_samples_gradients(data_loader, model, device):
    model.eval()
    all_gradients = []

    # 利用批处理，但对每个样本分别计算梯度
    for data, target in tqdm(data_loader, desc="Processing batches"):
        data, target = data.to(device), target.to(device)

        # 处理整个批次，但反向传播时只关注一个样本
        outputs = model(data)
        losses = F.cross_entropy(outputs, target, reduction='none')

        # 对每个样本单独进行反向传播
        for i in range(len(losses)):
            model.zero_grad()
            losses[i].backward(retain_graph=True if i < len(losses) - 1 else False)

            # 收集并存储当前样本的梯度
            sample_gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    sample_gradients.extend(param.grad.view(-1).detach().cpu().numpy())
            all_gradients.append(np.array(sample_gradients))

    return np.array(all_gradients)


def compute_gradients(data_loader, model, device):
    model.eval()
    all_gradients = []

    for data, target in tqdm(data_loader, desc="Processing batches"):
        data, target = data.to(device), target.to(device)

        outputs = model(data)
        loss = F.cross_entropy(outputs, target)

        model.zero_grad()
        loss.backward()

        sample_gradients = []
        for param in model.parameters():
            if param.grad is not None:
                sample_gradients.extend(param.grad.view(-1).detach().cpu().numpy())
        all_gradients.append(np.array(sample_gradients))

    return np.array(all_gradients)


def find_coefficients(train_gradients, other_gradients):
    """ 使用线性最小二乘法找到最佳系数。"""
    coefficients, _, _, _ = np.linalg.lstsq(train_gradients.T, other_gradients.T, rcond=None)
    return coefficients


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
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader, other_loader = load_data(data_dir, pruning_rate)

    surrogate_model = ConvNet().to(device)
    surrogate_model = load_model(model_path, surrogate_model)

    train_gradients = compute_gradients(train_loader, surrogate_model, device)
    other_gradients = compute_gradients(other_loader, surrogate_model, device)

    print(train_gradients.shape)
    print(other_gradients.shape)

    coefficients = find_coefficients(train_gradients, other_gradients)
    coefficients = np.sum(coefficients, axis=1) + 1
    print("Coefficients Shape:", coefficients.shape)

    model = ConvNet().to(device)
    model.apply(init_weights)

    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train(model, device, train_loader, optimizer)
    # test(model, device, test_loader)