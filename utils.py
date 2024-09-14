import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from tqdm import tqdm
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def save_model(path, model):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({'model': model.state_dict()}, os.path.join(path, "model.pth"))


def load_model(path, model):
    state_dict = torch.load(os.path.join(path, "model.pth"))
    model.load_state_dict(state_dict['model'])
    return model


class CustomSubset(Subset):
    '''A custom subset class'''
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.data = torch.tensor(dataset.data)[indices]
        self.targets = torch.tensor(dataset.targets)[indices]

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        return x, y 

    def __len__(self):
        return len(self.indices)

def plot_loss(loss, save_dir):
    plt.figure(figsize=(12, 8))
    plt.plot(loss)
    plt.savefig(os.path.join(save_dir, 'loss.png'))

def plot_frequency(similarity_matrix, save_dir):
    plt.figure(figsize=(12, 8))
    bins = np.round(np.linspace(-1, 1, num=101), 1)
    plt.hist(similarity_matrix.flatten(), bins=bins, color='lightgreen')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.xlim(-1, 1)
    plt.savefig(os.path.join(save_dir, 'frequency.png'))

def plot_coefficient(coefficient, lower_bound, upper_bound, save_dir):
    # coefficient = np.clip(coefficient.flatten(), 0, 6)
    coefficient = coefficient.flatten()
    bins = np.round(np.linspace(lower_bound, upper_bound, num=11), 1)
    xtick_labels = []
    xtick_locs = []
    for i in range(len(bins)-1):
        xtick_labels.append(f'{bins[i]}-{bins[i+1]}')
        xtick_locs.append((bins[i]+bins[i+1])/2)
    plt.figure(figsize=(12, 8))
    plt.hist(coefficient, bins=bins, edgecolor='black', alpha=0.75)
    plt.xticks(xtick_locs, xtick_labels)
    plt.title('Value Distribution')
    plt.xlabel('Value')
    plt.savefig(os.path.join(save_dir, 'coefficient.png'))

def plot_cosine_similarity(similarity_matrix, save_dir):
    plt.figure(figsize=(12, 8))
    # sns.heatmap(similarity_matrix, cmap='viridis', cbar=True)
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Cosine Similarity')
    plt.title('Cosine Similarity Heatmap')
    plt.xlabel('Gradients Index')
    plt.ylabel('Gradients Index')
    plt.savefig(os.path.join(save_dir, 'similarity_matrix.png'))

def trf(train_gradients, other_gradients, lower_bound, upper_bound, device, save_dir, max_iter=10000):
    losses = []
    l1 = 1
    train_gradients = torch.tensor(train_gradients, device=device, dtype=torch.float32)
    other_gradients = torch.tensor(other_gradients, device=device, dtype=torch.float32)

    num_train_batches = train_gradients.shape[0]
    num_other_batches = other_gradients.shape[0]
    p = (1 - lower_bound) / (upper_bound - lower_bound)
    X = torch.randn(num_other_batches, num_train_batches, device=device, dtype=torch.float32, requires_grad=True)
    # X.data = X.data + torch.log(torch.tensor(p / (1 - p))).item()

    # C = lower_bound + (upper_bound - lower_bound) * torch.sigmoid(X)
    # print("C: ", C)
    # print("X_mean: ", torch.mean(X))
    # print("C_mean: ", torch.mean(C))

    # def loss_fn(C, A, B):
    #     return torch.mean((torch.matmul(C, A) - B) ** 2)

    # optimizer = torch.optim.LBFGS([X], max_iter=20, tolerance_grad=1e-5, tolerance_change=1e-9)
    optimizer = torch.optim.Adam([X], lr=1e-3)

    def closure():
        optimizer.zero_grad()
        # C = lower_bound + (X - X.min()) * (upper_bound - lower_bound) / (X.max() - X.min())
        # C = lower_bound + (upper_bound - lower_bound) * torch.sigmoid(X)
        C = torch.relu(X)
        # C = torch.exp(X)
        # C = X
        loss = F.mse_loss(torch.matmul(C, train_gradients), other_gradients)
        # loss = F.mse_loss(torch.matmul(C, train_gradients), other_gradients)
        # loss = loss_fn(X, train_gradients, other_gradients)
        losses.append(loss.item())
        loss.backward()
        return loss

    for step in tqdm(range(max_iter), desc="Computing coefficients"):
        optimizer.step(closure)
        # if (step+1) % 10 == 0:
        #     with torch.no_grad():
        #         C = lower_bound + (X - X.min()) * (upper_bound - lower_bound) / (X.max() - X.min())
        #         current_loss = F.mse_loss(torch.matmul(C, train_gradients), other_gradients)
        #         print(f"Step {step+1}: Loss = {current_loss}")
    C = torch.relu(X)
    plot_loss(losses, save_dir)
    plot_coefficient(C.detach().cpu().numpy(), C.min().item(), C.max().item(), save_dir)
    B_estimated = torch.matmul(C, train_gradients)
    tolerance = 1e-6
    print("C_min: ", C.min())
    print("C_max: ", C.max())
    print("loss: ", F.mse_loss(B_estimated, other_gradients))
    is_valid = torch.allclose(B_estimated, other_gradients, atol=tolerance)
    print(is_valid)
    return C.T.detach()


if __name__ == "__main__":
    seed = 0
    pruning_rate = 0.1
    data_dir = "./data"
    dataset = "cifar-10"
    save_dir = f"./checkpoints/{dataset}/pruning_rate={pruning_rate}/seed={seed}"
    coefficients_path = f"./checkpoints/{dataset}/pruning_rate={pruning_rate}/seed={seed}/coefficients.npy"
    coefficients = np.load(coefficients_path)
    plot_coefficient(coefficients, 0, 2, save_dir)