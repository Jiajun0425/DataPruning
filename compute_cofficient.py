from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.optimize import least_squares, lsq_linear
import time
from torch.autograd import grad
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
from torch.utils.data import TensorDataset, DataLoader
from data_loader import load_data, load_dataset
from model import ConvNet
from utils import *


def compute_gradients(data_loader, model, device):
    model.eval()
    all_gradients = []
    all_params = [ p for p in model.parameters() if p.requires_grad ]
    params = all_params[-2:]

    for data, target in tqdm(data_loader, desc="Processing batches"):
        data, target = data.to(device), target.to(device)

        outputs = model(data)
        loss = F.cross_entropy(outputs, target)

        sample_gradients = list(grad(loss, params, create_graph=False))
        sample_gradients = torch.nn.utils.parameters_to_vector(sample_gradients).detach().cpu().numpy()

        # model.zero_grad()
        # loss.backward()

        # sample_gradients = []
        # for param in model.parameters():
        #     if param.grad is not None:
        #         sample_gradients.extend(param.grad.view(-1).detach().cpu().numpy())
        all_gradients.append(sample_gradients)

    return np.array(all_gradients)

def find_coefficients(train_gradients, other_gradients):
    """ 使用线性最小二乘法找到最佳系数。"""
    coefficients, _, _, _ = np.linalg.lstsq(train_gradients.T, other_gradients.T, rcond=None)

    return coefficients

# def objective_function(coeffs, train_matrix, target_vector):
#     """计算预测梯度与目标梯度之间的误差。"""
#     predicted = train_matrix.T @ coeffs  # 计算预测的梯度
#     return predicted - target_vector

# def solve_coefficients(train_matrix, other_matrix, lower_bound, upper_bound):

#     num_train_batches = train_matrix.shape[0]  # 训练集批次数
#     coefficients = []

#     # for target_vector in tqdm(other_matrix, desc="Computing cofficients"):
#     for i in range(1):
#         target_vector = other_gradients[i]
#         x0 = np.zeros(num_train_batches)
#         bounds = (lower_bound, upper_bound)

#         result = least_squares(objective_function, x0, args=(train_matrix, target_vector), 
#                                bounds=bounds, method='trf')
#         coefficients.append(result.x)
#     return np.array(coefficients).T

# def objective_function(coeffs, train_gradients, other_gradients):
#     """计算所有预测与目标之间的总残差。"""
#     predicted = train_gradients @ coeffs.reshape(train_gradients.shape[0], other_gradients.shape[1])
#     residuals = predicted - other_gradients
#     return residuals.flatten()

# def solve_coefficients(train_gradients, other_gradients, lower_bound, upper_bound):
#     num_train_batches = train_gradients.shape[0]
#     num_other_batches = other_gradients.shape[0]
#     num_coeffs = num_train_batches * num_other_batches
#     AAT = train_gradients @ train_gradients.T
#     ABT = train_gradients @ other_gradients.T
#     x0 = np.zeros(num_coeffs)
#     bounds = (lower_bound, upper_bound)
#     start_time = time.time()
#     result = least_squares(objective_function, x0, args=(AAT, ABT), bounds=bounds, method='dogbox')
#     end_time = time.time()
#     print("Run time: ", end_time - start_time)
#     cofficients = result.x.reshape(num_train_batches, num_other_batches)

#     pred_unclamped = train_gradients.T @ coefficients
#     error_unclamped = ((pred_unclamped - other_gradients.T) ** 2).sum()
#     print(f"Error with unclamped coefficients: {error_unclamped}")

#     return coefficients

def solve_coefficients(train_gradients, other_gradients, lower_bound, upper_bound, device, save_dir):
    # 确保数据在 GPU 上
    # train_gradients = torch.tensor(train_gradients, device=device, dtype=torch.float32)
    # other_gradients = torch.tensor(other_gradients, device=device, dtype=torch.float32)

    # num_train_batches = train_gradients.shape[0]
    # num_other_batches = other_gradients.shape[0]
    # coefficients = torch.zeros((num_train_batches, num_other_batches), device=device)

    start_time = time.time()
    # 使用 PyTorch 进行批处理运算
    # 解 Ax = b，使用正规方程法 A * A.T * x = A * b.T
    # AAT = train_gradients @ train_gradients.T
    # AbT = train_gradients @ other_gradients.T
    # coefficients = torch.linalg.solve(AAT, AbT)
    # coefficients, residuals, rank, singular_values = torch.linalg.lstsq(train_gradients.T, other_gradients.T)
    print("upper bound: ", upper_bound)
    coefficients = trf(train_gradients, other_gradients, lower_bound, upper_bound, device, save_dir)
    end_time = time.time()
    print("Run time: ", end_time - start_time)

    # clamped_coefficients = torch.clamp(coefficients, min=lower_bound, max=upper_bound)
    # error_unclamped = ((train_gradients.T @ coefficients - other_gradients.T) ** 2).sum()
    # error_clamped = ((train_gradients.T @ clamped_coefficients - other_gradients.T) ** 2).sum()

    # print(f"Error with unclamped coefficients: {error_unclamped}")
    # print(f"Error with clamped coefficients: {error_clamped}")

    # 应用边界
    # coefficients = torch.clamp(coefficients)

    return coefficients.cpu().numpy()  # 如果需要在 CPU 上使用结果

def process_task(queue, idx, chunked_data, chunked_target, model_dir, device):
    # 注意确保模型在这里移到适当的设备
    torch.cuda.set_device(device)
    surrogate_model = ConvNet().to(device)
    surrogate_model = load_model(model_dir, surrogate_model)
    dataset = TensorDataset(chunked_data, chunked_target)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    gradients = compute_gradients(loader, surrogate_model, device)
    queue.put(gradients)

if __name__ == "__main__":
    seed = 0
    lower_bound = 0  # 系数的上限
    upper_bound = 1  # 系数的上限
    pruning_rate = 0.5
    data_dir = "./data"
    dataset = "cifar-10"
    save_dir = f"./checkpoints/{dataset}/pruning_rate={pruning_rate}/seed={seed}"
    gradients_path = f"./checkpoints/{dataset}/pruning_rate={pruning_rate}/seed={seed}/gradients.npz"
    coefficients_path = f"./checkpoints/{dataset}/pruning_rate={pruning_rate}/seed={seed}/coefficients.npy"
    set_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(num_gpus)
    print("pruning_rate: ", pruning_rate)

    # train_loader, test_loader, other_loader = load_data(data_dir, 1, pruning_rate)
    # train_data, train_targets, other_data, other_targets = load_dataset(data_dir, 1, pruning_rate)
    # chunked_train_data = torch.chunk(train_data, num_gpus)
    # chunked_train_target = torch.chunk(train_targets, num_gpus)
    # chunked_other_data = torch.chunk(other_data, num_gpus)
    # chunked_other_target = torch.chunk(other_targets, num_gpus)

    if not os.path.exists(gradients_path):
        surrogate_model = ConvNet().to(device)
        surrogate_model = load_model(save_dir, surrogate_model)

        # mp.set_start_method('spawn')
        # processes = []
        # queue = Queue()
        # for i in range(num_gpus):
        #     device = torch.device(f"cuda:{i}" if torch.cuda.is_available() else "cpu")
        #     p = Process(target=process_task, args=(queue, i, chunked_train_data[i], chunked_train_target[i], model_dir, device))
        #     p.start()
        #     processes.append(p)
        
        # train_gradients = []
        # for _ in range(num_gpus):
        #     train_gradients.append(queue.get())

        # for p in processes:
        #     p.join()

        train_gradients = compute_gradients(train_loader, surrogate_model, device)
        other_gradients = compute_gradients(other_loader, surrogate_model, device)

        print(train_gradients.shape)
        print(other_gradients.shape)

        gradients = {"train_gradients": train_gradients, "other_gradients": other_gradients}
        
        np.savez(gradients_path, **gradients)
    else: 
        with np.load(gradients_path) as gradients:
            train_gradients = gradients['train_gradients']
            other_gradients = gradients['other_gradients']
            other_gradients = np.sum(other_gradients, axis=0, keepdims=True)

    # coefficients = find_coefficients(train_gradients, other_gradients)
    coefficients = solve_coefficients(train_gradients, other_gradients, lower_bound, upper_bound, device, save_dir)
    print(coefficients)
    coefficients = np.sum(coefficients, axis=1) + 1

    np.save(coefficients_path, coefficients)