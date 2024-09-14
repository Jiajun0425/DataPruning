from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from utils import *


def compute_cosine_similarity(gradients, device, threshold, save_dir):
    # gradients = torch.tensor(gradients, device=device, dtype=torch.float32)
    # similarity_matrix = F.cosine_similarity(gradients.unsqueeze(1), gradients.unsqueeze(0), dim=2)
    similarity_matrix = cosine_similarity(gradients)
    # plot_cosine_similarity(similarity_matrix, save_dir)

    # total_size = similarity_matrix.size
    # thresholds = np.arange(0, 1, 0.1)
    # frequencies = []

    # for threshold in tqdm(thresholds, desc="Computing frequency"):
    #     frequency = np.sum(similarity_matrix >= threshold) / total_size
    #     frequencies.append(frequency)
    # print(frequencies)
    plot_frequency(similarity_matrix, save_dir)

    groups = []
    # visited = set()

    # for i in tqdm(range(similarity_matrix.shape[0]), desc="Computing cosine similarity"):
    #     if i not in visited:
    #         group = [i]
    #         visited.add(i)
            
    #         for j in range(i + 1, similarity_matrix.shape[1]):
    #             if similarity_matrix[i, j] >= threshold and j not in visited:
    #                 group.append(j)
    #                 visited.add(j)
    #         if len(group) > max_gropu_length:
    #             max_gropu_length = len(group)
            
    #         groups.append(group)
    
    return groups


if __name__ == "__main__":
    seed = 0
    pruning_rate = 0.1
    threshold = 0.99
    data_dir = "./data"
    dataset = "cifar-10"
    save_dir = f"./checkpoints/{dataset}/pruning_rate={pruning_rate}/seed={seed}"
    gradients_path = f"./checkpoints/{dataset}/pruning_rate={pruning_rate}/seed={seed}/gradients.npz"
    set_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("pruning_rate: ", pruning_rate)
    
    with np.load(gradients_path) as gradients:
        train_gradients = gradients['train_gradients']
        other_gradients = gradients['other_gradients']

    gradients = np.concatenate((train_gradients, other_gradients), axis=0)

    groups = compute_cosine_similarity(gradients, device, threshold, save_dir)

    # print("相似的向量索引分组：", groups)
    