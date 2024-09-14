import torch
import numpy as np
import random
from utils import CustomSubset

def prune_dataset(dataset, scores, nclass, pruning_rate=0.1):

    num_dataset = len(dataset)
    num_coreset = int(num_dataset * (1-pruning_rate))
    num_class = [num_coreset // nclass] * nclass
    for i in random.sample(range(nclass), num_coreset % nclass):
        num_class[i] += 1

    all_indices = torch.arange(num_dataset)
    mask = torch.ones(num_dataset, dtype=torch.bool)
    indices = np.zeros(num_coreset)
    # scores = np.load(score_path)
    start_indice = 0
    
    for class_label in range(nclass):
        class_indices = np.where(np.array(dataset.targets) == class_label)[0]
        class_scores = scores[class_indices]
        top_indices_in_class = class_indices[np.argsort(-class_scores)[:num_class[class_label]]]
        indices[start_indice: start_indice+num_class[class_label]] = top_indices_in_class
        start_indice += num_class[class_label]

    indices, _ = torch.sort(torch.tensor(indices, dtype=torch.int))
    mask[indices] = False
    other_indices = all_indices[mask]

    coreset = CustomSubset(dataset, indices)
    otherset = CustomSubset(dataset, other_indices)
    return coreset, otherset

