import torch
from utils import CustomSubset

def random_prune_dataset(dataset, pruning_rate=0.1):

    total_size = len(dataset)

    num_to_prune = int(total_size * (1-pruning_rate))

    result = torch.randperm(total_size)

    indices, _ = torch.sort(result[:num_to_prune])
    other_indices, _ = torch.sort(result[num_to_prune:])

    coreset = CustomSubset(dataset, indices)
    otherset = CustomSubset(dataset, other_indices)
    return coreset, otherset

