import torch
from torch.utils.data import Subset

def prune_dataset(dataset, pruning_rate=0.1):

    total_size = len(dataset)

    num_to_prune = int(total_size * (1-pruning_rate))

    indices = torch.randperm(total_size)[:num_to_prune]
    other_indices = torch.randperm(total_size)[num_to_prune:]

    pruned_dataset = Subset(dataset, indices)
    other_dataset = Subset(dataset, other_indices)
    return pruned_dataset, other_dataset

