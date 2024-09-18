from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from prune import prune_dataset
from random_prune import random_prune_dataset
from infobatch import InfoBatch


def load_data(data_dir, dataset, shuffle, batch_size=64, test_batch_size=64):
    stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    if dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=test_transform)
    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=False, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=False, transform=test_transform)
    
    if batch_size == 0:
        batch_size = len(train_dataset)
        test_batch_size = len(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

def load_coreset(data_dir, dataset, shuffle, batch_size=64, test_batch_size=64, pruning_method="prune", pruning_rate=0.1, scores=None, num_epoch=300, delta=0.875):
    stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    if dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=test_transform)
        nclass = 10
    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=False, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=False, transform=test_transform)
        nclass = 100
    
    if batch_size == 0:
        batch_size = len(train_dataset)
        test_batch_size = len(test_dataset)

    print("pruning method:", pruning_method)
    if pruning_method == "prune":
        coreset, otherset = prune_dataset(train_dataset, scores, nclass, pruning_rate)
        sampler = None
    elif pruning_method == "random":
        coreset, otherset = random_prune_dataset(train_dataset, pruning_rate)
        sampler = None
    elif pruning_method == "infobatch":
        coreset = InfoBatch(train_dataset, pruning_rate, num_epoch, delta, nclass)
        sampler = coreset.pruning_sampler()
    train_loader = DataLoader(coreset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, coreset, nclass

def load_dataset(data_dir, batch_size=64, pruning_rate=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 加载原始数据集
    original_train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)
    # 应用随机剪枝
    pruned_train_dataset, other_train_dataset = prune_dataset(original_train_dataset, pruning_rate=pruning_rate)

    batch_size1 = len(pruned_train_dataset)
    batch_size2 = len(other_train_dataset)
    train_loader = DataLoader(pruned_train_dataset, batch_size=batch_size1, shuffle=False, num_workers=0)
    other_loader = DataLoader(other_train_dataset, batch_size=batch_size2, shuffle=False, num_workers=0)
    for images, labels in train_loader:
        train_data, train_targets = images, labels
    for images, labels in other_loader:
        other_data, other_targets = images, labels

    print("Original train dataset shape: ", (len(original_train_dataset), original_train_dataset[0][0].shape))
    print("Pruned train dataset shape: ", (len(pruned_train_dataset), pruned_train_dataset[0][0].shape))
    if len(other_train_dataset) != 0:
        print("Other train dataset shape: ", (len(other_train_dataset), other_train_dataset[0][0].shape))
    
    return train_data, train_targets, other_data, other_targets
