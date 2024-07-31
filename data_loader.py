from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from prune import prune_dataset

def load_data(data_dir, pruning_rate=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 加载原始数据集
    original_train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)
    # 应用随机剪枝
    pruned_train_dataset, other_train_dataset = prune_dataset(original_train_dataset, pruning_rate=pruning_rate)
    train_loader = DataLoader(pruned_train_dataset, batch_size=64, shuffle=True, num_workers=0)
    other_loader = DataLoader(other_train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    # print(next(iter(test_loader))[0].shape)
    print((64, 3, 32, 32))
    print("Original train dataset shape: ", (len(original_train_dataset), original_train_dataset[0][0].shape))
    print("Pruned train dataset shape: ", (len(pruned_train_dataset), pruned_train_dataset[0][0].shape))
    if len(other_train_dataset) != 0:
        print("Other train dataset shape: ", (len(other_train_dataset), other_train_dataset[0][0].shape))
    return train_loader, test_loader, other_loader

