from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from data_loader import load_data
from model import *
from utils import *
from argument import args


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        _, output = model(data)
        loss = criterion(output, target)
        # print(loss)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        # 更新进度条的显示信息
        progress_bar.set_postfix(loss=loss.item())
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Epoch: {epoch+1}, Train set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy, train_loss


def test():
    model.eval()
    test_loss = 0
    correct = 0
    # 使用tqdm显示测试进度
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            _, output = model(data)
            test_loss += test_criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # 更新进度条的显示信息
            progress_bar.set_postfix(loss=test_loss / len(test_loader.dataset), accuracy=100. * correct / len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


if __name__ == "__main__":
    save_dir = f"{args.ckpt_dir}/{args.dataset}/{args.model}"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = load_data(args.data_dir, args.dataset, args.shuffle, args.batch_size, args.test_batch_size)

    if args.dataset == "cifar10":
        nclass = 10
    elif args.dataset == "cifar100":
        nclass = 100

    if args.model.lower()=='r18':
        model = ResNet18(nclass)
    elif args.model.lower()=='r50':
        model = ResNet50(num_classes=nclass)
    elif args.model.lower()=='r101':
        model = ResNet101(num_classes=nclass)
    else:
        model = ResNet50(num_classes=nclass)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    test_criterion = nn.CrossEntropyLoss(reduction='sum')

    if args.optimizer.lower()=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    elif args.optimizer.lower()=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'lars':
        from lars import Lars
        optimizer = Lars(model.parameters(), lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'lamb':
        from lamb import Lamb
        optimizer  = Lamb(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,args.max_lr,steps_per_epoch=len(train_loader),
                                                  epochs=args.num_epoch,div_factor=args.div_factor,
                                                  final_div_factor=args.final_div,pct_start=args.pct_start)

    accuracies = np.zeros(args.num_epoch)
    test_accuracies = np.zeros(args.num_epoch)
    losses = []
    for epoch in range(args.num_epoch):
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr,
                                                            steps_per_epoch=len(train_loader),
                                                            epochs=args.num_epoch, div_factor=args.div_factor,
                                                            final_div_factor=args.final_div, pct_start=args.pct_start,
                                                            last_epoch=epoch * len(train_loader) - 1)
        accuracy, loss = train(epoch)
        test_accuracy = test()
        losses.append(loss)

        model_dir = os.path.join(save_dir, f"models/epoch={epoch}")
        accuracies[epoch] = accuracy
        test_accuracies[epoch] = test_accuracy
        save_model(model_dir, model)
    plot_loss(losses, save_dir)
    
    np.save(os.path.join(save_dir, "accuracy.npy"), accuracies)
    np.save(os.path.join(save_dir, "test_accuracy.npy"), test_accuracies)
    print("Whole dataset for training")
    print("shuffle", args.shuffle)
    print("optimizer:", args.optimizer)
