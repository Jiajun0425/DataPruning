from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from data_loader import load_coreset
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
    print(f'Epoch: {epoch+1}, Train set: Total loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy, train_loss

def infobatch_train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for data, target, indices, rescale_weight in progress_bar:
        data, target, rescale_weight = data.to(device), target.to(device), rescale_weight.to(device)
        optimizer.zero_grad()
        _, output = model(data)
        loss = criterion(output, target)
        coreset.__setscore__(indices.detach().cpu().numpy(),loss.detach().cpu().numpy())
        loss = loss*rescale_weight
        loss = torch.mean(loss)
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
    print(f'Epoch: {epoch+1}, Train set: Total loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)')
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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    num_epoch_each_stage = [args.num_epoch // args.num_stage] * args.num_stage
    for i in range(args.num_epoch % args.num_stage):
        num_epoch_each_stage[i] += 1
    score_path = os.path.join(save_dir, "test_batch_scores.npy")
    print(score_path)
    scores = np.load(score_path)
    i_stage = 0
    score = scores[i_stage]
    next_stage_epoch = num_epoch_each_stage[i_stage]
    i_stage += 1

    train_loader, test_loader, coreset, nclass = load_coreset(args.data_dir, args.dataset, args.shuffle, args.batch_size, args.test_batch_size, args.pruning_method, args.ratio, score, args.num_epoch, args.delta)

    if args.model.lower()=='r18':
        model = ResNet18(nclass)
    elif args.model.lower()=='r50':
        model = ResNet50(num_classes=nclass)
    elif args.model.lower()=='r101':
        model = ResNet101(num_classes=nclass)
    else:
        model = ResNet50(num_classes=nclass)
    model = model.to(device)
    
    if args.pruning_method in ["infobatch"]:
        reduction = "none"
    else :
        reduction = "mean"
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction=reduction)
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

    for epoch in range(args.num_epoch):
        if args.pruning_method == "prune" and epoch == next_stage_epoch:
            score = scores[i_stage]
            next_stage_epoch += num_epoch_each_stage[i_stage]
            i_stage += 1
            train_loader, test_loader, coreset, nclass = load_coreset(args.data_dir, args.dataset, args.shuffle, args.batch_size, args.test_batch_size, args.pruning_method, args.ratio, score, args.num_epoch, args.delta)

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr,
                                                            steps_per_epoch=len(train_loader),
                                                            epochs=args.num_epoch, div_factor=args.div_factor,
                                                            final_div_factor=args.final_div, pct_start=args.pct_start,
                                                            last_epoch=epoch * len(train_loader) - 1)
        if args.pruning_method == "infobatch":
            infobatch_train(epoch)
        else:
            train(epoch)
    test()
    print("pruning-method:", args.pruning_method)
    print("ratio:", args.ratio)
    print("delta:", args.delta)
    print("shuffle:", args.shuffle)
