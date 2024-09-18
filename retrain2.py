from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from data_loader import load_data, load_coreset
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

def prune_train(epoch, pruning_rate, selection=None):
    model.train()
    train_loss = 0
    correct = 0
    num_coreset = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        batch = data.shape[0]
        num_corebatch = int(batch * (1-pruning_rate))
        num_coreset += num_corebatch
        optimizer.zero_grad()
        _, output = model(data)
        sample_loss = criterion(output, target)
        
        if selection == "max":
            max_loss, indices = torch.topk(sample_loss, num_corebatch)
        elif selection == "min":
            min_loss, indices = torch.topk(sample_loss, num_corebatch, largest=False)
        elif selection == "mean":
            mean_loss = torch.mean(sample_loss)
            close_loss, indices = torch.topk(torch.abs(sample_loss - mean_loss), num_corebatch, largest=False)
        else :
            sorted_loss, raw_indices = torch.sort(sample_loss)
            mean_loss = torch.mean(sample_loss)
            kernel = torch.ones(num_corebatch).view(1, 1, num_corebatch).to(device)
            subset_sum = F.conv1d(sorted_loss.view(1, 1, batch), kernel, stride=1)
            subset_mean = subset_sum.view(-1) / num_corebatch
            corebatch_index = torch.argmin(torch.abs(subset_mean - mean_loss)).item()
            # loss = torch.mean(sorted_loss[corebatch_index: corebatch_index + num_corebatch])
            indices = raw_indices[corebatch_index: corebatch_index + num_corebatch]

        loss = torch.mean(sample_loss[indices])

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()
        pred = output[indices].argmax(dim=1, keepdim=True)
        correct += pred.eq(target[indices].view_as(pred)).sum().item()
        # 更新进度条的显示信息
        progress_bar.set_postfix(loss=loss.item())
    accuracy = 100. * correct / num_coreset
    print(f'Epoch: {epoch+1}, Train set: Total loss: {train_loss:.4f}, Accuracy: {correct}/{num_coreset} ({accuracy:.2f}%)')
    return accuracy, train_loss

def infobatch_train(epoch, class_features):
    model.train()
    train_loss = 0
    correct = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for i, (data, target, indices, rescale_weight) in enumerate(progress_bar):
        data, target, rescale_weight = data.to(device), target.to(device), rescale_weight.to(device)
        optimizer.zero_grad()
        feature, output = model(data)

        if epoch == 0 and i == 0:
            class_features = class_features.expand(-1, feature.shape[1])
        
        unique_class = torch.unique(target)
        scores = torch.zeros(data.shape[0]).to(device)
        for class_label in unique_class:
            class_indices = (target == class_label).nonzero(as_tuple=True)[0]
            scores[class_indices] = torch.norm(feature[class_indices].detach() - class_features[class_label], p=2, dim=1)
            class_feature = class_features.clone()
            class_feature[class_label] = class_feature[class_label] * i/(i+1) + torch.sum(feature[class_indices], dim=0).detach() / (i+1)
            class_features = class_feature

        loss = criterion(output, target)
        coreset.__setscore__(indices.detach().cpu().numpy(),scores.cpu().numpy())
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
    return accuracy, train_loss, class_features


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
    score_path = os.path.join(save_dir, "scores.npy")
    print(score_path)
    scores = np.load(score_path)
    i_stage = 0
    score = scores[i_stage]
    next_stage_epoch = num_epoch_each_stage[i_stage]
    i_stage += 1

    if args.pruning_method == "infobatch":
        train_loader, test_loader, coreset, nclass = load_coreset(args.data_dir, args.dataset, args.shuffle, args.batch_size, args.test_batch_size, args.pruning_method, args.ratio, score, args.num_epoch, args.delta)
    else:
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
    
    if args.pruning_method in ["infobatch", "prune"]:
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

    
    class_features = torch.zeros((nclass, 1)).to(device)
    selection = None
    print("selection:", selection)
    for epoch in range(args.num_epoch):
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr,
                                                            steps_per_epoch=len(train_loader),
                                                            epochs=args.num_epoch, div_factor=args.div_factor,
                                                            final_div_factor=args.final_div, pct_start=args.pct_start,
                                                            last_epoch=epoch * len(train_loader) - 1)
        if args.pruning_method == "infobatch":
            _, _, class_features = infobatch_train(epoch, class_features)
        elif args.pruning_method == "prune":
            if epoch < args.delta * args.num_epoch:
                prune_train(epoch, args.ratio, selection)
            else :
                prune_train(epoch, 0, selection)
        else:
            train(epoch)
    test()
    print("pruning-method:", args.pruning_method)
    print("ratio:", args.ratio)
    print("delta:", args.delta)
    print("shuffle:", args.shuffle)
    print("selection:", selection)
