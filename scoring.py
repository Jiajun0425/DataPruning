import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad
from tqdm import tqdm
import random
from argument import args
from data_loader import load_data
from model import *
from utils import *


def compute_mean_gradients(data_loader, model):
    model.eval()
    all_params = [ p for p in model.parameters() if p.requires_grad ]
    mean_gradients = 0

    for i, (data, target) in enumerate(tqdm(data_loader, desc="Computing mean gradients")):
        data, target = data.to(device), target.to(device)

        _, outputs = model(data)
        loss = criterion(outputs, target)

        batch_gradients = list(grad(loss, all_params, create_graph=False))
        batch_gradients = torch.nn.utils.parameters_to_vector(batch_gradients).detach()

        mean_gradients = mean_gradients * i/(i+1) + batch_gradients / (i+1)

    return mean_gradients

def scoring(data_loader, mean_gradients, model):
    model.eval()
    all_params = [ p for p in model.parameters() if p.requires_grad ]
    score = np.zeros(len(data_loader))

    for i, (data, target) in enumerate(tqdm(data_loader, desc="Scoring")):
        data, target = data.to(device), target.to(device)

        _, outputs = model(data)
        loss = criterion(outputs, target)

        sample_gradients = list(grad(loss, all_params, create_graph=False))
        sample_gradients = torch.nn.utils.parameters_to_vector(sample_gradients).detach()
        
        score[i] = torch.norm(sample_gradients, p=2) * F.cosine_similarity(mean_gradients.unsqueeze(0), sample_gradients.unsqueeze(0))

    return score

def batch_scoring(data_loader, model, num_sample):
    model.eval()
    all_params = [ p for p in model.parameters() if p.requires_grad ]
    score = np.zeros(num_sample)
    start_indice = 0

    for i, (data, target) in enumerate(tqdm(data_loader, desc="Scoring")):
        batch = data.shape[0]
        data, target = data.to(device), target.to(device)

        _, outputs = model(data)
        loss = batch_criterion(outputs, target)
        mean_loss = torch.mean(loss)

        mean_gradients = list(grad(mean_loss, all_params, create_graph=False, retain_graph=True))
        mean_gradients = torch.nn.utils.parameters_to_vector(mean_gradients).detach()
        for l in loss:
            sample_gradients = list(grad(l, all_params, create_graph=False, retain_graph=True))
            sample_gradients = torch.nn.utils.parameters_to_vector(sample_gradients).detach()
            score[start_indice] = torch.norm(sample_gradients, p=2) * F.cosine_similarity(mean_gradients.unsqueeze(0), sample_gradients.unsqueeze(0))
            start_indice += 1

    return score


if __name__ == "__main__":
    save_dir = f"{args.ckpt_dir}/{args.dataset}/{args.model}"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    accuracy = np.load(os.path.join(save_dir, "test_accuracy.npy"))
    num_epoch_each_stage = [args.num_epoch // args.num_stage] * args.num_stage
    for i in range(args.num_epoch % args.num_stage):
        num_epoch_each_stage[i] += 1

    whole_loader, _ = load_data(args.data_dir, args.dataset, args.shuffle, args.batch_size, args.test_batch_size)
    sample_loader, _ = load_data(args.data_dir, args.dataset, args.shuffle, 1, args.test_batch_size)

    if args.dataset == "cifar10":
        nclass = 10
    elif args.dataset == "cifar100":
        nclass = 100

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    batch_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='none')

    start_indice = 0
    scores = np.zeros((args.num_stage, args.topK, len(sample_loader)))
    for idx in range(args.num_stage):
        acc = accuracy[start_indice: start_indice + num_epoch_each_stage[idx]]
        acc_diff = np.diff(acc)
        sorted_indices = np.argsort(acc_diff)[::-1]
        top_k_epoch = sorted_indices[:args.topK] + start_indice
        print(top_k_epoch)
        start_indice += num_epoch_each_stage[idx]
        
        for i, k in enumerate(top_k_epoch):
            model_dir = os.path.join(save_dir, f"models/epoch={k}")
            if args.model.lower()=='r18':
                model = ResNet18(nclass)
            elif args.model.lower()=='r50':
                model = ResNet50(num_classes=nclass)
            elif args.model.lower()=='r101':
                model = ResNet101(num_classes=nclass)
            else:
                model = ResNet50(num_classes=nclass)
            model = load_model(model_dir, model)
            model = model.to(device)

            # mean_gradients = compute_mean_gradients(whole_loader, model)
            # mean_gradients = compute_mean_gradients(sample_loader, model)
            # score = scoring(sample_loader, mean_gradients, model)

            score = batch_scoring(whole_loader, model, len(sample_loader))
            scores[idx, i] = score
            
            # np.save(os.path.join(save_dir, f"test_sample_score{i}.npy"), score)
    mean_score = np.mean(scores, axis=1)
    print(mean_score.shape)
    np.save(os.path.join(save_dir, f"test_batch_scores.npy"), mean_score)