import torch
import torch.nn as nn
import numpy as np
import random
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def save_model(path, model):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({'model': model.state_dict()}, os.path.join(path, "model.pth"))


def load_model(path, model):
    state_dict = torch.load(os.path.join(path, "model.pth"))
    model.load_state_dict(state_dict['model'])
    return model
