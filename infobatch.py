import math
import numpy as np
from torch.utils.data import Dataset
import random

class InfoBatch(Dataset):
    def __init__(self, dataset, ratio = 0.5, num_epoch=None, delta = 0.875, nclass = 10):
        self.dataset = dataset
        self.ratio = (1 - ratio)
        self.num_epoch = num_epoch
        self.delta = delta
        self.scores = np.ones([len(self.dataset)])
        self.transform = dataset.transform
        self.weights = np.ones(len(self.dataset))
        self.save_num = 0
        self.nclass = nclass
        self.num_coreset = int(len(self.dataset) * self.ratio)
        self.num_class = [self.num_coreset // self.nclass] * self.nclass
        for i in random.sample(range(self.nclass), self.num_coreset % self.nclass):
            self.num_class[i] += 1

    def __setscore__(self, indices, values):
        self.scores[indices] = values

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        weight = self.weights[index]
        return data, target, index, weight

    # def prune(self):
    #     # prune samples that are well learned, rebalence the weight by scaling up remaining
    #     # well learned samples' learning rate to keep estimation about the same
    #     # for the next version, also consider new class balance

    #     b = self.scores<self.scores.mean()
    #     well_learned_samples = np.where(b)[0]
    #     pruned_samples = []
    #     pruned_samples.extend(np.where(np.invert(b))[0])
    #     selected = np.random.choice(well_learned_samples, int(self.ratio*len(well_learned_samples)),replace=False)
    #     self.reset_weights()
    #     if len(selected)>0:
    #         self.weights[selected]=1/self.ratio
    #         pruned_samples.extend(selected)
    #     print('Cut {} samples for next iteration'.format(len(self.dataset)-len(pruned_samples)))
    #     self.save_num += len(self.dataset)-len(pruned_samples)
    #     np.random.shuffle(pruned_samples)
    #     return pruned_samples

    def prune(self):
        # prune samples that are well learned, rebalence the weight by scaling up remaining
        # well learned samples' learning rate to keep estimation about the same
        # for the next version, also consider new class balance

        pruned_samples = []
        for class_label in range(self.nclass):
            class_indices = np.where(np.array(self.dataset.targets) == class_label)[0]
            class_scores = self.scores[class_indices]
            top_indices_in_class = class_indices[np.argsort(-class_scores)[:self.num_class[class_label]]]
            if np.all(self.scores == np.ones([len(self.dataset)])):
                pruned_samples.extend(class_indices)
            else:
                pruned_samples.extend(top_indices_in_class)

        print('Cut {} samples for next iteration'.format(len(self.dataset)-len(pruned_samples)))
        self.save_num += len(self.dataset)-len(pruned_samples)
        np.random.shuffle(pruned_samples)
        return pruned_samples

    def pruning_sampler(self):
        return InfoBatchSampler(self, self.num_epoch, self.delta)

    def no_prune(self):
        samples = list(range(len(self.dataset)))
        np.random.shuffle(samples)
        return samples

    def mean_score(self):
        return self.scores.mean()

    def normal_sampler_no_prune(self):
        return InfoBatchSampler(self.no_prune)

    def get_weights(self,indexes):
        return self.weights[indexes]

    def total_save(self):
        return self.save_num

    def reset_weights(self):
        self.weights = np.ones(len(self.dataset))



class InfoBatchSampler():
    def __init__(self, infobatch_dataset, num_epoch = math.inf, delta = 1):
        self.infobatch_dataset = infobatch_dataset
        self.seq = None
        self.stop_prune = num_epoch * delta
        self.seed = 0
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.seed+=1
        if self.seed>self.stop_prune:
            if self.seed <= self.stop_prune+1:
                self.infobatch_dataset.reset_weights()
            self.seq = self.infobatch_dataset.no_prune()
        else:
            self.seq = self.infobatch_dataset.prune()
        self.ite = iter(self.seq)
        self.new_length = len(self.seq)

    def __next__(self):
        try:
            nxt = next(self.ite)
            return nxt
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self

