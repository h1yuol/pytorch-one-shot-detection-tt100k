import random
import os

import numpy as np

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from config import cfg

class MyBatchSampler(Sampler):
    """
    MyBatchSampler is intended for triplet generator.

    Input:
        - samples: a list of samples, each sample is (<path/to/img>, <classIdx>), i.e. Dataset.samples
        - num_class: # of different classes
        - num_sample_per_class: # of samples for each class
    """
    def __init__(self, samples, num_class, num_sample_per_class):
        self.samples = samples
        self.P = num_class
        self.K = num_sample_per_class
        self.classIdx2sampleIdx = {}
        for sampleIdx,sample in enumerate(samples):
            classIdx = sample[1]
            self.classIdx2sampleIdx[classIdx] = self.classIdx2sampleIdx.get(classIdx, []) + [sampleIdx]
        self.classIdxList = list(filter(lambda classIdx: len(self.classIdx2sampleIdx[classIdx])>self.K, list(self.classIdx2sampleIdx.keys())))
        self.enlarge = 16
        # self.num_samples = sum([len(self.classIdx2sampleIdx[classIdx]) for classIdx in self.classIdxList])
        # self.num_batch = int(self.num_samples / (num_class * num_sample_per_class))

    def __iter__(self):
        shuffled_classIdxList = []
        for __ in range(self.enlarge):
            shuffled_classIdxList += [self.classIdxList[i] for i in torch.randperm(len(self.classIdxList))]
        for i in range(self.P, len(shuffled_classIdxList), self.P):
            batch = []
            chosen_classeIdxs = shuffled_classIdxList[i-self.P:i]
            for classIdx in chosen_classeIdxs:
                sampleIdxList = self.classIdx2sampleIdx[classIdx]
                batch += [sampleIdxList[i] for i in torch.randperm(len(sampleIdxList))[:self.K]]
            assert len(batch) == self.P * self.K, (len(batch), self.P * self.K, len(chosen_classeIdxs))
            np.random.shuffle(batch)
            yield batch

        # for __ in range(self.num_batch):
        #     batch = []
        #     chosen_classeIdxs = [self.classIdxList[i] for i in torch.randperm(len(self.classIdxList))[:self.P]]
        #     for classIdx in chosen_classeIdxs:
        #         sampleIdxList = self.classIdx2sampleIdx[classIdx]
        #         batch += [sampleIdxList[i] for i in torch.randperm(len(sampleIdxList))[:self.K]]
        #     assert len(batch) == self.P * self.K, (len(batch), self.P * self.K)
        #     np.random.shuffle(batch)
        #     yield batch

    def __len__(self):
        return (self.enlarge * len(self.classIdxList)) * self.K

def get_dataloaders(num_workers, P, K, phaseList):
    data_transforms = {
        'train': transforms.Compose([
                transforms.Resize([cfg.TRAIN.sign_input_size, cfg.TRAIN.sign_input_size]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        'test': transforms.Compose([
                transforms.Resize([cfg.TRAIN.sign_input_size, cfg.TRAIN.sign_input_size]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        'other': transforms.Compose([
                transforms.Resize([cfg.TRAIN.sign_input_size, cfg.TRAIN.sign_input_size]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
    }

    image_datasets = {
        x: ImageFolder(str(cfg.PATH.sign_data_dir/x), data_transforms[x]) for x in phaseList
    }

    samplers = {
        x: MyBatchSampler(image_datasets[x].samples, P, K) for x in phaseList
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
                image_datasets[x], 
                batch_sampler=samplers[x],
                num_workers=num_workers[x],
            ) for x in phaseList
    }

    dataset_sizes = {x: len(samplers[x]) for x in phaseList }  # used for averaging loss term

    return dataloaders, dataset_sizes

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--phase', default='train+test', help='dataset to use')
    parser.add_argument('--workers', default='1,1', help='<trainWorkers>,<testWorkers>')

    args = parser.parse_args()

    workerList = list(map(int, args.workers.split(',')))

    num_workers = {
        'train': workerList[0],
        'test': workerList[1],
    }

    from IPython import embed
    embed()



