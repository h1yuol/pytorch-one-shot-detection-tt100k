import numpy as np
import time
import datetime
from pathlib import Path
from IPython import embed
from tqdm import tqdm

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from lib.model.one_shot.one_shot_resnet import one_shot_resnet

from config import cfg

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')

    parser.add_argument('--load_name', dest='load_name',
                      help='path to the loading model relative to "experiment / net"',
                      type=str)
    parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=3, type=int)  
    parser.add_argument('--net', dest='net',
                      default='res101', type=str)
    parser.add_argument('--margin', dest='margin',
                      help='triplet loss margin',
                      default=0.5, type=float)
    parser.add_argument('--dhigh', dest='dhigh',
                      help='d high',
                      default=10, type=float)
    parser.add_argument('--dlow', dest='dlow',
                      help='d low',
                      default=0, type=float)


    args = parser.parse_args()
    return args

class gallerySampler(Sampler):
    """
    gallerySampler is intended for evaluation of oneshot model.

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
        self.galleryIdxList = [sampleIdxList[0] for sampleIdxList in self.classIdx2sampleIdx.values()]
        self.classIdxList = list(filter(lambda classIdx: len(self.classIdx2sampleIdx[classIdx])>self.K, list(self.classIdx2sampleIdx.keys())))

    def __iter__(self):
        for i in range(self.P, len(self.classIdxList), self.P):
            batch = []
            chosen_classeIdxs = self.classIdxList[i-self.P:i]
            for classIdx in chosen_classeIdxs:
                sampleIdxList = self.classIdx2sampleIdx[classIdx][1:]
                batch += [sampleIdxList[i] for i in torch.randperm(len(sampleIdxList))[:self.K]]
            assert len(batch) == self.P * self.K, (len(batch), self.P * self.K, len(chosen_classeIdxs))
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.classIdxList) * self.K

class constSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        yield self.indices
    def __len__(self):
        return len(self.indices)


def get_test_dataset(num_workers, P, K):
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
    test_ds = ImageFolder(str(cfg.PATH.sign_data_dir/"test"), data_transforms["test"])
    # train_ds = ImageFolder(str(cfg.PATH.sign_data_dir/"train"), data_transforms["train"])
    # selected_classes = list(set(test_ds) & set(train_ds))

    sampler = gallerySampler(test_ds.samples, P, K)

    dataloader = torch.utils.data.DataLoader(
                test_ds, 
                batch_sampler=sampler,
                num_workers=num_workers,
            )
    galleryLoader = torch.utils.data.DataLoader(
                test_ds, 
                batch_sampler=constSampler(sampler.galleryIdxList),
                num_workers=num_workers,
            )

    return dataloader, galleryLoader, sampler, test_ds

def compute_mat_dist(a,b,squared=False):
    """
    Input 2 embedding matrices and output distance matrix
    """
    assert a.size(1) == b.size(1)
    dot_product = a.mm(b.t())
    a_square = torch.pow(a, 2).sum(dim=1, keepdim=True)
    b_square = torch.pow(b, 2).sum(dim=1, keepdim=True)
    dist = a_square - 2*dot_product + b_square.t()
    dist = dist.clamp(min=0)
    if not squared:
        epsilon = 1e-12
        mask = (dist.eq(0))
        dist += epsilon * mask.float()
        dist = dist.sqrt()
        dist *= (1-mask.float())
    return dist

if __name__ == '__main__':
    args = parse_args()

    P, K = 1, cfg.TRAIN.K

    dataloader, galleryLoader, sampler, test_ds = get_test_dataset(args.num_workers, P, K)

    im_data = torch.FloatTensor(1).cuda()
    labels = torch.FloatTensor(1).cuda()
    gallery_labels = torch.FloatTensor(1).cuda()
    im_data = Variable(im_data)
    labels = Variable(labels)
    gallery_labels = Variable(gallery_labels)

    model_path = args.load_name
    norm = cfg.TRAIN.norm
    embedding_dim = cfg.TRAIN.embedding_dim
    margin = args.margin
    model = one_shot_resnet(embedding_dim, model_path, margin, pretrained=False, norm=norm,onlyEmbeddings=True)

    model.eval_create_architecture()

    output_dir = cfg.PATH.experiment_dir / args.net
    state_dict = torch.load(str(output_dir/args.load_name))
    model.load_state_dict(state_dict['model'])
    model.cuda()

    model.eval()
    # get gallery
    data = next(iter(galleryLoader))
    im_data.data.resize_(data[0].size()).copy_(data[0])
    labels.data.resize_(data[1].size()).copy_(data[1])
    gallery_labels.data.resize_(data[1].size()).copy_(data[1])
    gallery_embeddings = model(im_data, labels)

    # embed()

    accus = []
    for step,data in tqdm(enumerate(dataloader)):
        im_data.data.resize_(data[0].size()).copy_(data[0])
        labels.data.resize_(data[1].size()).copy_(data[1])
        # batch_size = labels.size(0)

        embeddings = model(im_data, labels)
        dists = compute_mat_dist(embeddings, gallery_embeddings)
        predicts_indices = dists.argmin(dim=1)
        predicts = gallery_labels[predicts_indices]

        accuracy = (predicts.eq(labels)).float().mean().item()
        accus.append(accuracy)
    print("Top 1 accuracy: {:.4g}".format(np.mean(accus)))
    embed()






    #     for i in range(batch_size-1):
    #         for j in range(i+1,batch_size):
    #             y_true.append(labels[i].eq(labels[j]).int().item())
    #             y_scores.append(dists[i,j].item())

    # y_true = np.array(y_true)
    # y_scores = np.array(y_scores)

    # true_label_indices = np.arange(len(y_true))[y_true==1]
    # false_label_indices = np.arange(len(y_true))[y_true==0]

    # selected_false_indices = np.random.choice(false_label_indices, size=len(true_label_indices),replace=False)

    # indices = np.concatenate((true_label_indices,selected_false_indices))

    # mAP = average_precision_score(y_true[indices], y_scores[indices])
    # print("mAP:{}".format(mAP))
    # print('-'*15)

    # embed()

















