import numpy as np
import time
import datetime
from pathlib import Path
import pandas as pd
from IPython import embed
from sklearn.metrics import average_precision_score


import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from lib.model.one_shot.one_shot_resnet import one_shot_resnet
from lib.datasets.data_loader import get_dataloaders

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


if __name__ == '__main__':
    args = parse_args()

    phaseList = ['test']
    num_workers = {}
    num_workers['test'] = args.num_workers

    P, K = cfg.TRAIN.P, cfg.TRAIN.K

    dataloaders, dataset_sizes = get_dataloaders(num_workers, P, K, phaseList)

    im_data = torch.FloatTensor(1).cuda()
    labels = torch.FloatTensor(1).cuda()
    im_data = Variable(im_data)
    labels = Variable(labels)

    model_path = args.load_name
    norm = cfg.TRAIN.norm
    embedding_dim = cfg.TRAIN.embedding_dim
    margin = args.margin
    model = one_shot_resnet(embedding_dim, model_path, margin, pretrained=False, norm=norm)

    model.eval_create_architecture()

    output_dir = cfg.PATH.experiment_dir / args.net
    state_dict = torch.load(str(output_dir/args.load_name))
    model.load_state_dict(state_dict['model'])
    model.cuda()

    for phase in phaseList:
        model.eval()

        num_steps = 100
        delta = (args.dhigh - args.dlow)/num_steps
        y_true, y_scores = [], []
        for step,data in enumerate(dataloaders[phase]):
            im_data.data.resize_(data[0].size()).copy_(data[0])
            labels.data.resize_(data[1].size()).copy_(data[1])
            batch_size = labels.size(0)

            embeddings, loss, dists, dist_ap, dist_an = model(im_data, labels)

            for i in range(batch_size-1):
                for j in range(i+1,batch_size):
                    y_true.append(labels[i].eq(labels[j]).int().item())
                    y_scores.append(dists[i,j].item())

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        true_label_indices = np.arange(len(y_true))[y_true==1]
        false_label_indices = np.arange(len(y_true))[y_true==0]

        selected_false_indices = np.random.choice(false_label_indices, size=len(true_label_indices),replace=False)

        indices = np.concatenate((true_label_indices,selected_false_indices))

        mAP = average_precision_score(y_true[indices], y_scores[indices])
        print("mAP:{}".format(mAP))
        print('-'*15)

        embed()

















