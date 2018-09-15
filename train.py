import numpy as np
import time
import datetime
from pathlib import Path

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
    parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
    parser.add_argument('--net', dest='net',
                      default='res101', type=str)
    parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
    # parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
    #                   help='number of iterations to display',
    #                   default=10000, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
    parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=3, type=int)                    
    parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
    parser.add_argument('--load_name', dest='load_name',
                      help='path to the loading model',
                      type=str)
    parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

    # Other Hyperparams
    parser.add_argument('--margin', dest='margin',
                      help='triplet loss margin',
                      default=0.5, type=float)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    # print('Using config:')
    # pprint.pprint(cfg)

    assert torch.cuda.is_available(), "GPU is in need"

    # ------------------------------- get dataloaders
    num_workers = {}
    num_workers['test'] = args.num_workers//3
    num_workers['train'] = args.num_workers - num_workers['test']
    P, K = cfg.TRAIN.P, cfg.TRAIN.K

    dataloaders, dataset_sizes = get_dataloaders(num_workers, P, K, ['train','test'])

    im_data = torch.FloatTensor(1).cuda()
    labels = torch.FloatTensor(1).cuda()

    im_data = Variable(im_data)
    labels = Variable(labels)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #-------------------------------- Data Visualization
    # it = iter(dataloaders['train'])
    # data = next(it)
    # im_data = data[0]
    # labels = data[1]
    # indices = list(map(lambda tup: tup[0], sorted([(idx, label) for idx,label in enumerate(labels)], key=lambda tup: tup[1])))
    # im_data = im_data[indices]
    # import torchvision
    # import matplotlib.pyplot as plt
    # out = torchvision.utils.make_grid(im_data)
    # def imshow(inp, title=None):
    #     """Imshow for Tensor."""
    #     inp = inp.numpy().transpose((1, 2, 0))
    #     mean = np.array([0.485, 0.456, 0.406])
    #     std = np.array([0.229, 0.224, 0.225])
    #     inp = std * inp + mean
    #     inp = np.clip(inp, 0, 1)
    #     plt.imshow(inp)
    #     if title is not None:
    #         plt.title(title)
    #     plt.pause(0.001)

    # from IPython import embed
    # embed()
    # imshow(out, title=[labels[idx] for idx in indices])

    #-------------------------------- get model here
    model_path = cfg.TRAIN.init_model

    norm = cfg.TRAIN.norm
    embedding_dim = cfg.TRAIN.embedding_dim
    margin = args.margin
    model = one_shot_resnet(embedding_dim, model_path, margin, pretrained=True, norm=norm)

    model.create_architecture()

    lr = args.lr

    params = []
    for key,value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params.append({
                        'params': [value],
                        'lr': lr*(cfg.TRAIN.double_bias+1),
                        'weight_decay': cfg.TRAIN.bias_decay and cfg.TRAIN.weight_decay or 0,
                    })
            else :
                params.append({
                        'params': [value],
                        'lr': lr,
                        'weight_decay': cfg.TRAIN.weight_decay,
                    })

    if args.optimizer == 'adam':
        lr = lr * 0.1
        optimizer = optim.Adam(params)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params, momentum=cfg.TRAIN.momentum)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    output_dir = cfg.PATH.experiment_dir / args.net
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.resume:
        load_name = output_dir / args.load_name
        print("loading checkpoint {}".format(load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("loaded checkpoint {}".format(load_name))

    if args.mGPUs:
        model = torch.nn.DataParallel(model)

    model.cuda()

    args.save_dir = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")) + args.save_dir + "{}_{}{}_{}{}_{}{}_{}{}{}{}{}{}_{}{}".format(
            args.net, 
            'marg', args.margin,
            'bs', args.batch_size,
            'opt', args.optimizer,
            'lr', args.lr,'decStep',args.lr_decay_step,'decGamma',args.lr_decay_gamma,
            'ses',args.session,
        )
    (output_dir / args.save_dir / 'models').mkdir(exist_ok=True, parents=True)
    (output_dir / args.save_dir / 'logs').mkdir(exist_ok=True, parents=True)
    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        train_tb = SummaryWriter(str(output_dir/args.save_dir/'logs'/'train'))
        test_tb = SummaryWriter(str(output_dir/args.save_dir/'logs'/'test'))

    totsteps = {
        x: dataset_sizes[x]//args.batch_size for x in ['train','test']
    }

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        model.train()

        for phase in ['train','test']:
            if phase == 'train':
                exp_lr_scheduler.step()
                model.train()
                logger = train_tb
            else:
                model.eval()
                logger = test_tb

            running_loss = 0.0
            ap_mean, ap_max, an_mean, an_min = [],[],[],[]
            start = time.time()
            epoch_start = start
            for step,data in enumerate(dataloaders[phase]):
                im_data.data.resize_(data[0].size()).copy_(data[0])
                labels.data.resize_(data[1].size()).copy_(data[1])
                dltime = time.time()

                optimizer.zero_grad()

                #forward
                embeddings, loss, dists, dist_ap, dist_an = model(im_data, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * im_data.size(0)
                ap_mean.append(dist_ap.data.mean())
                ap_max.append(dist_ap.data.max())
                an_mean.append(dist_an.data.mean())
                an_min.append(dist_an.data.min())

                if args.use_tfboard and phase == 'train':
                    info = {
                        'loss': loss.item(),
                        'dist_ap_mean': dist_ap.data.mean(),
                        'dist_ap_max': dist_ap.data.max(),
                        'dist_an_mean': dist_an.data.mean(),
                        'dist_an_min': dist_an.data.min(),
                        'learning_rate': exp_lr_scheduler.get_lr()[0],
                    }
                    for k,v in info.items():
                        logger.add_scalar(k,v,((epoch-1)*totsteps[phase] + step)*totsteps['train']//totsteps[phase])

                end = time.time()
                if step % args.disp_interval == 0:
                    print("{} e:{}/{} step:{} loss:{:.4g} ap_mean:{:.4g} {:.4g} an_mean:{:.4g} {:.4g} dload:{:.3g}".format(
                            phase, epoch,args.max_epochs, step, 
                            loss.item(), dist_ap.data.mean(), dist_ap.data.max(), dist_an.data.mean(), dist_ap.data.min(),
                            (dltime-start)/(end-start)
                        ))
                start = time.time()

            epoch_loss = running_loss / dataset_sizes[phase]
            print('Finish {} e:{} loss:{:4g} dist_ap:{:4g} {:4g} dist_an:{:4g} {:4g} time:{:3g}'.format(
                    phase, epoch, epoch_loss, 
                    np.mean(ap_mean), np.mean(ap_max),
                    np.mean(an_mean), np.mean(an_min),
                    start - epoch_start
                ))

            if args.use_tfboard and phase == 'test':
                info = {
                    'loss': epoch_loss,
                    'dist_ap_mean': np.mean(ap_mean),
                    'dist_ap_max': np.mean(ap_max),
                    'dist_an_mean': np.mean(an_mean),
                    'dist_an_min': np.mean(an_min),
                    'learning_rate': exp_lr_scheduler.get_lr()[0],
                }
                for k,v in info.items():
                    logger.add_scalar(k,v,((epoch-1)*totsteps[phase] + step)*totsteps['train']//totsteps[phase])

            # if args.use_tfboard:
            #     info = {
            #         'loss': epoch_loss,
            #         'dist_ap_mean': np.mean(ap_mean),
            #         'dist_ap_max': np.mean(ap_max),
            #         'dist_an_mean': np.mean(an_mean),
            #         'dist_an_min': np.mean(an_min),
            #     }
            #     for k,v in info.items():
            #         logger.add_scalar(k,v,(epoch-1))
            #     # logger.add_scalars("logs_{}".format(phase), info, (epoch-1))

            if phase == 'train':
                save_name = str(output_dir/args.save_dir/'models'/"{}_{}_{}.pth".format(args.session, epoch, step))
                state = {
                    'session': args.session,
                    'epoch': epoch,
                    'model': model.module.state_dict() if args.mGPUs else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, save_name)
                print('save model: {}'.format(save_name))

            print('-'*20)

    if args.use_tfboard:
        train_tb.close()
        test_tb.close()

    from IPython import embed
    embed()

"""
- 根据labels.data计算2D valid_positive_mask 和 valid_negative_mask
- 根据mask分别计算 mean dist，并输出
- 根据mask计算给定threshold d的情况下，true accepts rate 和 false accepts rate，并输出
"""







