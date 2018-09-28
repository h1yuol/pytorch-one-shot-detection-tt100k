import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from IPython import embed

import sys
sys.path.append('./lib')
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes

from model.faster_rcnn.resnet import resnet101

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader

from config import cfg
from utils import compute_mat_dist, computeIOU_torch
import pdb

try:
  xrange
except:
  xrange = range

class pseudo_siamese_det(nn.Module):
  def __init__(self, det_model_path):
    super(pseudo_siamese_det, self).__init__()
    self.det_model_path = det_model_path
    self.dout_base_model = 1024
    # self.resnet101_path = '/home/huangyucheng/MYDATA/git-cores/FasterRCNN/data/pretrained_model/resnet101_caffe.pth'



    # define rpn
    self.RCNN_rpn = _RPN(self.dout_base_model)
    # self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
    self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
    self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

    self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
    self.RCNN_roi_crop = _RoICrop()

    


  def _init_modules(self, load_det_model=True):
    resnet = resnet101()

    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)
    self.RCNN_top = nn.Sequential(resnet.layer4)
    self.RCNN_bbox_pred = nn.Linear(2048, 4)

    if load_det_model:
      state_dict = torch.load(self.det_model_path)['model']
      self.load_state_dict({k:v for k,v in state_dict.items() if k in self.state_dict()})

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7

  def forward(self, im_data, im_info, gt_boxes, num_boxes):
    """
    Input:
      - im_data: (batch_size, )
      ......
    """
    batch_size = im_data.size(0)

    im_info = im_info.data
    gt_boxes = gt_boxes.data
    num_boxes = num_boxes.data

    base_feat = self.RCNN_base(im_data)

    rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

    rois = Variable(rois)

    if cfg.POOLING_MODE == 'crop':
      grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
      grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
      pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
      if cfg.CROP_RESIZE_WITH_MAX_POOL:
        pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
    elif cfg.POOLING_MODE == 'align':
      pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
    elif cfg.POOLING_MODE == 'pool':
      pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

    pooled_feat = self._head_to_tail(pooled_feat)

    bbox_pred = self.RCNN_bbox_pred(pooled_feat)
    bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

    return rois, bbox_pred

  def create_architecture(self, load_det_model=True):
    self._init_modules(load_det_model=load_det_model)

class pseudo_siamese(nn.Module):

  def __init__(self, det_model_path, oneshot_model_path, embedding_dim, threshold):
    super(pseudo_siamese, self).__init__()
    self.det_model_path = det_model_path
    self.oneshot_model_path = oneshot_model_path
    self.embedding_dim = embedding_dim
    self.threshold = threshold
    self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
    self.pixel_means = Variable(torch.Tensor(cfg.PIXEL_MEANS).view(1,3,1,1))
    self.mean = Variable(torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
    self.std = Variable(torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

  def _init_modules(self, load_model=True):
    resnet = resnet101()

    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)

    self.Linear_top = nn.Linear(2048, self.embedding_dim)

    if load_model:
      state_dict = torch.load(self.oneshot_model_path)['model']
      self.load_state_dict({k:v for k,v in state_dict.items() if k in self.state_dict()})

    self.det_module = pseudo_siamese_det(self.det_model_path)
    self.det_module.create_architecture(load_det_model=load_model)
    self.det_module.training = False

    self.RCNN_roi_crop = _RoICrop()
    self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
    self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)


  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7

  def forward(self, im_data, im_info, gt_boxes, num_boxes):
    batch_size = im_data.size(0)

    rois, bbox_pred = self.det_module(im_data, im_info, gt_boxes, num_boxes)
    rois = Variable(rois)

    im_data = ((im_data+self.pixel_means.cuda())/256.0 - self.mean.cuda()) / self.std.cuda()

    base_feat_big = self.RCNN_base(im_data)

    if cfg.POOLING_MODE == 'crop':
      grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat_big.size()[2:], self.grid_size)
      grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
      pooled_feat_big = self.RCNN_roi_crop(base_feat_big, Variable(grid_yx).detach())
      if cfg.CROP_RESIZE_WITH_MAX_POOL:
        pooled_feat_big = F.max_pool2d(pooled_feat_big, 2, 2)
    elif cfg.POOLING_MODE == 'align':
      pooled_feat_big = self.RCNN_roi_align(base_feat_big, rois.view(-1, 5))
    elif cfg.POOLING_MODE == 'pool':
      pooled_feat_big = self.RCNN_roi_pool(base_feat_big, rois.view(-1,5))

    pooled_feat_big = self._head_to_tail(pooled_feat_big)

    embeddings_data = self.Linear_top(pooled_feat_big)

    # base_feat_template = self.RCNN_base(im_obj)
    # pooled_feat_template = self._head_to_tail(base_feat_template)
    # embeddings_obj = self.Linear_top(pooled_feat_template)

    dists = compute_mat_dist(embeddings_data.data, self.embeddings_obj.data)
    tup = dists.min(dim=1)
    # predict_labels = dists.argmin(dim=1)
    predict_not_background = tup[0].le(self.threshold)
    predict_labels = tup[1]

    return predict_labels, predict_not_background, rois, bbox_pred

  def create_architecture(self, load_model=True):
    self._init_modules(load_model=load_model)

  def precompute_gallery_tensors(self, im_obj, im_labels):
    base_feat_template = self.RCNN_base(im_obj)
    pooled_feat_template = self._head_to_tail(base_feat_template)
    self.embeddings_obj = self.Linear_top(pooled_feat_template)
    self.gallery_labels = im_labels


def get_gallery():
  import json
  gallery_transforms = transforms.Compose([
                            transforms.Resize([cfg.TRAIN.sign_input_size, cfg.TRAIN.sign_input_size]),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ])
  gallery_dataset = ImageFolder(str(cfg.PATH.root_dir/'data'/'gallery'), gallery_transforms)
  gallery_dataloader = torch.utils.data.DataLoader(
                          gallery_dataset, 
                          batch_size=len(gallery_dataset),
                          num_workers=1,
                      )
  gallery_im, gallery_labels = next(iter(gallery_dataloader))
  # mapping gallery_labels to all_classes based indices
  idx_to_class = {v:k for k,v in gallery_dataset.class_to_idx.items()}
  all_classes = json.loads(open(str(cfg.PATH.dataset_dir/'all_classes.json')).read())
  class_to_newIdx = {cls:idx for idx, cls in enumerate(all_classes)}
  gallery_labels = list(map(lambda label: class_to_newIdx[idx_to_class[label.item()]], gallery_labels))
  gallery_labels = torch.Tensor(gallery_labels)
  return gallery_im, gallery_labels

def display(im_data, labels):
  import torchvision
  import matplotlib.pyplot as plt
  import numpy as np
  out = torchvision.utils.make_grid(im_data)
  def imshow(inp, title=None):
      """Imshow for Tensor."""
      inp = inp.numpy().transpose((1, 2, 0))
      mean = np.array([0.485, 0.456, 0.406])
      std = np.array([0.229, 0.224, 0.225])
      inp = std * inp + mean
      inp = np.clip(inp, 0, 1)
      plt.imshow(inp)
      if title is not None:
          plt.title(title)
      plt.pause(0.001)

  imshow(out, title=labels)








if __name__ == '__main__':
  det_model_path = '/home/huangyucheng/MYDATA/EXPERIMENTS/FasterRCNN/2gpu_0831/res101/tt100k/faster_rcnn_1_50_2032.pth'
  oneshot_model_path = '/home/huangyucheng/MYDATA/public_git/pytorch-one-shot-detection-tt100k/experiment/res101/20180914-14:54:02fstres101_marg0.5_bs128_optadam_lr0.001decStep7decGamma0.5_ses0/models/0_100_47.pth'
  model = pseudo_siamese(det_model_path, oneshot_model_path, embedding_dim=128, threshold=1.0)
  model.create_architecture()
  model.cuda()

  imdb, roidb, ratio_list, ratio_index = combined_roidb("tt100k_test", training=False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))



  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(imdb.num_classes)]
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, imdb.num_classes, training=True, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)
  data_iter = iter(dataloader)

  gallery_im, gallery_labels = get_gallery()

  # initilize the tensor holder here.
  im_data = Variable(torch.FloatTensor(1).cuda())
  im_info = Variable(torch.FloatTensor(1).cuda())
  num_boxes = Variable(torch.LongTensor(1).cuda())
  gt_boxes = Variable(torch.FloatTensor(1).cuda())

  gallery_im = gallery_im.cuda()
  gallery_labels = gallery_labels.cuda()

  model.precompute_gallery_tensors(gallery_im, gallery_labels)

  with torch.no_grad():
    model.eval()
    for i in range(num_images):
      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      predict_labels, predict_not_background, rois, bbox_pred = model(im_data, im_info, gt_boxes, num_boxes)

      boxes = rois.data[:, :, 1:5]
      if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
          box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
          box_deltas = box_deltas.view(1, -1, 4)

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

      # pred_boxes /= data[1][0][2].item()

      num_box = num_boxes[0].item()
      iou = computeIOU_torch(pred_boxes[0], gt_boxes[0,:num_box,:4])

      maxIou, maxInd = iou.max(dim=0, keepdim=True)

      print(maxIou)
      print(maxInd)
      tmp = torch.zeros(222).int()
      tmp[predict_labels * predict_not_background.long()] = 1
      print(torch.arange(222)[tmp==1])


      embed()














