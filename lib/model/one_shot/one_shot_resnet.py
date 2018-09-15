import torch
import torch.nn as nn

import sys
sys.path.append('/unsullied/sharefs/huangyucheng/data/public_git/pytorch-one-shot-detection-tt100k/lib')
from model.faster_rcnn.resnet import resnet101
from model.one_shot.triplet_loss import batch_hard_triplet_loss

class TripletLoss(nn.Module):
  """Triplet loss with hard positive/negative mining.
  Reference:
  Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
  Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
  Args:
      margin (float): margin for triplet.
  """
  def __init__(self, margin=0.3, mutual_flag = False, an_weight=10):
    super(TripletLoss, self).__init__()
    self.margin = margin
    self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    self.mutual = mutual_flag
    self.an_weight = an_weight

  def forward(self, inputs, targets):
    """
    Args:
        inputs: feature matrix with shape (batch_size, feat_dim)
        targets: ground truth labels with shape (num_classes)
    """
    n = inputs.size(0)
    # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs, inputs.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    # For each anchor, find the hardest positive and negative
    mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    dist_ap, dist_an = [], []
    for i in range(n):
      dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
      dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)
    # Compute ranking hinge loss
    # y = torch.ones_like(dist_an)
    # loss = self.ranking_loss(dist_an, dist_ap, y)
    loss = (dist_ap - dist_an + self.margin).clamp(min=0).mean()
    # loss += - self.an_weight * dist_an.min()
    if self.mutual:
      return loss, dist, dist_ap, dist_an
    return loss

class one_shot_resnet(nn.Module):
  def __init__(self, embedding_dim, model_path, margin, pretrained=False, norm=False, onlyEmbeddings=False):
    self.model_path = model_path
    self.pretrained = pretrained
    self.embedding_dim = embedding_dim
    self.margin = margin
    self.norm = norm
    self.onlyEmbeddings = onlyEmbeddings

    super(one_shot_resnet, self).__init__()

  def _load_pretrained_model(self, resnet):
    print("Loading pretrained weights from %s" %(self.model_path))
    state_dict = torch.load(self.model_path)
    # get nameMap
    nameMap = {}
    for k,v in state_dict['model'].items():
      if k.startswith('RCNN_base'):
        klist = k.split('.')
        if klist[1] == '0':
          newklist = ['conv1',klist[-1]]
        elif klist[1] == '1':
          newklist = ['bn1',klist[-1]]
        else :
          newklist = ['layer'+str(int(klist[1])-3)] + klist[2:]
        newk = '.'.join(newklist)
        nameMap[k] = newk
      elif k.startswith('RCNN_top'):
        klist = k.split('.')
        newklist = ['layer4'] + klist[2:]
        newk = '.'.join(newklist)
        nameMap[k] = newk
    dct = {nameMap[k]:v for k,v in state_dict['model'].items() if k in nameMap}
    dct['fc.weight'] = torch.randn(1000,2048)
    dct['fc.bias'] = torch.randn(1000)
    resnet.load_state_dict(dct)

  def _init_modules(self):
    resnet = resnet101()

    if self.pretrained:
      self._load_pretrained_model(resnet)

    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)

    self.Linear_top = nn.Linear(2048, self.embedding_dim)

    # Fix blocks
    # for p in self.RCNN_base.parameters(): p.requires_grad = False

    # def set_bn_fix(m):
    #   classname = m.__class__.__name__
    #   if classname.find('BatchNorm') != -1:
    #     for p in m.parameters(): p.requires_grad = False

    # self.RCNN_base.apply(set_bn_fix)
    # self.RCNN_top.apply(set_bn_fix)

    self.tripletloss = TripletLoss(margin=self.margin, mutual_flag=True)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7

  # def train(self, mode=True):
  #   # Override train so that the training mode is set as we want
  #   nn.Module.train(self, mode)
  #   if mode:
  #     # Set fixed blocks to be in eval mode
  #     self.RCNN_base.eval()

  #     def set_bn_eval(m):
  #       classname = m.__class__.__name__
  #       if classname.find('BatchNorm') != -1:
  #         m.eval()

  #     self.RCNN_base.apply(set_bn_eval)
  #     self.RCNN_top.apply(set_bn_eval)

  def forward(self, im_data, labels):
    """
    Input:
      - im_data: (batch_size, num_channels, img_height, img_width)
      - labels: (batch_size)
      - margin: int
    """
    batch_size = im_data.size(0)

    base_feat = self.RCNN_base(im_data)
    assert base_feat.size() == (batch_size, 1024, 7, 7)

    pooled_feat = self._head_to_tail(base_feat)
    assert pooled_feat.size() == (batch_size, 2048)

    embeddings = self.Linear_top(pooled_feat)
    # embeddings = pooled_feat

    if self.onlyEmbeddings:
      return embeddings

    if self.norm:
      embeddings_norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
      embeddings = embeddings / embeddings_norm

    # loss = batch_hard_triplet_loss(labels, embeddings, margin, squared=False)
    loss, dists, dist_ap, dist_an = self.tripletloss(embeddings, labels)

    return embeddings, loss, dists, dist_ap, dist_an

  def _init_weights(self):
    def normal_init(m, mean, stddev, truncated=False):
      """
      weight initalizer: truncated normal and random normal.
      """
      # x is a parameter
      if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
      else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

    normal_init(self.Linear_top, 0, 0.01, False)

  def eval_create_architecture(self):
    self._init_modules()

  def create_architecture(self):
    self._init_modules()
    self._init_weights()


