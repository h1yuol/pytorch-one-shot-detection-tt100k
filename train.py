import torch

from lib.model.one_shot.one_shot_resnet import one_shot_resnet

model_path = '/home/huangyucheng/MYDATA/EXPERIMENTS/FasterRCNN/2gpu_0831/res101/tt100k/faster_rcnn_1_50_2032.pth'

model = one_shot_resnet(128, model_path, pretrained=True)

model.create_architecture()
# model.test()

from IPython import embed
embed()