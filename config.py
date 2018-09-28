from easydict import EasyDict as edict 
from pathlib import Path

__C = edict()

__C.PATH = edict()
__C.PATH.dataset_dir = Path('/unsullied/sharefs/huangyucheng/data/DATASETS/traffic-sign/data')
__C.PATH.root_dir = Path('/home/huangyucheng/MYDATA/public_git/pytorch-one-shot-detection-tt100k/')
__C.PATH.sign_data_dir = __C.PATH.root_dir / 'data'
__C.PATH.sign_data_dir.mkdir(exist_ok=True, parents=True)
__C.PATH.experiment_dir = __C.PATH.root_dir / 'experiment'
__C.PATH.experiment_dir.mkdir(exist_ok=True, parents=True)

__C.TRAIN = edict()
__C.TRAIN.pooling_size = 7
__C.TRAIN.sign_input_size = __C.TRAIN.pooling_size * 16
__C.TRAIN.norm = False
__C.TRAIN.embedding_dim = 128

__C.TRAIN.init_model = '/home/huangyucheng/MYDATA/EXPERIMENTS/FasterRCNN/2gpu_0831/res101/tt100k/faster_rcnn_1_50_2032.pth'
__C.TRAIN.P = 32
__C.TRAIN.K = 4
__C.TRAIN.double_bias = True
__C.TRAIN.bias_decay = False
__C.TRAIN.weight_decay = 5e-4
__C.TRAIN.momentum = 0.9

__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

__C.TEST = edict()
__C.TEST.BBOX_REG = True


__C.POOLING_MODE = 'align'
__C.CROP_RESIZE_WITH_MAX_POOL = True
__C.POOLING_SIZE = 7
__C.PIXEL_MEANS = [102.9801, 115.9465, 122.7717]


cfg = __C