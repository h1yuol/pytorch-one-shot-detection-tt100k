from easydict import EasyDict as edict 
from pathlib import Path




__C = edict()

__C.PATH = edict()
__C.PATH.dataset_dir = Path('/unsullied/sharefs/huangyucheng/data/DATASETS/traffic-sign/data')
__C.PATH.root_dir = Path('/home/huangyucheng/MYDATA/public_git/pytorch-one-shot-detection-tt100k/')
__C.PATH.sign_data_dir = __C.PATH.root_dir / 'data'

__C.TRAIN = edict()
__C.TRAIN.pooling_size = 7
__C.TRAIN.sign_input_size = __C.TRAIN.pooling_size * 16

cfg = __C