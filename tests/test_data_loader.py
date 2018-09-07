
from torchvision.datasets import ImageFolder
from torchvision import transforms

from lib.datasets.data_loader import MyBatchSampler, get_dataloaders
from config import cfg

def test_MyBatchSampler():
	data_transforms = transforms.Compose([
	    transforms.Resize([112,112]),
	    transforms.ToTensor(),
	    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	])
	train = ImageFolder(str(cfg.PATH.sign_data_dir / 'train'), data_transforms)
	sampler = MyBatchSampler(train.samples, 32, 4)
	it = iter(sampler)
	lst = next(it)
	assert len(lst) == 128
	dct = {}
	for idx in lst:
		classIdx = train.samples[idx][1]
		dct[classIdx] = dct.get(classIdx, 0) + 1
	assert len(dct) == 32
	for v in dct.values():
		assert v == 4

def test_get_dataloaders():
	num_workers = {
		'train': 1,
		'test': 1,
	}
	dataloaders, dataset_sizes = get_dataloaders(num_workers, 32, 4, ['test'])
	from IPython import embed
	embed()

if __name__ == '__main__':
	num_workers = {
		'train': 1,
		'test': 1,
	}
	dataloaders, dataset_sizes = get_dataloaders(num_workers, 32, 4, ['test'])
	from IPython import embed
	embed()