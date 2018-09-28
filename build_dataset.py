import argparse
from PIL import Image
from tqdm import tqdm
import json
from torchvision.datasets import ImageFolder

from config import cfg



def get_annotations():
    with (cfg.PATH.dataset_dir / 'annotations.json').open() as hd:
        text = hd.read()
        annotations = json.loads(text)
    return annotations

def get_imageList(phase):
    assert phase in ['train','test','other']
    imageList = []
    for file in (cfg.PATH.dataset_dir / phase).glob('*'):
        if file.name.endswith('.jpg'):
            sample = {
                'img_id': file.name.split('.')[0],
                'path': file.absolute(),
            }
            imageList.append(sample)
    return imageList

def crop_resize_save(phase,sample,objects,size=cfg.TRAIN.sign_input_size):
    img = Image.open(sample['path'])
    for objIdx,obj in enumerate(objects):
        bbox = obj['bbox']
        category = obj['category']
        img_obj = img.crop(box=(bbox['xmin'],bbox['ymin'],bbox['xmax'],bbox['ymax']))
        # img_obj = img_obj.resize((size, size), Image.BILINEAR)
        (cfg.PATH.sign_data_dir / phase / category).mkdir(parents=True,exist_ok=True)
        img_obj.save(str(cfg.PATH.sign_data_dir / phase / category / (sample['img_id']+'_'+str(objIdx) + '.jpg')))

def build(phase):
    annotations = get_annotations()
    imageList = get_imageList(phase)
    print('--------------------------------------------')
    print('building {} dataset...'.format(phase))
    count = 0
    for sample in tqdm(imageList):
        try:
            crop_resize_save(phase,sample,annotations['imgs'][sample['img_id']]['objects'])
            count += 1
        except KeyboardInterrupt as e:
            raise KeyboardInterrupt
        except Exception as e:
            errorMessage = str(e)
            print("encountered an error when dealing with {}/{}:".format(phase,(sample['img_id']+'.jpg')))
            print(errorMessage)
    print('finished building {} dataset!'.format(phase))
    print('successful processing {} images'.format(count))

def test_get_imageList():
    assert len(get_imageList('train')) == 6100
    assert len(get_imageList('test')) == 3071
    assert len(get_imageList('other')) == 7638

def test_crop_resize_save():
    annotations = get_annotations()
    imageList = get_imageList('train')
    sample = list(filter(lambda sample: sample['img_id']=='57154', imageList))[0]
    crop_resize_save('train', sample, annotations['imgs'][sample['img_id']]['objects'])

# ---------------------------------------------

def crop_gallery(path, obj):
    img = Image.open(str(cfg.PATH.dataset_dir / path))
    bbox = obj['bbox']
    category = obj['category']
    img_obj = img.crop(box=(bbox['xmin'],bbox['ymin'],bbox['xmax'],bbox['ymax']))
    (cfg.PATH.root_dir / 'data' / 'gallery' / category).mkdir(parents=True, exist_ok=True)
    img_obj.save(str(cfg.PATH.root_dir / 'data' / 'gallery' / category / "{}.png".format(category)))

def build_gallery():
    className2imgDict = json.loads(open(str(cfg.PATH.dataset_dir/'className2imgDict.json')).read())
    lst = []
    for className in tqdm(className2imgDict):
        flag = False
        for imgDict in className2imgDict[className]:
            if not imgDict['path'].startswith('test/'):
                flag = True
                crop_gallery(imgDict['path'], imgDict['object'])
                break
        if not flag:
            lst.append(className)
            imgDict = className2imgDict[className][0]
            crop_gallery(imgDict['path'], imgDict['object'])
    print("The following classes only has test-dataset gallery:")
    print(lst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', default='phase', help='specify what you want to do')
    parser.add_argument('--phase', default='train+test', help='specify the dataset you want to build')
    args = parser.parse_args()

    if args.build == 'phase':
        phaseList = args.phase.split('+')

        for phase in phaseList:
            if phase not in ['train','test','other']:
                print("You entered wrong phase: {}".format(phase))
                continue
            else :
                build(phase)
    elif args.build == 'gallery':
        build_gallery()


