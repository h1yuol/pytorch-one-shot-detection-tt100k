from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
	xrange          # Python 2
except NameError:
	xrange = range  # Python 3

# <<<< obsolete

class tt100k(imdb):
	def __init__(self, phase):
		if phase in ['validation','val']:
			phase = 'test'
		elif phase in ['training']:
			phase = 'train'
		imdb.__init__(self, 'tt100k' + '_' + phase)
		self._phase = phase
		self.dataset_dir = self._get_default_path()
		self._data_path = os.path.join(self.dataset_dir, 'data')
#         self._classes = self._get_marks()
#         self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
		import json
		annos = json.loads(open(os.path.join(self._data_path,'annotations.json')).read())
		# self._classes = tuple(['__background__'] + sorted(annos['types']))
		self._classes = json.loads(open(str(cfg.DATA_DIR+'/tt100k/data/all_classes.json')).read())
		self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
		self._image_ext = '.jpg'
		self._image_index = self._get_img_ids()
		# Default to roidb handler
		self._roidb_handler = self.gt_roidb

		assert os.path.exists(self._data_path), \
			'Path does not exist: {}'.format(self._data_path)

		self.config = {'use_salt': True,
                   'cleanup': True}
		
	def gt_roidb(self):
		"""
		Return the database of ground-truth regions of interest.

		This function loads/saves from/to a cache file to speed up future calls.
		"""
		cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
		if os.path.exists(cache_file):
			with open(cache_file, 'rb') as fid:
				roidb = pickle.load(fid)
			print('{} gt roidb loaded from {}'.format(self.name, cache_file))
			return roidb
		
		import json
		annos = json.loads(open(os.path.join(self._data_path,'annotations.json')).read())
		# check classes
#         print(len(self.classes))
#         print('=======')
#         print(len(tuple(['__background__'] + sorted(annos['types']))))
#         st = set()
#         self._classes = tuple(['__background__'] + sorted(annos['types']))
#         pickle.dump(self.classes, open(os.path.join(self.cache_path, self.name+'_classes.pkl','wb')))
#         self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
#         print('number of classes changed to',len(self.classes))
		# deal with annotation
		img_id2anno = annos['imgs']
		gt_roidb = []
		for img_id in self.image_index:
			anno = img_id2anno[str(img_id)]
			path = anno['path']
			num_objs = len(anno['objects'])
			boxes = np.zeros((num_objs, 4), dtype=np.uint16)
			gt_classes = np.zeros((num_objs), dtype=np.int32)
			overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
			for ix,obj in enumerate(anno['objects']):
				bbox = obj['bbox']
				x1 = max(0,bbox['xmin'])
				y1 = max(0,bbox['ymin'])
				x2 = max(0,bbox['xmax'])
				y2 = max(0,bbox['ymax'])
				cls = self._class_to_ind[obj['category']]
#                 assert (np.array([x1,y1,x2,y2]) >= 0).all(), (x1,y1,x2,y2)
				boxes[ix,:] = [x1,y1,x2,y2]
				gt_classes[ix] = cls
				overlaps[ix, cls] = 1.0
			overlaps = scipy.sparse.csr_matrix(overlaps)
			gt_roidb.append({
				'boxes': boxes,
				'gt_classes': gt_classes,
				'gt_overlaps': overlaps,
				'flipped': False,
			})
		assert len(gt_roidb) == len(self.image_index)
		with open(cache_file, 'wb') as fid:
			pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
		print('wrote gt roidb to {}'.format(cache_file))

		return gt_roidb
		  
	def image_path_at(self, i):
		"""
		Return the absolute path to image i in the image sequence.
		"""
		return self.image_path_from_index(self._image_index[i])

	def image_id_at(self, i):
		"""
		Return the absolute path to image i in the image sequence.
		"""
		return i

	def image_path_from_index(self, index):
		"""
		Construct an image path from the image's "index" identifier.
		"""
		image_path = os.path.join(self._data_path, self.phase, str(index) + self._image_ext)
		assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
		return image_path
			
	def _get_img_ids(self):
		ids = []
#         with open(os.path.join(self._data_path,self.phase,'ids.txt')) as hd:
#             for line in hd:
#                 if line:
#                     img_id = int(line)
#                 ids.append(img_id)
		ids = os.popen('ls {}/*.jpg'.format(os.path.join(self._data_path, self.phase))).read().strip().split('\n')
		ids = list(map(lambda f: int(f.split('/')[-1][:-4]), ids))
		return ids
		
			
	def _get_marks(self):
		"""
		Return all the traffic sign mark classes
		"""
		classes = []
		with open(os.path.join(self._data_path,'marks','genlist.txt')) as hd:
			for line in hd:
				if len(line)>10:
					line = line.strip()
					classes.append(line.split('/')[-1].replace('.png',''))
		classes.sort()
		classes = tuple(['__background__'] + classes)
		return classes
			
	def _get_default_path(self):
		"""
		Return the default path where traffic-sign dataset is settled.
		"""
		from pathlib import Path
		data_dir = Path(cfg.DATA_DIR)
		data_dir.mkdir(exist_ok=True)
		if not (data_dir/'tt100k').exists():
			os.popen('ln -s /home/huangyucheng/MYDATA/DATASETS/traffic-sign {}/tt100k'.format(str(data_dir)))
		return os.path.join(cfg.DATA_DIR, 'tt100k')
	
	@property
	def phase(self):
		return self._phase


	# def _write_tt100k_results_file(self, all_boxes, res_file):
 #    # [{"image_id": 42,
 #    #   "category_id": 18,
 #    #   "bbox": [258.15,41.29,348.26,243.78],
 #    #   "score": 0.236}, ...]
 #    	raise NotImplementedError
	#     results = []
	#     for cls_ind, cls in enumerate(self.classes):
	#       if cls == '__background__':
	#         continue
	#       print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
	#                                                        self.num_classes - 1))
	#       coco_cat_id = self._class_to_coco_cat_id[cls]
	#       results.extend(self._coco_results_one_category(all_boxes[cls_ind],
	#                                                      coco_cat_id))
	#     print('Writing results json to {}'.format(res_file))
	#     with open(res_file, 'w') as fid:
	#       json.dump(results, fid)

	def evaluate_detections(self, all_boxes, output_dir):
		res_file = osp.join(output_dir, ('detections_' +
										 self._image_set +   # image_set is 'test'
										 '_results'))
		if self.config['use_salt']:
			res_file += '_{}'.format(str(uuid.uuid4()))
		res_file += '.json'  # detections_test_results_xxxx.json
		self._write_tt100k_results_file(all_boxes, res_file)
		# Only do evaluation on non-test sets
		if self._image_set.find('test') == -1:
			self._do_detection_eval(res_file, output_dir)
		# Optionally cleanup results json file
		if self.config['cleanup']:
			os.remove(res_file)

	def competition_mode(self, on):
		if on:
			self.config['use_salt'] = False
			self.config['cleanup'] = False
		else:
			self.config['use_salt'] = True
			self.config['cleanup'] = True


# class pascal_voc(imdb):
	

#     def selective_search_roidb(self):
#         """
#         Return the database of selective search regions of interest.
#         Ground-truth ROIs are also included.

#         This function loads/saves from/to a cache file to speed up future calls.
#         """
#         cache_file = os.path.join(self.cache_path,
#                                   self.name + '_selective_search_roidb.pkl')

#         if os.path.exists(cache_file):
#             with open(cache_file, 'rb') as fid:
#                 roidb = pickle.load(fid)
#             print('{} ss roidb loaded from {}'.format(self.name, cache_file))
#             return roidb

#         if int(self._year) == 2007 or self._image_set != 'test':
#             gt_roidb = self.gt_roidb()
#             ss_roidb = self._load_selective_search_roidb(gt_roidb)
#             roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
#         else:
#             roidb = self._load_selective_search_roidb(None)
#         with open(cache_file, 'wb') as fid:
#             pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
#         print('wrote ss roidb to {}'.format(cache_file))

#         return roidb

#     def _load_selective_search_roidb(self, gt_roidb):
#         filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
#                                                 'selective_search_data',
#                                                 self.name + '.mat'))
#         assert os.path.exists(filename), \
#             'Selective search data not found at: {}'.format(filename)
#         raw_data = sio.loadmat(filename)['boxes'].ravel()

#         box_list = []
#         for i in xrange(raw_data.shape[0]):
#             boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
#             keep = ds_utils.unique_boxes(boxes)
#             boxes = boxes[keep, :]
#             keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
#             boxes = boxes[keep, :]
#             box_list.append(boxes)

#         return self.create_roidb_from_box_list(box_list, gt_roidb)

#     def rpn_roidb(self):
#         if int(self._year) == 2007 or self._image_set != 'test':
#             gt_roidb = self.gt_roidb()
#             rpn_roidb = self._load_rpn_roidb(gt_roidb)
#             roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
#         else:
#             roidb = self._load_rpn_roidb(None)

#         return roidb
	
#     def _load_rpn_roidb(self, gt_roidb):
#         filename = self.config['rpn_file']
#         print('loading {}'.format(filename))
#         assert os.path.exists(filename), \
#             'rpn data not found at: {}'.format(filename)
#         with open(filename, 'rb') as f:
#             box_list = pickle.load(f)
#         return self.create_roidb_from_box_list(box_list, gt_roidb)
	

#     def _do_python_eval(self, output_dir='output'):
#         annopath = os.path.join(
#             self._devkit_path,
#             'VOC' + self._year,
#             'Annotations',
#             '{:s}.xml')
#         imagesetfile = os.path.join(
#             self._devkit_path,
#             'VOC' + self._year,
#             'ImageSets',
#             'Main',
#             self._image_set + '.txt')
#         cachedir = os.path.join(self._devkit_path, 'annotations_cache')
#         aps = []
#         # The PASCAL VOC metric changed in 2010
#         use_07_metric = True if int(self._year) < 2010 else False
#         print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
#         if not os.path.isdir(output_dir):
#             os.mkdir(output_dir)
#         for i, cls in enumerate(self._classes):
#             if cls == '__background__':
#                 continue
#             filename = self._get_voc_results_file_template().format(cls)
#             rec, prec, ap = voc_eval(
#                 filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
#                 use_07_metric=use_07_metric)
#             aps += [ap]
#             print('AP for {} = {:.4f}'.format(cls, ap))
#             with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
#                 pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
#         print('Mean AP = {:.4f}'.format(np.mean(aps)))
#         print('~~~~~~~~')
#         print('Results:')
#         for ap in aps:
#             print('{:.3f}'.format(ap))
#         print('{:.3f}'.format(np.mean(aps)))
#         print('~~~~~~~~')
#         print('')
#         print('--------------------------------------------------------------')
#         print('Results computed with the **unofficial** Python eval code.')
#         print('Results should be very close to the official MATLAB eval code.')
#         print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
#         print('-- Thanks, The Management')
#         print('--------------------------------------------------------------')

#     def _do_matlab_eval(self, output_dir='output'):
#         print('-----------------------------------------------------')
#         print('Computing results with the official MATLAB eval code.')
#         print('-----------------------------------------------------')
#         path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
#                             'VOCdevkit-matlab-wrapper')
#         cmd = 'cd {} && '.format(path)
#         cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
#         cmd += '-r "dbstop if error; '
#         cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
#             .format(self._devkit_path, self._get_comp_id(),
#                     self._image_set, output_dir)
#         print('Running:\n{}'.format(cmd))
#         status = subprocess.call(cmd, shell=True)

#     def evaluate_detections(self, all_boxes, output_dir):
#         self._write_voc_results_file(all_boxes)
#         self._do_python_eval(output_dir)
#         if self.config['matlab_eval']:
#             self._do_matlab_eval(output_dir)
#         if self.config['cleanup']:
#             for cls in self._classes:
#                 if cls == '__background__':
#                     continue
#                 filename = self._get_voc_results_file_template().format(cls)
#                 os.remove(filename)

#     def competition_mode(self, on):
#         if on:
#             self.config['use_salt'] = False
#             self.config['cleanup'] = False
#         else:
#             self.config['use_salt'] = True
#             self.config['cleanup'] = True


if __name__ == '__main__':
	d = tt100k('train')
	res = d.roidb
	from IPython import embed

	embed()
