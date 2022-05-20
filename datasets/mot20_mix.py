from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
try:
  from .generic_dataset_mix import GenericDataset
except:
  from generic_dataset_mix import GenericDataset


class MOT20(GenericDataset):
  num_classes = 1
  default_resolution = [640, 1088]
  max_objs = 500
  class_name = ['person']
  cat_ids = {1: 1}

  def __init__(self, opt, split):
    super(MOT20, self).__init__()
    data_dir = opt.data_dir
    data_dir_ch = opt.data_dir_ch

    if split == 'test':
      img_dir = os.path.join(
        data_dir, 'test')
    else:
      img_dir = os.path.join(
        data_dir, 'train')

    if not opt.ignoreIsCrowd:
      ann_dir = 'annotations'
    else:
      ann_dir = 'annotations_withIgnore'

    if split == 'train':
      if opt.small:
        ann_path = os.path.join(data_dir, ann_dir, '{}_half.json').format(split)
        ann_path_ch = os.path.join(data_dir_ch, 'annotations_small', '{}.json').format(split)
      else:
        ann_path = os.path.join(data_dir, ann_dir, '{}.json').format(split)
        ann_path_ch = os.path.join(data_dir_ch, 'annotations_mix', '{}.json').format(split)

      img_dir_ch = os.path.join(
      data_dir_ch, 'Images')
    else:
      ann_path = os.path.join(data_dir, ann_dir,
                              '{}_half.json').format(split)

      ann_path_ch = None
      img_dir_ch = None

    print('==> initializing MOT20 {} data and CH {} data.'.format(split, split))

    self.images = None
    # load image list and coco
    super(MOT20, self).__init__(opt, split, ann_path, img_dir, img_dir_ch=img_dir_ch, ann_path_ch=ann_path_ch)

    self.num_samples = len(self.merged_images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def __len__(self):
    return self.num_samples