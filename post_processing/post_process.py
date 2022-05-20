from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from util.image import transform_preds_with_trans, get_affine_transform
import torch

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)


def generic_post_process(opt, dets, pre_cts, dws, dhs, ratios, filter_by_scores=0.3):
  if not ('scores' in dets):
    return [{}], [{}]

  ret = []
  pre_ret = []
  assert len(dws) == len(dhs) == len(ratios) == len(dets['scores'])
  # batch #
  for i in range(len(dets['scores'])):

    if 'tracking' in dets:
      pre_item = {}
      # B,M,2
      # print("pre_cts.shape ", pre_cts.shape)
      # print("dets['tracking'].shape ", dets['tracking'].shape)
      assert pre_cts.shape == dets['tracking'].shape

      # displacement to original image space
      # print(pre_cts.device)
      # print(dets['tracking'].device)
      tracking = opt.down_ratio * (dets['tracking'][i] + pre_cts[i])

      tracking[:, 0] -= dws[i]
      tracking[:, 1] -= dhs[i]
      tracking /= ratios[i]

      pre_item['pre2cur_cts'] = tracking  # ct in the ct int in original image plan
      pre_ret.append(pre_item)

    preds = []
    # number detections #
    for j in range(len(dets['scores'][i])):

      if dets['scores'][i][j] < filter_by_scores:
        # because dets['scores'][i] is descending ordered, if dets['scores'][i][j] < filter_by_scores,
        # then dets['scores'][i][j+n] < filter_by_scores , so we can safely "break" here.
        break

      # print("I am here.", filter_by_scores)
      item = {}
      item['score'] = dets['scores'][i][j]
      item['class'] = int(dets['clses'][i][j]) + 1

      item['ct'] = opt.down_ratio*dets['cts'][i][j].clone()
      item['ct'][0] -= dws[i]
      item['ct'][1] -= dhs[i]
      item['ct'] /= ratios[i]

      if 'bboxes' in dets:
        #xyxy
        bbox = opt.down_ratio*dets['bboxes'][i][j]
        if opt.clip:
          bbox[0::2] = torch.clamp(bbox[0::2], min=0, max=opt.input_w-1)
          bbox[1::2] = torch.clamp(bbox[1::2], min=0, max=opt.input_h-1)
        bbox[0::2] -= dws[i]
        bbox[1::2] -= dhs[i]
        bbox /= ratios[i]
        item['bbox'] = bbox

      preds.append(item)

    ret.append(preds)
  
  return ret, pre_ret