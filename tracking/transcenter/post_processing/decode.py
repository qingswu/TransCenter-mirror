##
## TransCenter: Transformers with Dense Queries for Multiple-Object Tracking
## Copyright Inria
## Year 2021
## Contact : yihong.xu@inria.fr
##
## The software TransCenter is provided "as is", for reproducibility purposes only.
## The user is not authorized to distribute the software TransCenter, modified or not.
## The user expressly undertakes not to remove, or modify, in any manner, the intellectual property notice attached to the software TransCenter.
## code modified from
## (1) Deformable-DETR, https://github.com/fundamentalvision/Deformable-DETR, distributed under Apache License 2.0 2020 fundamentalvision.
## (2) tracking_wo_bnw, https://github.com/phil-bergmann/tracking_wo_bnw, distributed under GNU General Public License v3.0 2020 Philipp Bergmann, Tim Meinhardt.
## (3) CenterTrack, https://github.com/xingyizhou/CenterTrack, distributed under MIT Licence 2020 Xingyi Zhou.
## (4) DCNv2, https://github.com/CharlesShang/DCNv2, distributed under BSD 3-Clause 2019 Charles Shang.
## (5) correlation_package, https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package, Apache License, Version 2.0 2017 NVIDIA CORPORATION.
## (6) pytorch-liteflownet, https://github.com/sniklaus/pytorch-liteflownet, GNU General Public License v3.0 Simon Niklaus.
## (7) LiteFlowNet, https://github.com/twhui/LiteFlowNet, Copyright (c) 2018 Tak-Wai Hui.
## Below you can find the License associated to this LiteFlowNet software:
## 
##     This software and associated documentation files (the "Software"), and the research paper
##     (LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation) including
##     but not limited to the figures, and tables (the "Paper") are provided for academic research purposes
##     only and without any warranty. Any commercial use requires my consent. When using any parts
##     of the Software or the Paper in your work, please cite the following paper:
## 
##     @InProceedings{hui18liteflownet,
##     author = {Tak-Wai Hui and Xiaoou Tang and Chen Change Loy},
##     title = {LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation},
##     booktitle  = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
##     year = {2018},
##     pages = {8981--8989},
##     }
## 
##     The above copyright notice and this permission notice shall be included in all copies or
##     substantial portions of the Software.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat
from .utils import _nms, _topk,  _topk_channel
from util.image import gaussian_radius, draw_umich_gaussian
import math
import numpy as np

def _update_kps_with_hm(
  kps, output, batch, num_joints, K, bboxes=None, scores=None):
  if 'hm_hp' in output:
    hm_hp = output['hm_hp']
    hm_hp = _nms(hm_hp)
    thresh = 0.2
    kps = kps.view(batch, K, num_joints, 2).permute(
        0, 2, 1, 3).contiguous() # b x J x K x 2
    reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
    hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K
    if 'hp_offset' in output or 'reg' in output:
        hp_offset = output['hp_offset'] if 'hp_offset' in output \
                    else output['reg']
        hp_offset = _tranpose_and_gather_feat(
            hp_offset, hm_inds.view(batch, -1))
        hp_offset = hp_offset.view(batch, num_joints, K, 2)
        hm_xs = hm_xs + hp_offset[:, :, :, 0]
        hm_ys = hm_ys + hp_offset[:, :, :, 1]
    else:
        hm_xs = hm_xs + 0.5
        hm_ys = hm_ys + 0.5
    
    mask = (hm_score > thresh).float()
    hm_score = (1 - mask) * -1 + mask * hm_score
    hm_ys = (1 - mask) * (-10000) + mask * hm_ys
    hm_xs = (1 - mask) * (-10000) + mask * hm_xs
    hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
        2).expand(batch, num_joints, K, K, 2)
    dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
    min_dist, min_ind = dist.min(dim=3) # b x J x K
    hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
    min_dist = min_dist.unsqueeze(-1)
    min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
        batch, num_joints, K, 1, 2)
    hm_kps = hm_kps.gather(3, min_ind)
    hm_kps = hm_kps.view(batch, num_joints, K, 2)        
    mask = (hm_score < thresh)
    
    if bboxes is not None:
      l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
              (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + mask
    else:
      l = kps[:, :, :, 0:1].min(dim=1, keepdim=True)[0]
      t = kps[:, :, :, 1:2].min(dim=1, keepdim=True)[0]
      r = kps[:, :, :, 0:1].max(dim=1, keepdim=True)[0]
      b = kps[:, :, :, 1:2].max(dim=1, keepdim=True)[0]
      margin = 0.25
      l = l - (r - l) * margin
      r = r + (r - l) * margin
      t = t - (b - t) * margin
      b = b + (b - t) * margin
      mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
              (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + mask
      # sc = (kps[:, :, :, :].max(dim=1, keepdim=True) - kps[:, :, :, :].min(dim=1))
    # mask = mask + (min_dist > 10)
    mask = (mask > 0).float()
    kps_score = (1 - mask) * hm_score + mask * \
      scores.unsqueeze(-1).expand(batch, num_joints, K, 1) # bJK1
    kps_score = scores * kps_score.mean(dim=1).view(batch, K)
    # kps_score[scores < 0.1] = 0
    mask = mask.expand(batch, num_joints, K, 2)
    kps = (1 - mask) * hm_kps + mask * kps
    kps = kps.permute(0, 2, 1, 3).contiguous().view(
        batch, K, num_joints * 2)
    return kps, kps_score
  else:
    return kps, kps


def generic_decode(output, K=100, opt=None):
  # print("output['wh']", output['wh'].shape)
  # print("output['reg']", output['reg'].shape)
  # print("output['hm']", output['hm'].shape)

  if not ('hm' in output):
    return {}

  # if not opt.tracking:
  #   output['tracking'] = 0
  
  heat = output['hm']
  batch, cat, height, width = heat.size()

  heat = _nms(heat)
  scores, inds, clses, ys0, xs0 = _topk(heat, K=K)

  clses  = clses.view(batch, K)
  scores = scores.view(batch, K)
  bboxes = None
  cts = torch.cat([xs0.unsqueeze(2), ys0.unsqueeze(2)], dim=2)
  ret = {'scores': scores, 'clses': clses.float(), 
         'xs': xs0, 'ys': ys0, 'cts': cts}

  if 'reg' in output:
    reg = output['reg']
    reg = _tranpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs0.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys0.view(batch, K, 1) + reg[:, :, 1:2]
  else:
    xs = xs0.view(batch, K, 1) + 0.5
    ys = ys0.view(batch, K, 1) + 0.5

  # xyh center_offset #
  if 'center_offset' in output:
    # print("I am here!")
    center_offset = output['center_offset']

    center_offset = _tranpose_and_gather_feat(center_offset, inds)  # B x K x (F)
    # wh = wh.view(batch, K, -1)
    center_offset = center_offset.view(batch, K, 2)

    xs = xs + center_offset[:, :, 0:1]
    ys = ys + center_offset[:, :, 1:2]

  if 'wh' in output:
    wh = output['wh']

    wh = _tranpose_and_gather_feat(wh, inds)  # B x K x (F)
    # wh = wh.view(batch, K, -1)
    wh = wh.view(batch, K, 2)
    wh[wh < 0] = 0
    if wh.size(2) == 2 * cat: # cat spec
      wh = wh.view(batch, K, -1, 2)
      cats = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2)
      wh = wh.gather(2, cats.long()).squeeze(2) # B x K x 2
    else:
      pass
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    ret['bboxes'] = bboxes
    # print('ret bbox', ret['bboxes'])
  #
  # if 'ltrb' in output:
  #   # print()
  #   # print('ltrb')
  #   ltrb = output['ltrb']
  #   ltrb = _tranpose_and_gather_feat(ltrb, inds) # B x K x 4
  #   ltrb = ltrb.view(batch, K, 4)
  #   bboxes = torch.cat([xs0.view(batch, K, 1) + ltrb[..., 0:1],
  #                       ys0.view(batch, K, 1) + ltrb[..., 1:2],
  #                       xs0.view(batch, K, 1) + ltrb[..., 2:3],
  #                       ys0.view(batch, K, 1) + ltrb[..., 3:4]], dim=2)
  #   ret['bboxes'] = bboxes

  if 'tracking' in output:
    regression_heads = ['tracking']
    for head in regression_heads:
      if head in output:
        ret[head] = _tranpose_and_gather_feat(
          output[head], inds).view(batch, K, -1)

  #
  # if 'ltrb_amodal' in output:
  #   # print('ltrb_amodal')
  #   ltrb_amodal = output['ltrb_amodal']
  #   ltrb_amodal = _tranpose_and_gather_feat(ltrb_amodal, inds) # B x K x 4
  #   ltrb_amodal = ltrb_amodal.view(batch, K, 4)
  #   bboxes_amodal = torch.cat([xs0.view(batch, K, 1) + ltrb_amodal[..., 0:1],
  #                         ys0.view(batch, K, 1) + ltrb_amodal[..., 1:2],
  #                         xs0.view(batch, K, 1) + ltrb_amodal[..., 2:3],
  #                         ys0.view(batch, K, 1) + ltrb_amodal[..., 3:4]], dim=2)
  #   ret['bboxes_amodal'] = bboxes_amodal
  #   ret['bboxes'] = bboxes_amodal
  #
  # if 'hps' in output:
  #   kps = output['hps']
  #   num_joints = kps.shape[1] // 2
  #   kps = _tranpose_and_gather_feat(kps, inds)
  #   kps = kps.view(batch, K, num_joints * 2)
  #   kps[..., ::2] += xs0.view(batch, K, 1).expand(batch, K, num_joints)
  #   kps[..., 1::2] += ys0.view(batch, K, 1).expand(batch, K, num_joints)
  #   kps, kps_score = _update_kps_with_hm(
  #     kps, output, batch, num_joints, K, bboxes, scores)
  #   ret['hps'] = kps
  #   ret['kps_score'] = kps_score

  if 'pre_inds' in output and output['pre_inds'] is not None:
    pre_inds = output['pre_inds'] # B x pre_K
    pre_K = pre_inds.shape[1]
    pre_ys = (pre_inds / width).int().float()
    pre_xs = (pre_inds % width).int().float()

    ret['pre_cts'] = torch.cat(
      [pre_xs.unsqueeze(2), pre_ys.unsqueeze(2)], dim=2)
  
  return ret