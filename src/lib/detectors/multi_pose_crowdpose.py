from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
  from external.nms import soft_nms_39
except:
  print('NMS not imported! If you need it,' 
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import multi_pose_decode_wodet, multi_pose_decode_wodet_vis, multi_pose_decode
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_wodet_post_process,multi_pose_wodet_post_process_vis, multi_pose_crowdpose_post_process_vis
from utils.debugger import Debugger

from .base_detector import BaseDetector

class MultiPoseDetector_crowdpose(BaseDetector):
  def __init__(self, opt):
    super(MultiPoseDetector_crowdpose, self).__init__(opt)
    self.flip_idx = opt.flip_idx

  def process(self, images, return_time=False):
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images)[-1]
      output['hm'] = output['hm'].sigmoid_()
      if self.opt.hm_hp and not self.opt.mse_loss:
        output['hm_hp'] = output['hm_hp'].sigmoid_()

      reg = output['reg'] if self.opt.reg_offset else None
      # reg = None
      hm_hp = output['hm_hp'] if self.opt.hm_hp else None
      hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
      # hp_offset = None
      torch.cuda.synchronize()
      forward_time = time.time()
      
      if self.opt.flip_test:
        output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
        output['hps'] = output['hps'][0:1]   
        hm_hp = None #hm_hp[0:1]
        # output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
        # output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
        # output['hps'] = (output['hps'][0:1] + 
        #   flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
        # hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
        #         if hm_hp is not None else None
        # reg = reg[0:1] if reg is not None else None
        # hp_offset = hp_offset[0:1] if hp_offset is not None else None 
      
      # dets = multi_pose_decode(
      #     output['hm'],output['wh'] ,output['hps'],
      #     reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
      dets = multi_pose_decode_wodet(
         output['hm'], output['hps'],
         reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)
      # dets = multi_pose_decode_wodet_vis(
      #      output['hm'], output['hps'],output['ap'],
      #      reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)


    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    # dets = multi_pose_wodet_post_process(
    #   dets.copy(), [meta['c']], [meta['s']],
    #   meta['out_height'], meta['out_width'])
    dets,adapt_pts = multi_pose_crowdpose_post_process_vis(
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width']) 

    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 29)
      # import pdb; pdb.set_trace()
      #dets[0][j][:, :4] /= scale
      dets[0][j][:, 1:] /= (scale*meta['sf'])
      adapt_pts /= (scale*meta['sf'])
    
    return dets[0] ,adapt_pts

  def kps_to_bbox(self, det, mode='max'):
    assert det.shape == (20,29)
    pts = det[:,1:].reshape(20,14,2)
    if mode == 'max':
      tl = np.min(pts,axis=1)
      rd = np.max(pts,axis=1)
      bbox = np.concatenate([tl,rd],axis=1)
      assert bbox.shape == (20,4)
    det_ = np.concatenate([bbox,det,det[:,:6]],axis=1)
    return det_


  def merge_outputs(self, detections):
    # import pudb; pudb.set_trace()
    results = {}    
    if self.opt.nms or len(self.opt.test_scales) > 1:
      results[1] = np.concatenate(
        [self.kps_to_bbox(detection[1]) for detection in detections], axis=0).astype(np.float32)
      soft_nms_39(results[1], Nt=0.5, method=2)
    else:
      results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)
    results[1] = results[1].tolist()
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    dets = dets.detach().cpu().numpy().copy()
    dets[:, :, :4] *= self.opt.down_ratio
    dets[:, :, 5:39] *= self.opt.down_ratio
    img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(((
      img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    debugger.add_blend_img(img, pred, 'pred_hm')
    if self.opt.hm_hp:
      pred = debugger.gen_colormap_hp( 
        output['hm_hp'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hmhp')
  
  def show_results(self, debugger, image, results, adapt_pts, image_path):
    debugger.add_img(image, img_id='multi_pose')
    for idx,bbox in enumerate(results[1]):
      if bbox[0] > self.opt.vis_thresh:
        # import pudb; pudb.set_trace()
        # debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose')
        debugger.add_coco_hp_with_ap(bbox[1:35], adapt_pts[idx], image_path,img_id='multi_pose') 
    # debugger.show_all_imgs(pause=self.pause)


