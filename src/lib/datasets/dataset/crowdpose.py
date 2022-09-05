from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import crowdposetools.coco as coco
# from pycocotools.cocoeval import COCOeval
from crowdposetools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class CrowdPose(data.Dataset):
  num_classes = 1
  num_joints = 14
  default_resolution = [512, 512]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  
  flip_idx = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], 
              [10, 11]]
  def __init__(self, opt, split):
    super(CrowdPose, self).__init__()
    
    self.edges = [[13,1], [0,2], [0,13], [13,12], [2,4], [1,3],
                  [3,5], [6,7], [6,8], [8,10], [7,9], [9,11]]
    self.data_format ='zip'
    self.acc_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    self.data_dir = os.path.join(opt.data_dir, 'crowdpose')
    self.img_dir = os.path.join(self.data_dir, 'images')
    if split == 'test':
      self.annot_path = os.path.join(
          self.data_dir, 'json', 
          'crowdpose_{}.json'.format(split))
    else:
      self.annot_path = os.path.join(
        self.data_dir, 'json', 
        'crowdpose_{}.json'.format(split))
    self.max_objs = 32
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32) 
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt

    print('==> initializing crowdpose {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    image_ids = self.coco.getImgIds()

    if split == 'train':
      self.images = []
      for img_id in image_ids:
        idxs = self.coco.getAnnIds(imgIds=[img_id])
        if len(idxs) > 0:
          self.images.append(img_id)
    else:
      self.images = image_ids
    self.num_samples = len(self.images)
    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def kps_to_bbox(self, kps, mode='max'):
    assert kps.shape == (20,35)
    pts = det[:,1:].reshape(20,17,2)
    if mode == 'max':
      tl = np.min(pts,axis=1)
      rd = np.max(pts,axis=1)
      bbox = np.concatenate([tl,rd],axis=1)
      assert bbox.shape == (20,4)
    det_ = np.concatenate([bbox,det],axis=1)
    return det_


  def convert_eval_format(self, all_bboxes, MS=False):
    # import pudb; pudb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = 1
        for dets in all_bboxes[image_id][cls_ind]:
          if MS==False:
         
            score = dets[0] #* np.log(area)
            
            keypoints = np.concatenate([
              np.array(dets[1:29], dtype=np.float32).reshape(-1, 2), 
              np.ones((14, 1), dtype=np.float32)], axis=1).reshape(42).tolist()
          else:
            score = dets[4] 
            keypoints = np.concatenate([
              np.array(dets[5:33], dtype=np.float32).reshape(-1, 2), 
              np.ones((14, 1), dtype=np.float32)], axis=1).reshape(42).tolist()
          keypoints  = list(map(self._to_float, keypoints))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              # "bbox": bbox_out,
              "score": float("{:.2f}".format(score)),
              "keypoints": keypoints
          }
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir,MS=False):
    json.dump(self.convert_eval_format(results,MS=MS), 
              open('{}/results.json'.format(save_dir), 'w'))


  def run_eval(self, results, save_dir, MS=False):
    
    self.save_results(results, save_dir, MS=MS)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
   
    stats_names = ['AP', 'Ap .5', 'AP .75', 'AR', 'AR .5',
                       'AR .75', 'AP (easy)', 'AP (medium)', 'AP (hard)']
    
    stats_index = [0, 1, 2, 5, 6, 7, 8, 9, 10]

    info_str = []
    for ind, name in enumerate(stats_names):
        info_str.append((name, coco_eval.stats[stats_index[ind]]))

    return info_str
