from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger


class BaseDetector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    self.model = create_model(opt.arch, opt.heads, opt.head_conv, is_train=False)
    self.model = load_model(self.model, opt.load_model)
    self.model = self.model.to(opt.device)
    self.model.eval()
    
    params_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    print('Params: %.2fM' % (params_num / 1e6))

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True
  
  def keep_aspect_resize(self, image,scale, input_res):
    height, width = image.shape[0:2]
    # print('og',height,width)
    if height > width:
      sf = (input_res/width)
      new_height = height * sf
      new_width =  input_res
    else:
      sf = (input_res/height)
      new_width = width * sf
      new_height =  input_res
    new_height = int(new_height * scale)
    new_width  = int(new_width * scale)
    return new_height,new_width,sf



  def pre_process(self, image, scale, meta=None):
    if self.opt.fix_res:
      #print('################################fix_res########################')
      sf = 1.0
      height, width = image.shape[0:2]
      new_height = int(height * scale)
      new_width  = int(width * scale)
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      new_height,new_width,sf = self.keep_aspect_resize(image, scale, self.opt.input_res)
      inp_height = (new_height | self.opt.pad) + 1
      inp_width = (new_width | self.opt.pad) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)
    

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    # cv2.imwrite('/data/yabo.xiao/pad_vis/'+ img_name,inp_image)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
    # print(inp_image.shape)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 'sf':sf, 
            'out_height': inp_height // self.opt.down_ratio, 
            'out_width': inp_width // self.opt.down_ratio}
    return images, meta

  def process(self, images, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results):
   raise NotImplementedError

  def run(self, image_or_path_or_tensor, meta=None):
    
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                        theme=self.opt.debugger_theme)
    start_time = time.time()
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''): 
      image = cv2.imread(image_or_path_or_tensor)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True 
    
    loaded_time = time.time()
    load_time += (loaded_time - start_time)

    detections = []
    for scale in self.scales:
      scale_start_time = time.time()
      if not pre_processed:
        images, meta = self.pre_process(image, scale, meta)
      else:
        # 
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
      images = images.to(self.opt.device)
      torch.cuda.synchronize()
      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time
      
      output, dets, forward_time = self.process(images, return_time=True)

      torch.cuda.synchronize()
      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time
      
      if self.opt.debug >= 2:
        self.debug(debugger, images, dets, output, scale)
      
      dets, adapt_pts = self.post_process(dets, meta, scale)
      torch.cuda.synchronize()
      post_process_time = time.time()
      post_time += post_process_time - decode_time

      detections.append(dets)
    
    results = self.merge_outputs(detections)
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time
    
    import pudb;pudb.set_trace()
    if self.opt.debug >= 1:
      save_path = self.opt.output_path + 'pred'
      import os
      if os.path.exists(save_path):
        os.mkdir(save_path)
      self.show_results(debugger, image, results, adapt_pts, save_path, pre_processed_images['img_name'])
    
    return {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}