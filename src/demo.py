from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import init_paths

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)
  # import pudb;pudb.set_trace()
  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #VideoWriter_fourcc为视频编解码器
    size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter('res.mp4', fourcc, 30.0, size)
    detector.pause = False
    if cam.isOpened():
      while True:
          a, img = cam.read()
          if not a:  ##########
            return    ###########
          # cv2.imshow('input', img)
          ret = detector.run(img)
          time_str = ''
          for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
          print(time_str)
          writer.write(ret['vis_img'])
          # if cv2.waitKey(1) == 27:
          #     return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      ret = detector.run(image_name,image_name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
