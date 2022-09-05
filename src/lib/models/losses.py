# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _transpose_and_gather_feat
import torch.nn.functional as F
# from torchvision.ops.boxes import box_area



def bboxes_giou(boxes1,boxes2):
    '''
    cal GIOU of two boxes or batch boxes
    such as: (1)
            boxes1 = np.asarray([[0,0,5,5],[0,0,10,10],[15,15,25,25]])
            boxes2 = np.asarray([[5,5,10,10]])
            and res is [-0.49999988  0.25       -0.68749988]
            (2)
            boxes1 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
            boxes2 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
            and res is [1. 1. 1.]
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''
    # import pudb;pudb.set_trace()
    # cal the box's area of boxes1 and boxess
    boxes1Area = (boxes1[...,2]-boxes1[...,0])*(boxes1[...,3]-boxes1[...,1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # ===========cal IOU=============#
    #cal Intersection
    left_up = torch.max(boxes1[...,:2],boxes2[...,:2])
    right_down = torch.min(boxes1[...,2:],boxes2[...,2:])

    inter_section = torch.max(right_down-left_up, torch.zeros_like(right_down-left_up))
    inter_area = inter_section[...,0] * inter_section[...,1]
    union_area = boxes1Area + boxes2Area - inter_area
    ious = torch.max(1.0 * inter_area/(union_area + 1e-5))

    # ===========cal enclose area for GIOU=============#
    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(right_down-left_up))
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # cal GIOU
    gious = ious - 1.0 * (enclose_area - union_area) / (enclose_area + 1e-5)

    return gious


# def box_iou(boxes1, boxes2):
#     area1 = box_area(boxes1)
#     area2 = box_area(boxes2)

#     lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
#     rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

#     wh = (rb - lt).clamp(min=0)  # [N,M,2]
#     inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

#     union = area1[:, None] + area2 - inter

#     iou = inter / union
#     return iou, union


# def generalized_box_iou(boxes1, boxes2):
#     """
#     Generalized IoU from https://giou.stanford.edu/
#     The boxes should be in [x0, y0, x1, y1] format
#     Returns a [N, M] pairwise matrix, where N = len(boxes1)
#     and M = len(boxes2)
#     """
#     # degenerate boxes gives inf / nan results
#     # so do an early check
#     boxes1 = boxes1.reshape(-1,4)
#     boxes2 = boxes2.reshape(-1,4)
#     assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
#     assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
#     iou, union = box_iou(boxes1, boxes2)

#     lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
#     rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

#     wh = (rb - lt).clamp(min=0)  # [N,M,2]
#     area = wh[:, :, 0] * wh[:, :, 1]

#     return iou - (area - union) / area

def off_to_pose(output, inds):
  '''
  output is predicted offset
  '''

  batch, cat, height, width = output.size()
  num_kps = cat // 2
  ys = (inds / width).int().float()
  xs = (inds % width).int().float()
  ct_coord = torch.stack([xs,ys], dim=-1)
  pred_offset = _transpose_and_gather_feat(output, inds)
  pred_pose = ct_coord.repeat(1,1,num_kps) + pred_offset

  return pred_pose

def kps_to_pseudo(pose, is_valid):
  '''
  kps: b * num_person * 34 
  mask: b * num_person * 34
  '''
  # filter_pose = pose * is_valid
  is_valid = is_valid.reshape(is_valid.shape[0], is_valid.shape[1], 17, 2)
  filter_pose = pose.reshape(pose.shape[0], pose.shape[1], 17, 2)
  
  filter_pose[is_valid == 0] = 1e+5
  tl = torch.min(filter_pose, dim=2)[0]
  filter_pose[filter_pose == 1e+5] = -1e+5

  br = torch.max(filter_pose, dim =2)[0]
  pseudo_box = torch.cat([tl, br], dim = -1)
  
  pseudo_box[pseudo_box==1e+5] = 0
  pseudo_box[pseudo_box==-1e+5] = 0

  return pseudo_box.reshape(pose.shape[0], pose.shape[1], 4)   

  
class Giou(nn.Module):
  def __init__(self):
    super(Giou, self).__init__()
  
  def forward(self, output, kps_mask, inst_mask, ind, gt_pseudo):
    pred_pose = off_to_pose(output, ind)
    kps_mask = kps_mask.float()
    pred_pseudo = kps_to_pseudo(pred_pose, kps_mask)
    # import pudb;pudb.set_trace()
    loss = 1 - bboxes_giou(pred_pseudo, gt_pseudo)
    loss = loss.sum() / (inst_mask.sum() + 1e-4)
    return loss



def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1)
  neg_inds = gt.lt(1)

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()    
    num_pos  = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss_coco(nn.Module):
  def __init__(self, with_bone=False):
    super(RegWeightedL1Loss_coco, self).__init__()
    self.with_bone = with_bone
    self.edges = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], 
                    [5, 7], [7, 9], [6, 8], [8, 10], 
                    [5, 11], [6, 12], 
                    [11, 13], [13, 15], [12, 14], [14, 16]]

    # self.edges = [[13,1], [0,2], [0,13], [13,12], [2,4], [1,3],
    #               [3,5], [6,7], [6,8], [8,10], [7,9], [9,11]]
    self.num_edges = len(self.edges)
  
  def forward(self, output, mask, ind, target):
#     import pudb;pudb.set_ts
    bs, num_persons = target.shape[:2]
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    
    if self.with_bone:
        start, end = zip(*self.edges)
        mask = mask.view(bs, num_persons, 17, 2)
        start_mask = mask[:, :,start] 
        end_mask = mask[:, :, end]
        edge_mask = (start_mask * end_mask).view(bs, num_persons,self.num_edges*2)
        target = target.view(bs, num_persons, 17, 2)
        edge_target = (target[:, :, end] - target[:, :, start]).view(bs, num_persons,self.num_edges*2)
        pred = pred.view(bs, num_persons, 17, 2)
        edge_pred = (pred[:, :, end] - pred[:, :, start]).view(bs, num_persons,self.num_edges*2)
        loss_edge = F.l1_loss(edge_pred * edge_mask, edge_target * edge_mask, size_average=False)
        loss_edge = loss_edge/(edge_mask.sum() + 1e-4)
        loss = (loss + loss_edge)/2
    return loss

class RegWeightedL1Loss_crowdpose(nn.Module):
  def __init__(self, with_bone=True):
    super(RegWeightedL1Loss_crowdpose, self).__init__()
    self.with_bone = with_bone
  

    self.edges = [[13,1], [0,2], [0,13], [13,12], [2,4], [1,3],
                  [3,5], [6,7], [6,8], [8,10], [7,9], [9,11]]
    self.num_edges = len(self.edges)
  
  def forward(self, output, mask, ind, target):
#     import pudb;pudb.set_ts
    bs, num_persons = target.shape[:2]
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    
    if self.with_bone:
        start, end = zip(*self.edges)
        mask = mask.view(bs, num_persons, 14, 2)
        start_mask = mask[:, :,start] 
        end_mask = mask[:, :, end]
        edge_mask = (start_mask * end_mask).view(bs, num_persons,self.num_edges*2)
        target = target.view(bs, num_persons, 14, 2)
        edge_target = (target[:, :, end] - target[:, :, start]).view(bs, num_persons,self.num_edges*2)
        pred = pred.view(bs, num_persons, 14, 2)
        edge_pred = (pred[:, :, end] - pred[:, :, start]).view(bs, num_persons,self.num_edges*2)
        loss_edge = F.l1_loss(edge_pred * edge_mask, edge_target * edge_mask, size_average=False)
        loss_edge = loss_edge/(edge_mask.sum() + 1e-4)
        loss = (loss + loss_edge)/2
    return loss


class L1Loss(nn.Module):
  def __init__(self):
    super(L1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _transpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res
