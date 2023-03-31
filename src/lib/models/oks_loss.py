import numpy as np
import torch
import torch.nn as nn
from .utils import _transpose_and_gather_feat

def off_to_pose(output, target, inds):
  '''
  output is predicted offset
  '''

  batch, cat, height, width = output.size()
  num_kps = cat // 2
  ys = (inds / width).int().float()
  xs = (inds % width).int().float()
  ct_coord = torch.stack([xs,ys], dim=-1).repeat(1,1,num_kps)
  pred_offset = _transpose_and_gather_feat(output, inds)

  pred_pose = ct_coord + pred_offset
  target = ct_coord + target

  return pred_pose, target
  

def oks_overlaps(kpt_preds, kpt_gts, kpt_valids, kpt_areas, ind, sigmas):
    ## 第一个维度为有效人体数目
    kpt_preds, kpt_gts = off_to_pose(kpt_preds, kpt_gts, ind)
    
    assert kpt_preds.shape == kpt_gts.shape
    bs, max_num, _ = kpt_preds.shape
    kpt_valids = kpt_valids.reshape(bs, max_num, kpt_valids.size(-1)// 2, 2)[:,:,:,0]
    valid_inst = kpt_valids.sum(-1).reshape(bs * max_num) > 0
    valid_preds = kpt_preds.reshape(bs * max_num, kpt_preds.size(-1))[valid_inst]
    valid_gts = kpt_gts.reshape(bs * max_num, kpt_gts.size(-1))[valid_inst]
    kpt_areas = kpt_areas.reshape(bs * max_num)[valid_inst]
    kpt_valids = kpt_valids.reshape(bs * max_num, kpt_valids.size(-1))[valid_inst]
    assert valid_preds.shape == valid_gts.shape
    
    kpt_preds, kpt_gts = valid_preds, valid_gts

    sigmas = kpt_preds.new_tensor(sigmas)
    variances = (sigmas * 2)**2
    
    # 
    kpt_preds = kpt_preds.reshape(-1, kpt_preds.size(-1) // 2, 2)
    kpt_gts = kpt_gts.reshape(-1, kpt_gts.size(-1) // 2, 2)

    squared_distance = (kpt_preds[:, :, 0] - kpt_gts[:, :, 0]) ** 2 + \
        (kpt_preds[:, :, 1] - kpt_gts[:, :, 1]) ** 2
    assert (kpt_valids.sum(-1) > 0).all()
    squared_distance0 = squared_distance / (
        kpt_areas[:, None] * variances[None, :] * 2)
    squared_distance1 = torch.exp(-squared_distance0)
    # import pudb;pudb.set_trace()
    squared_distance1 = squared_distance1 * kpt_valids.float()
    oks = squared_distance1.sum(dim=1) / kpt_valids.float().sum(dim=1)

    return oks


def oks_loss(pred,
             target,
             ind,
             valid=None,
             area=None,
             linear=False,
             sigmas=None,
             eps=1e-6):
    """Oks loss.

    Computing the oks loss between a set of predicted poses and target poses.
    The loss is calculated as negative log of oks.

    Args:
        pred (torch.Tensor): Predicted poses of format (x1, y1, x2, y2, ...),
            shape (n, K*2).
        target (torch.Tensor): Corresponding gt poses, shape (n, K*2).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        eps (float): Eps to avoid log(0).

    Returns:
        torch.Tensor: Loss tensor.
    """
    oks = oks_overlaps(pred, target, valid, area, ind, sigmas).clamp(min=eps)
    if linear:
        loss = 1 - oks
    else:
        loss = -oks.log()
    return loss


class OKSLoss(nn.Module):
    """OKSLoss.

    Computing the oks loss between a set of predicted poses and target poses.

    Args:
        linear (bool): If True, use linear scale of loss instead of log scale.
            Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 linear=False,
                 num_keypoints=17,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(OKSLoss, self).__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        if num_keypoints == 17:
            self.sigmas = np.array([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
                1.07, .87, .87, .89, .89
            ], dtype=np.float32) / 10.0
        elif num_keypoints == 14:
            self.sigmas = np.array([
                .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89,
                .79, .79
            ]) / 10.0
        else:
            raise ValueError(f'Unsupported keypoints number {num_keypoints}')

    def forward(self,
                pred,
                target,
                valid,
                area,
                ind,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction. (bs, max_num, 34）
            target (torch.Tensor): The learning target of the prediction. (bs, max_num, 34）
            valid (torch.Tensor): The visible flag of the target pose. (bs, max_num, 34）
            area (torch.Tensor): The area of the target pose. (bs, max_num）
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """ 
        
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * oks_loss(
            pred,
            target,
            ind,
            valid=valid,
            area=area,
            linear=self.linear,
            sigmas=self.sigmas,
            eps=self.eps,)
            # reduction=reduction,
            # avg_factor=avg_factor,
            # **kwargs)
        return loss