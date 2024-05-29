"""
Modified from https://github.com/microsoft/human-pose-estimation.pytorch
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class JointsMSELoss(nn.Module):
    """
    Typical MSE loss for keypoint detection.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Inputs:
        - output (tensor): heatmap predictions
        - target (tensor): heatmap labels
        - target_weight (tensor): whether the keypoint is visible. All keypoint is visible if None. Default: None.

    Shape:
        - output: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - target: :math:`(minibatch, K, H, W)`.
        - target_weight: :math:`(minibatch, K)`.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, K)`.

    """
    def __init__(self, reduction='mean'):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.reduction = reduction

    def forward(self, output, target, target_weight=None):
        B, K, _, _ = output.shape
        heatmaps_pred = output.reshape((B, K, -1))
        heatmaps_gt = target.reshape((B, K, -1))
        loss = self.criterion(heatmaps_pred, heatmaps_gt) * 0.5
        if target_weight is not None:
            loss = loss * target_weight.view((B, K, 1))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss.mean(dim=-1)


class JointsKLLoss(nn.Module):
    """
    KL Divergence for keypoint detection proposed by
    `Regressive Domain Adaptation for Unsupervised Keypoint Detection <https://arxiv.org/abs/2103.06175>`_.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Inputs:
        - output (tensor): heatmap predictions
        - target (tensor): heatmap labels
        - target_weight (tensor): whether the keypoint is visible. All keypoint is visible if None. Default: None.

    Shape:
        - output: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - target: :math:`(minibatch, K, H, W)`.
        - target_weight: :math:`(minibatch, K)`.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, K)`.

    """
    def __init__(self, reduction='mean', epsilon=0.):
        super(JointsKLLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, output, target, target_weight=None):
        B, K, _, _ = output.shape
        heatmaps_pred = output.reshape((B, K, -1))
        heatmaps_pred = F.log_softmax(heatmaps_pred, dim=-1)
        heatmaps_gt = target.reshape((B, K, -1))
        heatmaps_gt = heatmaps_gt + self.epsilon
        heatmaps_gt = heatmaps_gt / heatmaps_gt.sum(dim=-1, keepdims=True)
        loss = self.criterion(heatmaps_pred, heatmaps_gt).sum(dim=-1)
        if target_weight is not None:
            loss = loss * target_weight.view((B, K))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss.mean(dim=-1)

class EntLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(EntLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, threshold=-1):
        n, c, h, w = x.size()
        x = x.reshape((n, c, -1))
        P = F.softmax(x, dim=2) # [B, 18, 4096]
        logP = F.log_softmax(x, dim=2) # [B, 18, 4096]
        PlogP = P * logP               # [B, 18, 4096]
        ent = -1.0 * PlogP.sum(dim=2)  # [B, 18, 1]
        ent = ent / np.log(h * w)        # 8.3177661667: log_e(4096)
        if threshold > 0:
            ent = ent[ent < threshold]
        # compute robust entropy
        if self.reduction == 'mean':
            return ent.mean()
        elif self.reduction == 'none':
            return ent.mean(dim=-1)

class ConsLoss(nn.Module):

    def __init__(self):
        super(ConsLoss, self).__init__()

    def forward(self, stu_out, tea_out, valid_mask=None, tea_mask=None): # b, c, h, w
        diff = stu_out - tea_out # b, c, h, w
        if tea_mask is not None:
            diff *= tea_mask[:, :, None, None] # b, c, h, w
        loss_map = torch.mean((diff) ** 2, dim=1) # b, h, w
        if valid_mask is not None:
            loss_map = loss_map[valid_mask]

        return loss_map.mean()
    
class CurriculumLearningLoss(nn.Module):

    def __init__(self):
        super(CurriculumLearningLoss, self).__init__()

    def forward(self, stu_out, tea_out, visibility_scores, valid_mask=None, tea_mask=None): 
        """
        Compute the curriculum learning loss between student and teacher outputs, weighted by visibility scores.

        Args:
            stu_out (torch.Tensor): Student model output (B, C, H, W)
            tea_out (torch.Tensor): Teacher model output (B, C, H, W)
            visibility_scores (torch.Tensor): Visibility scores for each keypoint (B, C)
            valid_mask (torch.Tensor, optional): Mask to specify valid regions (B, H, W). Defaults to None.
            tea_mask (torch.Tensor, optional): Mask to specify teacher's valid keypoints (B, C). Defaults to None.

        Returns:
            torch.Tensor: Weighted consistency loss
        """
        # Compute the difference between student and teacher outputs
        diff = stu_out - tea_out  # (B, C, H, W)
        
        # Apply teacher mask if provided
        if tea_mask is not None:
            diff *= tea_mask[:, :, None, None]  # (B, C, H, W)
        
        # Compute the mean squared error across the channel dimension
        loss_map = torch.mean(diff ** 2, dim=1)  # (B, H, W)
        
        # Apply valid mask if provided
        if valid_mask is not None:
            loss_map = loss_map[valid_mask]  # (valid_elements,)
        
        # Compute the weighted loss using visibility scores
        # The visibility scores are expected to have the shape (B, C)
        # We need to expand them to match the dimensions of loss_map
        visibility_weights = visibility_scores.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        
        # Expand visibility_weights to match the shape of loss_map
        visibility_weights = visibility_weights.expand_as(stu_out)  # (B, C, H, W)
        
        # Compute the weighted loss map
        weighted_loss_map = loss_map * visibility_weights.mean(dim=1)  # (B, H, W)
        
        # Return the mean of the weighted loss map
        return weighted_loss_map.mean()

# Define the cross-entropy loss for segmentation maps
class SegLoss(nn.Module):
    def __init__(self):
        super(SegLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, pred_segmentation, target_segmentation):
        """
        Compute the cross-entropy loss between predicted and target segmentation maps.

        Args:
            pred_segmentation (torch.Tensor): Predicted segmentation map (B, 256, 256)
            target_segmentation (torch.Tensor): Target segmentation map (B, 256, 256)
        
        Returns:
            torch.Tensor: Cross-entropy loss
        """
        return self.criterion(pred_segmentation, target_segmentation)

class ConsSoftmaxLoss(nn.Module):

    def __init__(self):
        super(ConsSoftmaxLoss, self).__init__()

    def forward(self, stu_out, tea_out, valid_mask=None, tea_mask=None): # b, c, h, w
        B, K, H, W = stu_out.shape

        stu_out = F.softmax(stu_out.reshape((B, K, -1)), dim=-1).reshape(B, K, H, W)
        tea_out = F.softmax(tea_out.reshape((B, K, -1)), dim=-1).reshape(B, K, H, W)

        diff = stu_out - tea_out # b, c, h, w
        if tea_mask is not None:
            diff *= tea_mask[:, :, None, None] # b, c, h, w
        loss_map = torch.mean((diff) ** 2, dim=1) # b, h, w
        if valid_mask is not None:
            loss_map = loss_map[valid_mask]

        return loss_map.mean()

class ConsKLLoss(nn.Module):

    def __init__(self):
        super(ConsKLLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')

    def forward(self, stu_out, tea_out, valid_mask=None, tea_mask=None): # b, c, h, w
        B, K, H, W = stu_out.shape
        stu_out = stu_out.reshape((B, K, -1))
        stu_out = F.log_softmax(stu_out, dim=-1)
        tea_out = tea_out.reshape((B, K, -1))
        tea_out = F.log_softmax(tea_out, dim=-1)
        loss_map = self.criterion(stu_out, tea_out).reshape((B, K, H, W))
        if tea_mask is not None:
            loss_map *= tea_mask[:, :, None, None] # b, c, h, w
        loss_map = torch.mean(loss_map, dim=1) # b, h, w
        if valid_mask is not None:
            loss_map = loss_map[valid_mask]

        return loss_map.mean()


class CoralLoss(nn.Module):

    def __init__(self, coral_downsample, prior=None):
        super(CoralLoss, self).__init__()
        self.coral_downsample = coral_downsample
        self.prior = prior

    def forward(self, src_out, tgt_out): 
        if self.coral_downsample > 1:
            tgt_out = F.interpolate(tgt_out, scale_factor=1/self.coral_downsample, mode='bilinear')

        n, c, h, w = tgt_out.size()
        tgt_out = tgt_out.view(n, -1)

        if self.prior is not None:
            cs = self.prior
        else:
            # source covariance
            if self.coral_downsample > 1:
                src_out = F.interpolate(src_out, scale_factor=1/self.coral_downsample, mode='bilinear')
            src_out = src_out.view(n, -1)
            tmp_s = torch.ones((1, n)).cuda() @ src_out
            cs = (src_out.T @ src_out - (tmp_s.T @ tmp_s) / n) / (n - 1)

        # target covariance
        tmp_t = torch.ones((1, n)).cuda() @ tgt_out
        ct = (tgt_out.T @ tgt_out - (tmp_t.T @ tmp_t) / n) / (n - 1)

        # frobenius norm
        loss = (cs - ct).pow(2).sum().sqrt()
        loss = loss / (4 * (c * h * w)**2)

        return loss
