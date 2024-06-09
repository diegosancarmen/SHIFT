# help functions are adapted from original mean teacher network
# https://github.com/CuriousAI/mean-teacher/tree/master/pytorch

import numpy as np
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

class OldWeightEMA (object):
    """
    Exponential moving average weight optimizer for mean teacher model
    """
    def __init__(self, target_net, source_net, alpha=0.999):
        self.target_params = list(target_net.parameters())
        self.source_params = list(source_net.parameters())
        self.alpha = alpha

        for p, src_p in zip(self.target_params, self.source_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.target_params, self.source_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)

def softargmax(segm, alpha=65000.0):
    """
    Replacement for torch.argmax without breaking the gradient.
    """
    B, N, _, _ = segm.size()
    segm = segm * alpha
    segm = segm - segm.max(dim=1, keepdim=True)[0]
    softmax = torch.exp(segm) / torch.sum(torch.exp(segm), dim=1, keepdim=True) #F.softmax(segm, dim=1)
    indices_kernel = ((torch.arange(1,N+1).unsqueeze(0)).unsqueeze(-1)).unsqueeze(-1).cuda()
    conv = softmax * indices_kernel
    indices = conv.sum(1) - 1

    return indices

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    # assert 0 <= current <= rampdown_length
    current = np.clip(current, 0.0, rampdown_length)
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def rev_sigmoid(progress):

    progress = np.clip(progress, 0, 1)
    return float(1. / (1 + np.exp(10 * progress - 5)))

def sigmoid(progress):

    progress = np.clip(progress, 0, 1)
    return float(1. / (1 + np.exp(5 - 10 * progress)))

def get_max_preds_torch(batch_heatmaps):

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    width = batch_heatmaps.size(3)
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = torch.argmax(heatmaps_reshaped, 2)
    maxvals = torch.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

    pred_mask = (maxvals > 0.0).repeat(1, 1, 2)
    pred_mask = pred_mask.float()

    preds *= pred_mask
    return preds, maxvals

def rectify(hm, sigma): # b, c, h, w -> b, c, h, w
    b, c, h, w = hm.size()
    rec_hm = torch.zeros_like(hm)
    pred_coord, pred_val = get_max_preds_torch(hm) # b, c, 2
    tmp_size = 3 * sigma
    for b in range(rec_hm.size(0)):
        for c in range(rec_hm.size(1)):
            mu_x = pred_coord[b, c, 0]
            mu_y = pred_coord[b, c, 1]
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if mu_x >= h or mu_y >= w or mu_x < 0 or mu_y < 0:
                continue

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = torch.arange(0, size, 1).float()
            y = x.unsqueeze(1)
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], h)
            img_y = max(0, ul[1]), min(br[1], w)

            rec_hm[b][c][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return rec_hm

def generate_prior_map(prior, preds, gamma=2, sigma=2, epsilon=-10e10, v3=False): # prior: {mean: (k, k), std: (k, k)}, preds: (b, k, h, w) -> returns prior_map: (b, k, h, w)
    # for the prediction in each channel, generate the estimation of the rest channels (assign a weight for each according to confidence and std?) with shape of (k, k, h, w)
    # ensemble all the estimation and form a prior map, which should be a multiplier for the original prediction map.

    prior_mean = prior['mean'].cuda()
    prior_std = prior['std'].cuda()
    B, K, H, W = preds.size()
    pred_coord, pred_val = get_max_preds_torch(preds) # B, K, (1), 2 ; B, K, 1
    pred_coord = pred_coord.view(B,K,1,2,1,1)

    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,1,H,W).repeat(B,K,1,1,1)
    yy = yy.view(1,1,1,H,W).repeat(B,K,1,1,1)
    grid = torch.cat((xx,yy),2).float().cuda().view(B,1,K,2,H,W) # B, (1), K, 2, H, W

    dist = torch.norm(grid - pred_coord, dim=3) # B, K, K, H, W
    dist -= prior_mean.view(1,K,K,1,1) # B, K, K, H, W
    targets = torch.exp(-(dist**2) / (2 * sigma ** 2)) # B, K, K, H, W

    if v3:
        var_table = (1 / (1 + prior_std)).view(1,K,K) # 1, K, K
        conf_table = pred_val.view(B,K,1) # B, K, 1
        final_weight = var_table * conf_table # B, K, K
        # final_weight = F.softmax(final_weight, dim=1) # B, K, K, 1
        targets = torch.sum(final_weight.view(B, K, K, 1, 1) * targets, dim=1)

    else:
        temp_std = -prior_std / gamma
        temp_std.fill_diagonal_(epsilon)
        weights = F.softmax(temp_std, dim=0) # K, K

        targets = torch.sum(weights.view(1, K, K, 1, 1) * targets, dim=1)

    return targets

def heatmap_to_keypoints(heatmaps):
    """
    Convert heatmaps to keypoint coordinates.

    Args:
        heatmaps (torch.Tensor): Heatmaps from the student model (B, num_keypoints, H, W)
    
    Returns:
        torch.Tensor: Keypoint coordinates (B, num_keypoints, 2)
    """
    B, num_keypoints, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.view(B, num_keypoints, -1)
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)
    keypoints_y = (max_indices // W).float()
    keypoints_x = (max_indices % W).float()
    keypoints = torch.stack((keypoints_x, keypoints_y), dim=2)
    return keypoints

def calculate_s_max(segmentation_maps):
    batch_size = segmentation_maps.size(0)
    cardinalities = [torch.sum(segmentation_maps[i]) for i in range(batch_size)]
    s_max = max(cardinalities)
    return s_max

def plot_lambda_vs_epochs(lambda_c_values, epochs, save_path):
    # Create directory if it does not exist
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    save_path = re.sub(r'/\d+/', '/', save_path)
    # Plot the lambda_c vs. epochs curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lambda_c_values, label='lambda')
    plt.xlabel('Adaptation Epochs')
    plt.ylabel('lambda')
    plt.title('Exponential Decay of Lambda')
    plt.legend()
    plt.grid(True)

    # Save the plot to the specified directory
    #save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()