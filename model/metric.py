# -*- coding: utf-8 -*-
__author__ = 'zookeeper'
import numpy as np
import torch.nn.functional as F
import torch

def psnr(input, target, max_val=1.0):
    mse = np.square(input - target).mean()
    return 10 * np.log10(max_val ** 2 / mse)

def psnr_torch(input, target, max_val=1.0):
    mse = F.mse_loss(input, target)
    psnr = 10 * torch.log10(max_val * max_val / mse)
    return psnr

def vmaf(input, target):
    pass


