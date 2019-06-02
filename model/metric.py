# -*- coding: utf-8 -*-
__author__ = 'zookeeper'
import numpy as np
import torch.nn.functional as F
import torch

def psnr(input, target, max_val=1.0):
    '''

    :param input: batch * channel * h * w
    :param target: batch * channel * h * w
    :param max_val: 1.0 or 255.0
    :return: psnr
    '''
    mse = np.square(input - target).mean(axis=(1, 2, 3))
    return 10 * np.log10(max_val ** 2 / mse).mean()


def psnr_torch(input, target, max_val=1.0):
    mse = F.mse_loss(input, target, reduction='none').mean((1, 2, 3))
    psnr = 10 * torch.log10(max_val ** 2 / mse).mean()
    return psnr


def vmaf(input, target):
    pass


