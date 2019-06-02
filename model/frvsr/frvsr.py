# -*- coding: utf-8 -*-
####################################
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-19         #
####################################
__author__ = 'zookeeper'
import torch.nn as nn
import torch.nn.functional as F
from .layers import FNet, STN, SRNet, SpaceToDepth


class FRNet(nn.Module):
    def __init__(self, config):
        channel = config['channel']
        gain = config['gain']
        up_scale = config['up_scale']
        n_rb = config['n_rb']
        super(FRNet, self).__init__()

        self.fnet = FNet(channel, gain=gain, f=config['filter'], n_layer=config['flow_layer_num'])
        self.warp = STN(padding_mode='border')
        self.snet = SRNet(channel, up_scale, n_rb)
        self.space_to_depth = SpaceToDepth(up_scale)
        self.up_scale = up_scale

    def forward(self, x):
        lr, last_lr, last_sr = x
        flow = self.fnet(lr, last_lr)
        flow2 = self.up_scale * F.interpolate(flow, scale_factor=self.up_scale, mode='bilinear', align_corners=False)
        hw = self.warp(last_sr, flow2[:, 0], flow2[:, 1], normalized=False)
        lw = self.warp(last_lr, flow[:, 0], flow[:, 1], normalized=False)
        hws = self.space_to_depth(hw)
        y = self.snet(hws, lr)
        return y, hw, lw, flow2