#!/usr/bin/env bash
# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
from model.common.common import MeanShift
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv2d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv2d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


class WSDR_B(nn.Module):
    def __init__(self, config):
        super(WSDR_B, self).__init__()
        # hyper-params
        self.config = config
        n_resblocks = config.get('n_resblocks')
        n_feats = config.get('n_feats')
        up_scale = config.get('scale_factor')
        channels = config['num_channels']
        res_scale = config['res_scale']
        use_wn = config.get('use_wn', True)
        kernel_size = 3
        act = nn.ReLU(True)
        if not use_wn:
            wn = lambda x: x
        else:
            wn = lambda x: torch.nn.utils.weight_norm(x)

        self.sub_mean = MeanShift(sign=-1)
        self.add_mean = MeanShift(sign=1)

        # define head module
        head = []
        head.append(
            wn(nn.Conv2d(channels, n_feats, 3, padding=3//2)))

        # define body module
        body = []
        for i in range(n_resblocks):
            body.append(
                Block(n_feats, kernel_size, act=act, res_scale=res_scale, wn=wn))

        # define tail module
        tail = []
        out_feats = up_scale * up_scale * channels
        tail.append(
            wn(nn.Conv2d(n_feats, out_feats, 3, padding=3//2)))
        tail.append(nn.PixelShuffle(up_scale))

        skip = []
        skip.append(
            wn(nn.Conv2d(channels, out_feats, 5, padding=5//2))
        )
        skip.append(nn.PixelShuffle(up_scale))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x, bicubic = x
        x = self.sub_mean(x)
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        # print(x[0, 0, :, :])
        x = self.tail(x)
        # x += s
        x += bicubic
        x = self.add_mean(x)
        return x
