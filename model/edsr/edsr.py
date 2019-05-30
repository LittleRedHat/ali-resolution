#!/usr/bin/env python
# -*- coding: utf-8 -*-
####################################
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-28         #
####################################
from base.base_model import BaseModel
__author__ = "zookeeper"
from model.common.common import MeanShift, default_conv, Upsampler, ResBlock
import torch.nn as nn


class EDSR(BaseModel):
    def __init__(self, config):
        super(EDSR, self).__init__(config)
        n_resblocks = config.get('n_resblocks')
        n_feats = config.get('n_feats')
        up_scale = config.get('scale_factor')
        channels = config['num_channels']
        res_scale = config.get('res_scale', 1.0)
        # self.sub_mean = MeanShift(sign=-1)
        # self.add_mean = MeanShift(sign=1)

        # head for extract feature
        m_head = [default_conv(channels, n_feats, 3)]
        # body
        m_body = [
            ResBlock(
                default_conv, n_feats, 3, act=nn.ReLU(inplace=True), res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(default_conv(n_feats, n_feats, 3))
        # define tail module
        m_tail = [
            Upsampler(default_conv, up_scale, n_feats, act=False),
            default_conv(n_feats, channels, 3)
        ]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        inputs, _ = x
        # x = self.sub_mean(inputs)
        x = self.head(inputs)
        res = self.body(x)
        res += x
        x = self.tail(res)
        # x = self.add_mean(x)
        return x