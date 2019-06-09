#!/usr/bin/env bash
# -*- coding: utf-8 -*-

from base.base_model import BaseModel
from model.common.common import MeanShift, default_conv, Upsampler, ResBlock
import torch.nn as nn
import torch


class MSRB(nn.Module):
  def __init__(self, conv=default_conv, n_feats=64):
    super(MSRB, self).__init__()
    kernel_size_1 = 3
    kernel_size_2 = 5
    self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
    self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
    self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
    self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
    self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    input_1 = x
    output_3_1 = self.relu(self.conv_3_1(input_1))
    output_5_1 = self.relu(self.conv_5_1(input_1))
    input_2 = torch.cat([output_3_1, output_5_1], 1)
    output_3_2 = self.relu(self.conv_3_2(input_2))
    output_5_2 = self.relu(self.conv_5_2(input_2))
    input_3 = torch.cat([output_3_2, output_5_2], 1)
    output = self.confusion(input_3)
    output += x
    return output


class MSRN(BaseModel):
    def __init__(self, config):
        super(MSRN, self).__init__(config)

        self.n_resblocks = config.get('n_resblocks')
        self.n_feats = config.get('n_feats')
        self.up_scale = config.get('scale_factor')
        self.channels = config['num_channels']
        self.res_scale = config.get('res_scale', 1.0)
        kernel_size = 3
        self.sub_mean = MeanShift(sign=-1)
        self.add_mean = MeanShift(sign=1)
        # define head module
        modules_head = [default_conv(self.channels, self.n_feats, kernel_size)]

        # define body module
        modules_body = nn.ModuleList()
        for i in range(self.n_resblocks):
            modules_body.append(
                MSRB(n_feats=self.n_feats))

        # define tail module
        modules_tail = [
            nn.Conv2d(self.n_feats * (self.n_resblocks + 1), self.n_feats, 1, padding=0, stride=1),
            default_conv(self.n_feats, self.n_feats, kernel_size),
            Upsampler(default_conv, self.up_scale, self.n_feats, act=False),
            default_conv(self.n_feats, self.channels, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x, _ = x
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        MSRB_out = []
        for i in range(self.n_resblocks):
            x = self.body[i](x)
            MSRB_out.append(x)
        MSRB_out.append(res)
        res = torch.cat(MSRB_out, 1)
        x = self.tail(res)
        x = self.add_mean(x)
        return x
