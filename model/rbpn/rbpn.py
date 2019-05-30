# -*- coding: utf-8 -*-
####################################
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-19         #
####################################
__author__ = 'zookeeper'

import torch
import torch.nn as nn
from .base_network import ConvBlock, DeconvBlock, ResnetBlock
from .dbpn import DBPN_2, DBPN_3, DBPN_7



class RBPN(nn.Module):
    def __init__(self, config):
        super(RBPN, self).__init__()
        # base_filter=256
        # feat=64

        window = config['window']
        scale_factor = config['scale_factor']
        num_channels = config['num_channels']
        base_filter = config['base_filter']
        feat = config['feat']
        num_stages = config['num_stages']
        n_resblock = config['n_resblock']
        residual = config['residual']
        self.window = window
        self.residual = residual
        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(8, base_filter, 3, 1, 1, activation='prelu', norm=None)

        # DBPNS
        self.DBPN = DBPN_2(config)

        # Res-Block1
        modules_body1 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body1.append(DeconvBlock(base_filter, feat, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat1 = nn.Sequential(*modules_body1)

        # Res-Block2
        modules_body2 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body2.append(ConvBlock(feat, feat, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)

        # Res-Block3
        modules_body3 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body3.append(ConvBlock(feat, base_filter, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat3 = nn.Sequential(*modules_body3)

        # Reconstruction
        self.output = ConvBlock((window - 1) * feat, num_channels, 3, 1, 1, activation=None, norm=None)

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

    def forward(self, inputs):
        x, neigbors, flow, bicubic = inputs

        # Initial feature extraction
        feat_input = self.feat0(x)
        feat_frame = []
        for j in range(len(neigbors)):
            feat_frame.append(self.feat1(torch.cat((x, neigbors[j], flow[j]), 1)))

        # Projection
        Ht = []
        for j in range(len(neigbors)):
            h0 = self.DBPN(feat_input)
            h1 = self.res_feat1(feat_frame[j])
            e = h0 - h1
            e = self.res_feat2(e)
            h = h0 + e
            Ht.append(h)
            feat_input = self.res_feat3(h)

        # Reconstruction
        out = torch.cat(Ht, 1)
        output = self.output(out)
        if self.residual:
            output = output + bicubic
        return output