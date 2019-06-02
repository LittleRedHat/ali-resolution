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
from .base_network import ConvBlock, DeconvBlock, ResnetBlock, UpBlock, DownBlock
from .dbpn import DBPN_2, DBPN_3, DBPN_7
from model.frvsr.layers import FNet, STN


class RBPNComposer(nn.Module):
    def __init__(self, config):
        super(RBPNComposer, self).__init__()
        self.rbpn = RBPN(config)
        self.fnet = FNet(config['num_channels'], gain=config['gain'], f=config['f'], n_layer=config['flow_layer_num'])
        self.stn = STN(padding_mode='border')

    def forward(self, x):
        inputs, neighbors, bicubics = x
        flows = []
        warps = []
        for i in neighbors:
            flow = self.fnet(inputs, i)
            warp = self.stn(i, flow[:, 0], flow[:, 0])
            flows.append(flow)
            warps.append(warp)
        sr = self.rbpn((inputs, neighbors, flows, bicubics))
        return sr, flows, warps


class Dbpns(nn.Module):
  def __init__(self, base_filter, feat, num_stages, scale_factor):
    super(Dbpns, self).__init__()

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
    # self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
    self.feat1 = ConvBlock(base_filter, feat, 1, 1, 0, activation='prelu',
                           norm=None)
    # Back-projection stages
    self.up1 = UpBlock(feat, kernel, stride, padding)
    self.down1 = DownBlock(feat, kernel, stride, padding)
    self.up2 = UpBlock(feat, kernel, stride, padding)
    self.down2 = DownBlock(feat, kernel, stride, padding)
    self.up3 = UpBlock(feat, kernel, stride, padding)
    # Reconstruction
    self.output = ConvBlock(num_stages * feat, feat, 1, 1, 0, activation=None,
                            norm=None)

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
    # x = self.feat0(x)
    x = self.feat1(x)

    h1 = self.up1(x)
    h2 = self.up2(self.down1(h1))
    h3 = self.up3(self.down2(h2))

    x = self.output(torch.cat((h3, h2, h1), 1))

    return x


class RBPN(nn.Module):
    def __init__(self, config):
        super(RBPN, self).__init__()
        self.window = config['window']
        self.scale_factor = config['scale_factor']
        self.num_channels = config['num_channels']
        self.base_filter = config['base_filter']
        self.feat = config['feat']
        self.num_stages = config['num_stages']
        self.n_resblock = config['n_resblock']
        self.residual = config['residual']
        if self.scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif self.scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif self.scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        # Initial Feature Extraction
        self.feat0 = ConvBlock(self.num_channels, self.base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(8, self.base_filter, 3, 1, 1, activation='prelu', norm=None)

        # DBPNS
        self.DBPN = Dbpns(config['base_filter'], config['feat'], config['num_stages'], config['scale_factor'])

        # Res-Block1
        modules_body1 = [
            ResnetBlock(self.base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(self.n_resblock)]
        modules_body1.append(DeconvBlock(self.base_filter, self.feat, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat1 = nn.Sequential(*modules_body1)

        # Res-Block2
        modules_body2 = [
            ResnetBlock(self.feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(self.n_resblock)]
        modules_body2.append(ConvBlock(self.feat, self.feat, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)

        # Res-Block3
        modules_body3 = [
            ResnetBlock(self.feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(self.n_resblock)]
        modules_body3.append(ConvBlock(self.feat, self.base_filter, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat3 = nn.Sequential(*modules_body3)

        # Reconstruction
        self.output = ConvBlock((self.window - 1) * self.feat, self.num_channels, 3, 1, 1, activation=None, norm=None)

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