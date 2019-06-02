# -*- coding: utf-8 -*-
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-24         #
__author__ = "zookeeper"
import torch.nn as nn
import torch
import torch.nn.functional as F



class


class DUF(nn.Module):
    def __init__(self, config):
        super(DUF, self).__init__()
        feat = config['feat']
        filter = config['filter']
        n_layer = config['n_layer']
        channels = config['num_channels']
        scale = config['upscale_factor']
        bottle_feat = config.get('bottle_feat', 256)
        filter_generation_feat = config.get('filter_generation_feat', 512)
        F = feat
        G = filter
        self.conv = nn.Conv3d(channels, F, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        block1 = []
        # dense connection
        for i in range(n_layer):
            module = nn.Sequential(*[
                nn.BatchNorm3d(F),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(F, F, (1, 1, 1), (1, 1, 1)),
                nn.BatchNorm3d(F),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(F, G, (3, 3, 3), (1, 1, 1), (1, 1, 1))
            ])
            block1.append(module)
            F += G
        # decoder
        block2 = []
        for i in range(n_layer):
            module = nn.Sequential(
                *[
                    nn.BatchNorm3d(F),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv3d(F, F, (1, 1, 1), (1, 1, 1)),
                    nn.BatchNorm3d(F),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv3d(F, G, (3, 3, 3), (1, 1, 1), (1, 1, 1))
                ]
            )
            block2.append(module)
            F += G
        self.block1 = nn.ModuleList(modules=block1)
        self.block2 = nn.ModuleList(modules=block2)
        self.bottleneck = nn.Sequential(
            *[
                nn.BatchNorm3d(F),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(F, bottle_feat, (1, 3, 3), (1, 1, 1), (0, 1, 1))
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )
        self.residual_generation = nn.Sequential(
            *[
                nn.Conv3d(bottle_feat, bottle_feat, (1, 1, 1), (1, 1, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(bottle_feat, scale * scale * 3, (1, 1, 1), (1, 1, 1)),
            ]
        )
        self.filter_generation = nn.Sequential(
            *[
                nn.Conv3d(bottle_feat, filter_generation_feat, (1, 1, 1), (1, 1, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(filter_generation_feat, scale * scale * 5 * 5 * 1, (1, 1, 1))
            ]
        )
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x, bicubics = x
        x = self.conv(x)
        for module in self.block1.modules():
            t = module(x)
            x = torch.cat([x, t], dim=1)
        for module in self.block2.modules():
            t = module(x)
            x = torch.cat([x[:, :, 1:-1], t], dim=1)

        x = self.bottleneck(x)
        r = self.residual_generation(x)
        f = self.filter_generation(x)
        f = f.reshape((-1, ))
        f = F.softmax(f, dim=-1)
        return f, r
