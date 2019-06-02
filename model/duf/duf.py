# -*- coding: utf-8 -*-
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-24         #
__author__ = "zookeeper"
import torch.nn as nn
import torch
import torch.nn.functional as F


class DUF(nn.Module):
    def __init__(self, config):
        super(DUF, self).__init__()
        self.feat = config['feat']
        self.filter = config['filter']
        self.n_layer = config['n_layer']
        self.channels = config['num_channels']
        self.scale = config['upscale_factor']
        self.filter_size = config['filter_size']
        self.bottle_feat = config.get('bottle_feat', 256)
        self.filter_generation_feat = config.get('filter_generation_feat', 512)
        self.window = config['window']

        F = self.feat
        G = filter
        self.conv = nn.Conv3d(self.channels, F, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        block1 = []
        # dense connection
        for i in range(self.n_layer):
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
        for i in range(self.n_layer):
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
                nn.Conv3d(F, self.bottle_feat, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        )
        self.residual_generation = nn.Sequential(
            *[
                nn.Conv3d(self.bottle_feat, self.bottle_feat, (1, 1, 1), (1, 1, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(self.bottle_feat, self.scale * self.scale * self.channels, (1, 1, 1), (1, 1, 1)),
            ]
        )
        self.filter_generation = nn.Sequential(
            *[
                nn.Conv3d(self.bottle_feat, self.filter_generation_feat, (1, 1, 1), (1, 1, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(self.filter_generation_feat, self.scale * self.scale * self.filter_size * self.filter_size, (1, 1, 1))
            ]
        )
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _dyn_filter3d(self, x, F):
        '''
        3D Dynamic filtering
        :param x: batch * depth * h * w
        :param F: batch * tower_depth * tower_depth * h * w
        :return:
        '''
        pass


    def _depth_to_space(self, ):
        pass

    def forward(self, sample):
        """
        :param x:
        :return:
        """
        inputs, bicubics = sample
        x = self.conv(inputs)
        for module in self.block1.modules():
            t = module(x)
            x = torch.cat([x, t], dim=1)
        for module in self.block2.modules():
            t = module(x)
            x = torch.cat([x[:, :, 1:-1], t], dim=1)

        x = self.bottleneck(x)
        r = self.residual_generation(x)
        f = self.filter_generation(x) # f -> batch * (scale * scale * 5 * 5) * depth * h * w
        f = f.reshape((f.shape[0], self.scale * self.scale, self.filter_size * self.filter_size, f.shape[2], f.shape[3], f.shape[4]))
        ## channel softmax
        f = F.softmax(f, dim=1)
        sr = []
        for i in range(self.channels):
            t = self._dyn_filter3d(inputs[:, i, self.window // 2], f[:, 0])
            sr += [t]
        sr = torch.cat(sr, dim=1).unsqueeze(2) ## bs * 3 * 1 * h * w
        sr = r + sr

        return sr
