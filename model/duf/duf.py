# -*- coding: utf-8 -*-
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-24         #
__author__ = "zookeeper"
import torch.nn as nn

class DUF_16L(nn.Module):
    def __init__(self, config):
        super(DUF_16L, self).__init__()
        uf = config['uf']

    def forward(self, x):
        """
        :param x:
        :return:
        """
        nn.Conv3d()
