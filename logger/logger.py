# -*- coding: utf-8 -*-
####################################
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-19         #
####################################

from tensorboardX import SummaryWriter


class SummaryLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalar_summary(self, key, value, iteration):
        if self.writer:
            self.writer.add_scalar(key, value, iteration)

    def add_image_summary(self, tag, image_tensor, iteration):
        if self.writer:
            self.writer.add_image(tag, image_tensor, iteration)