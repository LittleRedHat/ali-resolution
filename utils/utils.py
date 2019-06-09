# -*- coding: utf-8 -*-
####################################
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-19         #
####################################
__author__ = 'zookeeper'
import os
import shutil
import numpy as np
import yaml


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def save_config(config, output):
    with open(output, 'w') as f:
        yaml.dump(config, f, allow_unicode=True)


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += (128 / 255.0)
    return ycbcr


def ycbcr2rgb(im, channel_first=True):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= (128.0 / 255.0)
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 1.0, 1.0)
    np.putmask(rgb, rgb < 0, 0.0)
    return rgb






