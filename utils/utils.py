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
import json


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def save_config(config, output):
    with open(output, 'w') as f:
        json.dump(config, f)






