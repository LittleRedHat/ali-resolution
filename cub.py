# -*- coding: utf-8 -*-
####################################
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-19         #
####################################
from PIL import Image
import os
from utils.utils import ensure_path

frames = 100
image_dir = '/home/xxc/workspace/ali-resolution/data'
file_list ='test_sisr.txt'
input_list = [line.rstrip().split()[0] for line in open(os.path.join(image_dir, file_list))]
up_scale = 4
output_root = '/home/xxc/workspace/ali-resolution/exps/cub/test'
for input_file in input_list:
    im = Image.open(os.path.join(image_dir, input_file))
    target = im.resize((im.size[0] * up_scale, im.size[1] * up_scale), Image.BICUBIC)
    track = input_file.split('/')[-2][:11]
    frame_id = input_file.split('/')[-1]
    target_file = os.path.join(output_root, 'generated', os.path.basename(track), frame_id)
    ensure_path(os.path.dirname(target_file))
    target.save(target_file)







