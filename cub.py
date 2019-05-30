# -*- coding: utf-8 -*-
####################################
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-19         #
####################################
from PIL import Image
import os
import shutil

# frames = 100
# image_dir = '/home/xxc/workspace/ali-resolution/data'
# file_list = 'test.txt'
# input_list = [line.rstrip() for line in open(os.path.join(image_dir, file_list))]
# up_scale = 4
# output_root = '/home/xxc/workspace/ali-resolution/data/output/cub'
#
# if not os.path.exists(output_root):
#     os.makedirs(output_root)
#
# for track in input_list:
#     for frame_id in range(1, frames + 1, 1):
#         input_file = os.path.join(image_dir, track + '_l', '{:03d}.bmp'.format(frame_id))
#         im = Image.open(input_file)
#         target = im.resize((im.size[0] * up_scale, im.size[1] * up_scale), Image.BICUBIC)
#         target_file = os.path.join(output_root, os.path.basename(track), '{:03d}.bmp'.format(frame_id))
#         if not os.path.exists(os.path.dirname(target_file)):
#             os.makedirs(os.path.dirname(target_file))
#         target.save(target_file)
# from PIL import Image
# import numpy as np
# def psnr(input, target, max_val=1.0):
#     mse = np.square(input - target).mean()
#     return 10 * np.log10(max_val ** 2 / mse)
#
# lr = Image.open('/Users/zookeeper/Desktop/187_100_pred.bmp')
# lrx2 = np.array(lr)
# # lrx2 = np.array(lr.resize((lr.width * 2, lr.height * 2), Image.BICUBIC).convert('YCbCr'))
# sr = np.array(Image.open('/Users/zookeeper/Desktop/187_100_h.bmp'))
#
# m = psnr(lrx2[:, :, 0], sr[:, :, 0], 255.0)
# print(m)






