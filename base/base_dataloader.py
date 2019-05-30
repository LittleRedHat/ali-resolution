# -*- coding: utf-8 -*-
####################################
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-24         #
####################################
__author__ = 'zookeeper'

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image, ImageOps
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


class SISRDataset(Dataset):
    def __init__(self, config, transform=None):
        super(SISRDataset, self).__init__()
        self.image_dir = config['image_dir']
        file_list = config['file_list']
        self.input_frames, self.target_frames = self._load_frames(file_list)
        self.data_augmentation = config['data_argumentation']
        self.patch_size = config['patch_size']
        self.upscale_factor = config['upscale_factor']
        self.mod = config.get("mod", 8)
        # self.mode = config['mode']
        self.repeat = config.get('repeat', 1)
        self.transform = transform
        self.config = config

    def _load_frames(self, file_list):
        input_frames = []
        target_frames = []
        with open(file_list) as f:
            for line in f:
                input_file, target_file = line.strip().split()
                input_frames.append(os.path.join(self.image_dir, input_file))
                target_frames.append(os.path.join(self.image_dir, target_file))
        return input_frames, target_frames

    @staticmethod
    def load_image(input_file, target_file, upscale_factor):
        input_f = Image.open(input_file).convert('RGB')
        if os.path.exists(target_file):
            tw, th = input_f.width * upscale_factor, input_f.height * upscale_factor
            target_f = Image.open(target_file).convert('RGB')
            if tw != target_f.width or th != target_f.height:
                target_f = target_f.resize((tw, th), Image.BICUBIC)
        else:
            tw, th = input_f.width * upscale_factor, input_f.height * upscale_factor
            target_f = input_f.resize((tw, th), Image.BICUBIC)
        return input_f, target_f

    @staticmethod
    def mod_pad(input_f, target_f, mod, upscale):
        ih, iw = input_f.size
        desired_h = ih + (mod - ih % mod) % mod
        desired_w = iw + (mod - iw % mod) % mod
        # print(desired_h, desired_w)
        pad_iw = desired_w - iw
        pad_ih = desired_h - ih
        target_w = desired_w * upscale
        target_h = desired_h * upscale
        pad_th = target_h - ih * upscale
        pad_tw = target_w - iw * upscale
        pad_input = ImageOps.expand(input_f, (pad_ih // 2, pad_iw // 2, pad_ih - pad_ih // 2, pad_iw - pad_iw // 2))
        pad_target = ImageOps.expand(target_f, (pad_th // 2, pad_tw // 2, pad_th - pad_th // 2, pad_tw - pad_tw // 2))
        return pad_input, pad_target

    @staticmethod
    def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
        (ih, iw) = img_in.size
        tp = [scale * item for item in patch_size]
        ip = patch_size
        if ix == -1:
            ix = random.randrange(0, iw - ip[1] + 1)
        if iy == -1:
            iy = random.randrange(0, ih - ip[0] + 1)

        (tx, ty) = (scale * ix, scale * iy)

        img_in = img_in.crop((iy, ix, iy + ip[0], ix + ip[1]))  # [:, iy:iy + ip, ix:ix + ip]
        img_tar = img_tar.crop((ty, tx, ty + tp[0], tx + tp[1]))  # [:, ty:ty + tp, tx:tx + tp]
        info_patch = {
            'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

        return img_in, img_tar, info_patch

    @staticmethod
    def argument(img_in, img_tar, flip_h=True, rot=True):
        info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

        if random.random() < 0.5 and flip_h:
            img_in = ImageOps.flip(img_in)
            img_tar = ImageOps.flip(img_tar)
            info_aug['flip_h'] = True
        if rot:
            if random.random() < 0.5:
                img_in = ImageOps.mirror(img_in)
                img_tar = ImageOps.mirror(img_tar)
                info_aug['flip_v'] = True
            if random.random() < 0.5:
                img_in = img_in.rotate(180)
                img_tar = img_tar.rotate(180)
                info_aug['trans'] = True
        return img_in, img_tar, info_aug

    def __getitem__(self, item):
        frame_id = item // self.repeat
        input_file = self.input_frames[frame_id]
        target_file = self.target_frames[frame_id]
        input_f, target_f = self.load_image(input_file, target_file, self.upscale_factor)
        if self.patch_size and not self.config.get('keep_full', False):
            input_f, target_f, _ = self.get_patch(input_f, target_f, self.patch_size, self.upscale_factor, ix=-1, iy=-1)
        input_f, target_f = self.mod_pad(input_f, target_f, self.mod, self.upscale_factor)
        if self.data_augmentation:
            input_f, target_f, aug_info = self.argument(input_f, target_f)
        bicubic = input_f.resize((input_f.width * self.upscale_factor, input_f.height * self.upscale_factor), Image.BICUBIC)
        if self.transform:
            input_f = self.transform(input_f)
            target_f = self.transform(target_f)
            bicubic = self.transform(bicubic)

        # input_f = torch.from_numpy(np.array(input_f).astype(np.float32)).permute(2, 0, 1)
        # bicubic = torch.from_numpy(np.array(bicubic).astype(np.float32)).permute(2, 0, 1)
        # target_f = torch.from_numpy(np.array(target_f).astype(np.float32)).permute(2, 0, 1)

        track = input_file.split('/')[-2].replace('_l', '')
        frame = input_file.split('/')[-1]
        return input_f, target_f, bicubic,track, frame

    def __len__(self):
        return len(self.input_frames) * self.repeat

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=4):
        data_loader = torch.utils.data.DataLoader(dataset=self,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  drop_last=False,
                                                  pin_memory=True)
        return data_loader


class SequentialDataset(Dataset):
    def __init__(self, config, transform=None):
        super(SequentialDataset, self).__init__()
        image_dir = config['image_dir']
        window = config['window']
        stride = config['stride']
        sample_stride = config.get('sample_stride', stride)
        sample_strategy = config['sample_strategy']
        upscale_factor = config['upscale_factor']
        data_augmentation = config['data_argumentation']
        file_list = config['file_list']
        patch_size = config.get('patch_size', None)
        mod = config.get("mod", 8)
        input_list = [line.rstrip() for line in open(file_list)]
        self.input_tracks = [os.path.join(image_dir, x + "_l") for x in input_list]
        self.target_tracks = [os.path.join(image_dir, x + "_h_GT") for x in input_list]
        self.input_frames, self.target_frames = self.load_frames()
        self.track_frames = len(self.input_frames[0])
        # sample ${window} frames with stride for train
        self.window = window
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size
        self.stride = stride
        self.sample_stride = sample_stride
        self.sample_strategy = sample_strategy
        self.mod = mod

    def load_frames(self):
        input_frames = []
        target_frames = []
        for i, track in enumerate(self.input_tracks):
            frames = [os.path.join(track, frame) for frame in sorted(os.listdir(track)) if is_image_file(frame)]
            target_track = self.target_tracks[i]
            targets = [os.path.join(target_track, frame) for frame in sorted(os.listdir(target_track)) if is_image_file(frame)]
            input_frames.append(frames)
            target_frames.append(targets)
        # print(input_frames, target_frames)
        # input_slices = []
        # target_slices = []
        # for s in self.stride:
        #     input_slice = input_frames[s:self.stride]
        #     target_slice = target_frames

        return input_frames, target_frames

    @staticmethod
    def load_image(self, input_track_path, target_track_path, frame_id, upscale_factor):
        target_file_path = os.path.join(target_track_path, '{:03d}.bmp'.format(frame_id + 1))
        input_file_path = os.path.join(input_track_path, '{:03d}.bmp'.format(frame_id + 1))
        input_f = Image.open(input_file_path).convert('RGB')
        if os.path.exists(target_file_path):
            tw, th = input_f.width * upscale_factor, input_f.height * upscale_factor
            # target_f = Image.open(target_file_path).convert('RGB')
            target_f = Image.open(target_file_path).convert('RGB').resize((tw, th), Image.BICUBIC)
        else:
            tw, th = input_f.width * upscale_factor, input_f.height * upscale_factor
            # target_f = input_f.resize((tw, th), Image.BICUBIC)
            target_f = None
        return input_f, target_f

    @staticmethod
    def mod_pad(input_f, target_f, mod, upscale):
        ih, iw = input_f.size
        desired_h = ih + (mod - ih % mod) % mod
        desired_w = iw + (mod - iw % mod) % mod
        # print(desired_h, desired_w)
        pad_iw = desired_w - iw
        pad_ih = desired_h - ih
        target_w = desired_w * upscale
        target_h = desired_h * upscale
        pad_th = target_h - ih * upscale
        pad_tw = target_w - iw * upscale
        pad_input = ImageOps.expand(input_f, (pad_ih // 2, pad_iw // 2, pad_ih - pad_ih // 2, pad_iw - pad_iw // 2))
        pad_target = ImageOps.expand(target_f, (pad_th // 2, pad_tw // 2, pad_th - pad_th // 2, pad_tw - pad_tw // 2))
        # print(pad_input.size)
        return pad_input, pad_target

    @staticmethod
    def mod_unpad(inputs, targets, mod):
        pass

    @staticmethod
    def sequential_augment(inputs, targets, flip_h=True, rot=True):
        info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

        # if random.random() < 0.5 and flip_h:
        #
        #     inputs = [ImageOps.flip(img_in) for img_in in inputs]
        #     targets = [ImageOps.flip(img_tar) for img_tar in targets]
        #     info_aug['flip_h'] = True
        if rot:
            if random.random() < 0.5:
                inputs = [ImageOps.mirror(img_in) for img_in in inputs]
                targets = [ImageOps.mirror(img_tar) for img_tar in targets]
                info_aug['flip_v'] = True
            if random.random() < 0.5:
                inputs = [img_in.rotate(180) for img_in in inputs]
                targets = [img_tar.rotate(180) for img_tar in targets]
                info_aug['trans'] = True
        return inputs, targets, info_aug

    @staticmethod
    def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
        (ih, iw) = img_in.size
        tp = [scale * item for item in patch_size]
        ip = patch_size
        if ix == -1:
            ix = random.randrange(0, iw - ip[1] + 1)
        if iy == -1:
            iy = random.randrange(0, ih - ip[0] + 1)

        (tx, ty) = (scale * ix, scale * iy)

        img_in = img_in.crop((iy, ix, iy + ip[0], ix + ip[1]))  # [:, iy:iy + ip, ix:ix + ip]
        img_tar = img_tar.crop((ty, tx, ty + tp[0], tx + tp[1]))  # [:, ty:ty + tp, tx:tx + tp]
        info_patch = {
            'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

        return img_in, img_tar, info_patch

    def __getitem__(self, index):
        track_index = index // self.stride
        frame_id = index % self.stride
        ids = range(frame_id, self.track_frames, self.stride)
        if self.sample_strategy == 'random':
            r = np.random.randint(len(ids) - self.window + 1)
            frame_id = ids[r]
        elif self.sample_strategy == 'start':
            frame_id = ids[0]
        # # frame_id += 1
        inputs = []
        targets = []
        infos = []
        for i in range(self.window):
            id = frame_id + i * self.stride
            input_f, target_f = self.load_image(self.input_tracks[track_index], self.target_tracks[track_index], id, self.upscale_factor)
            if self.patch_size:
                patch_info = {'ix': -1, 'iy': -1}
                input_f, target_f, patch_info = self.get_patch(input_f, target_f, self.patch_size, self.upscale_factor, ix=patch_info['ix'], iy=patch_info['iy'])
            input_f, target_f = self.mod_pad(input_f, target_f, self.mod, self.upscale_factor)
            inputs.append(input_f)
            targets.append(target_f)
            infos.append((track_index, id))

        if self.data_augmentation:
            inputs, targets, _ = self.sequential_augment(inputs, targets)
        if self.transform:
            bicubics = [self.transform(im_in.resize((self.upscale_factor * im_in.width, self.upscale_factor * im_in.height), Image.BICUBIC)).numpy() for im_in in inputs]
            inputs = [self.transform(im_in).numpy() for im_in in inputs] ## frames * c * h * w
            targets = [self.transform(im_tar).numpy() for im_tar in targets]

        inputs = torch.tensor(inputs)
        targets = torch.tensor(targets)
        bicubics = torch.tensor(bicubics)
        # print(inputs.shape)
        return inputs, targets, bicubics, infos

    def __len__(self):
        return len(self.input_tracks) * self.stride

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=4):
        data_loader = torch.utils.data.DataLoader(dataset=self,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  drop_last=False,
                                                  pin_memory=True)
        return data_loader


# class FlowDataset(Dataset):
#     def __init__(self, config, transform=None):
#         super(FlowDataset, self).__init__()
#         image_dir = config['image_dir']
#         window = config['window']
#         stride = config['stride']
#         # note that sample stride is used in sample data, stride is used in extract neighbor frame
#         sample_stride = config.get('sample_stride', stride)
#         upscale_factor = config['upscale_factor']
#         data_augmentation = config['data_argumentation']
#         file_list = config['file_list']
#         patch_size = config['patch_size']
#         future_frame = config['future_frame']
#         sample_strategy = config['sample_strategy']
#         input_list = [line.rstrip() for line in open(file_list)]
#
#         self.input_tracks = [os.path.join(image_dir, x + "_l") for x in input_list]
#         self.target_tracks = [os.path.join(image_dir, x + "_h_GT") for x in input_list]
#         self.input_frames, self.target_frames = self.load_frames()
#         self.track_frames = len(self.input_frames[0])
#         # sample ${window} frames with stride for train
#         self.window = window
#         self.upscale_factor = upscale_factor
#         self.transform = transform
#         self.data_augmentation = data_augmentation
#         self.patch_size = patch_size
#         self.future_frame = future_frame
#         self.stride = stride
#         self.sample_stride = sample_stride
#         self.sample_strategy = sample_strategy
#
#     def load_frames(self):
#         input_frames = []
#         target_frames = []
#         for i, track in enumerate(self.input_tracks):
#             frames = [os.path.join(track, frame) for frame in sorted(os.listdir(track)) if is_image_file(frame)]
#             target_track = self.target_tracks[i]
#             targets = [os.path.join(target_track, frame) for frame in sorted(os.listdir(target_track)) if is_image_file(frame)]
#             input_frames.append(frames)
#             target_frames.append(targets)
#         return input_frames, target_frames
#
#     def __getitem__(self, index):
#         samples_per_track = self.track_frames // self.sample_stride
#         track_index = index // samples_per_track
#         frame_id = index - samples_per_track * track_index
#         frame_id = frame_id * self.sample_stride + 1
#
#         # print("{} / {}".format(track_index, frame_id))
#
#         if self.sample_strategy == 'random':
#             r = np.random.randint(self.sample_stride)
#             frame_id += r
#         elif self.sample_strategy == 'start':
#             frame_id = frame_id
#
#         input_f, neighbor, target_f = load_img(self.input_tracks[track_index], self.target_tracks[track_index],
#                                                frame_id, self.window, self.stride,  self.upscale_factor,
#                                                future=self.future_frame)
#
#         if self.patch_size != 0:
#             input_f, target_f, neighbor, _ = get_patch(input_f, target_f, neighbor, self.patch_size,
#                                                        self.upscale_factor, self.window)
#         if self.data_augmentation:
#             input_f, target_f, neighbor, _ = augment(input_f, target_f, neighbor)
#
#         flow = [get_flow(input_f, j) for j in neighbor]
#         bicubic = rescale_img(input_f, self.upscale_factor)
#
#         if self.transform:
#             target_f = self.transform(target_f)
#             input_f = self.transform(input_f)
#             bicubic = self.transform(bicubic)
#             neighbor = [self.transform(j) for j in neighbor]
#             flow = [torch.from_numpy(j.transpose(2, 0, 1)) for j in flow]
#         return input_f, target_f, neighbor, flow, bicubic, (track_index, frame_id)
#
#     def __len__(self):
#         return len(self.input_tracks) * (self.track_frames // self.sample_stride)
#
#     def get_dataloader(self, batch_size=32, shuffle=True, num_workers=4):
#         data_loader = torch.utils.data.DataLoader(dataset=self,
#                                                   batch_size=batch_size,
#                                                   shuffle=shuffle,
#                                                   num_workers=num_workers,
#                                                   drop_last=False,
#                                                   pin_memory=True)
#         return data_loader
#
#
