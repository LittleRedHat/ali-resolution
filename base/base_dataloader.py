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
import pyflow


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
        self.mod = config.get("mod", 1)
        # self.mode = config['mode']
        self.repeat = config.get('repeat', 1)
        self.transform = transform
        self.format = config.get('format', 'RGB')
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
    def load_image(input_file, target_file, upscale_factor, image_format):
        input_f = Image.open(input_file).convert(image_format)
        if os.path.exists(target_file):
            tw, th = input_f.width * upscale_factor, input_f.height * upscale_factor
            target_f = Image.open(target_file).convert(image_format)
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
        index = item // self.repeat
        input_file = self.input_frames[index]
        target_file = self.target_frames[index]
        input_f, target_f = self.load_image(input_file, target_file, self.upscale_factor, self.format)
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

        track = input_file.split('/')[-2].replace('_l', '')
        frame = input_file.split('/')[-1]
        return input_f, target_f, bicubic, track, frame

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


class VSRDataset(Dataset):
    def __init__(self, config, transform=None):
        super(VSRDataset, self).__init__()
        self.image_dir = config['image_dir']
        self.window = config['window']
        self.stride = config['stride']
        self.repeat = config['repeat']
        self.upscale_factor = config['upscale_factor']
        self.data_augmentation = config['data_argumentation']
        self.patch_size = config.get('patch_size', None)
        self.mod = config.get("mod", 8)
        self.transform = transform
        input_list = [line.rstrip() for line in open(config['file_list'])]
        self.input_tracks = [os.path.join(self.image_dir, x + "_l") for x in input_list]
        self.target_tracks = [os.path.join(self.image_dir, x + "_h_GT") for x in input_list]
        self.input_frames, self.target_frames = self.load_frames()
        self.sample_strategy = config.get('sample_strategy')
        self.format = config.get('format', 'RGB')
        self.config = config

    def load_frames(self):
        input_frames = []
        target_frames = []
        for i, track in enumerate(self.input_tracks):
            frames = [os.path.join(track, frame) for frame in sorted(os.listdir(track)) if is_image_file(frame)]
            target_track = self.target_tracks[i]
            targets = [os.path.join(target_track, frame) for frame in sorted(os.listdir(target_track)) if is_image_file(frame)]
            input_frames.append(frames)
            target_frames.append(targets)

        return input_frames, target_frames

    @staticmethod
    def load_image(input_track_path, target_track_path, frame_id, upscale_factor, image_format):
        target_file_path = os.path.join(target_track_path, '{:03d}.bmp'.format(frame_id + 1))
        input_file_path = os.path.join(input_track_path, '{:03d}.bmp'.format(frame_id + 1))
        input_f = Image.open(input_file_path).convert(image_format)
        if os.path.exists(target_file_path):
            tw, th = input_f.width * upscale_factor, input_f.height * upscale_factor
            target_f = Image.open(target_file_path).convert(image_format).resize((tw, th), Image.BICUBIC)
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
    def mod_unpad(inputs, targets, mod):
        raise NotImplementedError

    @staticmethod
    def sequential_augment(inputs, targets, flip_h=True, rot=True):
        info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

        if random.random() < 0.5 and flip_h:

            inputs = [ImageOps.flip(img_in) for img_in in inputs]
            targets = [ImageOps.flip(img_tar) for img_tar in targets]
            info_aug['flip_h'] = True
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
        track_index = index // self.repeat
        track_frames = len(self.input_frames[track_index])
        if self.sample_strategy == 'random':
            up = track_frames - self.stride * self.window
            start_frame_id = np.random.randint(0, up)
        elif self.sample_strategy == 'start':
            start_frame_id = 0
        inputs = []
        targets = []
        infos = []
        for i in range(self.window):
            id = start_frame_id + i * self.stride
            input_f, target_f = self.load_image(self.input_tracks[track_index], self.target_tracks[track_index], id, self.upscale_factor, self.format)
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

        return inputs, targets, bicubics, infos

    def __len__(self):
        return len(self.input_tracks) * self.repeat

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=4):
        data_loader = torch.utils.data.DataLoader(dataset=self,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  drop_last=False,
                                                  pin_memory=True)
        return data_loader


class VSRTestDataset(Dataset):
    def __init__(self, config, transform=None):
        super(VSRTestDataset, self).__init__()
        self.image_dir = config['image_dir']
        self.window = config['window']
        self.stride = config['stride']
        self.upscale_factor = config['upscale_factor']
        self.data_augmentation = config['data_argumentation']
        self.patch_size = config.get('patch_size', None)
        self.mod = config.get("mod", 8)
        self.transform = transform
        self.sample_strategy = config.get('sample_strategy', 'start')
        file_list = config['file_list']
        self.input_frames, self.target_frames = self._load_frames(file_list)
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

    def __getitem__(self, item):
        file = self.input_frames[item]
        target = self.target_frames[item]
        track = file.split('/')[-2].replace('_l', '')
        frame = file.split('/')[-1]
        frame_id = int(frame.split('.')[0])
        if self.sample_strategy == 'start':
            ids = [int(frame_id) + i * self.stride for i in range(self.window)]
        elif self.sample_strategy == 'center':
            ids = [int(frame_id) + i * self.stride for i in range(-self.window // 2 + 1, self.window // 2 + 1)]
        elif self.sample_strategy == 'end':
            ids = [int(frame_id) + i * self.stride for i in range(-self.window+1, 1)]
        else:
            raise NotImplementedError
        inputs = []
        targets = []
        bicubics = []
        # print(ids, frame_id)
        # tracks = []
        # frames = []
        for _, id in enumerate(ids):
            input_file = os.path.join(track + '_l', '{:03d}.bmp'.format(id))
            target_file = os.path.join(track + '_h_GT', '{:03d}.bmp'.format(id))
            if not os.path.exists(input_file):
                input_file = file
                target_file = target
            input_f, target_f = SISRDataset.load_image(input_file, target_file, self.upscale_factor)
            if self.patch_size and not self.config.get('keep_full', False):
                patch_info = {'ix': -1, 'iy': -1}
                input_f, target_f, patch_info = SISRDataset.get_patch(input_f, target_f, self.patch_size, self.upscale_factor, ix=patch_info['ix'], iy=patch_info['iy'])

            input_f, target_f = SISRDataset.mod_pad(input_f, target_f, self.mod, self.upscale_factor)
            if self.data_augmentation:
                input_f, target_f, aug_info = SISRDataset.argument(input_f, target_f)

            bicubic = input_f.resize((input_f.width * self.upscale_factor, input_f.height * self.upscale_factor),
                                     Image.BICUBIC)
            if self.transform:
                input_f = self.transform(input_f)
                target_f = self.transform(target_f)
                bicubic = self.transform(bicubic)
            inputs.append(input_f)
            targets.append(target_f)
            bicubics.append(bicubic)
        inputs = torch.stack(inputs, dim=0)
        targets = torch.stack(targets, dim=0)
        bicubics = torch.stack(bicubics, dim=0)
        return inputs, targets, bicubics, track, frame

    def __len__(self):
        return len(self.input_frames)

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=4):
        data_loader = torch.utils.data.DataLoader(dataset=self,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  drop_last=False,
                                                  pin_memory=True)
        return data_loader


class VSRFlowDataset(Dataset):
    def __init__(self, config, transform=None):
        super(VSRFlowDataset, self).__init__()
        self.image_dir = config['image_dir']
        self.window = config['window']
        self.stride = config['stride']
        self.upscale_factor = config['upscale_factor']
        self.data_augmentation = config['data_argumentation']
        file_list = config['file_list']
        self.patch_size = config['patch_size']
        self.future_frame = config['future_frame']
        input_list = [line.rstrip() for line in open(file_list)]

        self.input_tracks = [os.path.join(self.image_dir, x + "_l") for x in input_list]
        self.target_tracks = [os.path.join(self.image_dir, x + "_h_GT") for x in input_list]
        self.input_frames, self.target_frames = self.load_frames()
        self.track_frames = len(self.input_frames[0])
        self.transform = transform
        self.sample_strategy = config.get('sample_strategy')
        self.repeat = config['repeat']
        self.config = config

    def load_frames(self):
        input_frames = []
        target_frames = []
        for i, track in enumerate(self.input_tracks):
            frames = [os.path.join(track, frame) for frame in sorted(os.listdir(track)) if is_image_file(frame)]
            target_track = self.target_tracks[i]
            targets = [os.path.join(target_track, frame) for frame in sorted(os.listdir(target_track)) if is_image_file(frame)]
            input_frames.append(frames)
            target_frames.append(targets)
        return input_frames, target_frames

    def load_img(self, input_track_path, target_track_path, frame_id):
        target_file_path = os.path.join(target_track_path, '{:03d}.bmp'.format(frame_id))
        input_file_path = os.path.join(input_track_path, '{:03d}.bmp'.format(frame_id))
        input_f = Image.open(input_file_path).convert('RGB')
        if os.path.exists(target_file_path):
            target_f = Image.open(target_file_path).convert('RGB')
        else:
            tw, th = input_f.width * self.upscale_factor, input_f.height * self.upscale_factor
            target_f = None
            # target_f = input_f.resize((tw, th), Image.BICUBIC)
        neighbor = []
        if not self.future_frame:
            seq = [x for x in range(-self.window, 0, 1)]
        else:
            tt = self.window // 2
            if self.window % 2 == 0:
                seq = [x for x in range(-tt, tt) if x != 0]
            else:
                seq = [x for x in range(-tt, tt + 1) if x != 0]
        for i in seq:
            neighbor_frame_id = frame_id + i * self.stride
            file_name = os.path.join(input_track_path, '{:03d}.bmp'.format(neighbor_frame_id))
            if os.path.exists(file_name):
                temp = Image.open(file_name).convert('RGB')
                neighbor.append(temp)
            else:
                temp = input_f
                neighbor.append(temp)
        return input_f, neighbor, target_f

    def get_patch(self, input_f, target_f, neighbor, ix=-1, iy=-1):
        (ih, iw) = input_f.size
        tp = [self.upscale_factor * item for item in self.patch_size]
        ip = self.patch_size
        if ix == -1:
            ix = random.randrange(0, iw - ip[1] + 1)
        if iy == -1:
            iy = random.randrange(0, ih - ip[0] + 1)
        (tx, ty) = (self.upscale_factor * ix, self.upscale_factor * iy)
        input_f = input_f.crop((iy, ix, iy + ip[0], ix + ip[1]))  # [:, iy:iy + ip, ix:ix + ip]
        target_f = target_f.crop((ty, tx, ty + tp[0], tx + tp[1]))  # [:, ty:ty + tp, tx:tx + tp]
        neighbor = [n.crop((iy, ix, iy + ip[0], ix + ip[1])) for n in neighbor]
        info_patch = {'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}
        return input_f, target_f, neighbor, info_patch

    def augment(self, input_f, target_f, neighbor, flip_h=True, rot=True):
        info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

        if random.random() < 0.5 and flip_h:
            input_f = ImageOps.flip(input_f)
            target_f = ImageOps.flip(target_f)
            neighbor = [ImageOps.flip(n) for n in neighbor]
            info_aug['flip_h'] = True
        if rot:
            if random.random() < 0.5:
                input_f = ImageOps.mirror(input_f)
                target_f = ImageOps.mirror(target_f)
                neighbor = [ImageOps.mirror(n) for n in neighbor]
                info_aug['flip_v'] = True
            if random.random() < 0.5:
                input_f = input_f.rotate(180)
                target_f = target_f.rotate(180)
                neighbor = [n.rotate(180) for n in neighbor]
                info_aug['trans'] = True
        return input_f, target_f, neighbor, info_aug

    @staticmethod
    def get_flow(im1, im2):
        im1 = np.array(im1)
        im2 = np.array(im2)
        im1 = im1.astype(float) / 255.
        im2 = im2.astype(float) / 255.
        # Flow Options:
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 0
        u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                                             nSORIterations, colType)
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        # flow = rescale_flow(flow,0,1)
        return flow

    def __getitem__(self, index):
        track_index = index // self.repeat
        track_frames = len(self.input_frames[track_index])
        if self.sample_strategy == 'random':
            up = track_frames - self.stride * self.window
            start_frame_id = np.random.randint(0, up)
        elif self.sample_strategy == 'start':
            start_frame_id = 0

        start_frame_id += 1

        input_f, neighbor, target_f = self.load_img(self.input_tracks[track_index], self.target_tracks[track_index], start_frame_id)
        if self.patch_size != 0:
            input_f, target_f, neighbor, _ = self.get_patch(input_f, target_f, neighbor)
        if self.data_augmentation:
            input_f, target_f, neighbor, _ = self.augment(input_f, target_f, neighbor)
        flow = [self.get_flow(input_f, j) for j in neighbor]
        bicubic = input_f.resize((self.upscale_factor * input_f.width, self.upscale_factor * input_f.height), Image.BICUBIC)
        if self.transform:
            target_f = self.transform(target_f)
            input_f = self.transform(input_f)
            bicubic = self.transform(bicubic)
            neighbor = [self.transform(j) for j in neighbor]
            flow = [torch.from_numpy(j.transpose(2, 0, 1)) for j in flow]
        return input_f, target_f, neighbor, flow, bicubic, (track_index, start_frame_id)

    def __len__(self):
        return len(self.input_tracks) * self.repeat

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=4):
        data_loader = torch.utils.data.DataLoader(dataset=self,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  drop_last=False,
                                                  pin_memory=True)
        return data_loader


