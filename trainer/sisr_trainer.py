#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-26         #

__author__ = "zookeeper"
from base.base_trainer import BaseTrainer
import time
import torch
import torch.nn as nn
import numpy as np
from model.metric import psnr as psnr_fn
from model.metric import psnr_torch
import os
from utils.utils import ensure_path
import json
from utils.utils import ycbcr2rgb, rgb2ycbcr


class SISRTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler, config):
        super(SISRTrainer, self).__init__(model, optimizer, scheduler, config)
        self.log_frq = self.config.get('log_frq', 1000)
        self.log_dir = self.config.get('log_dir', 'logs')
        self.loss_fn = nn.L1Loss()

        # self.loss_fn = nn.MSELoss(size_average=True)

    def _train_epoch(self, epoch, train_dataloader, val_dataloader=None, test_dataloader=None):
        self.model.train()
        result = {}
        start = time.time()
        epoch_total_loss = 0.0
        for step, sample in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            inputs, targets, bicubics,  _, _= sample  ## inputs - > batch_size * channel * height * width
            ensure_path(os.path.join(self.config['output_dir'], 'sample'))
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
                bicubics = bicubics.cuda()
            _inputs = inputs
            _targets = targets
            _bicubics = bicubics
            if self.config['format'] == 'YCbCr':
                inputs = inputs[:, 0].unsqueeze(1)
                targets = targets[:, 0].unsqueeze(1)
                bicubics = bicubics[:, 0].unsqueeze(1)

            sr = self.model((inputs, bicubics))
            loss = self.loss_fn(sr, targets)
            psnr = psnr_torch(sr, targets, max_val=1.0)
            loss.backward()
            self.optimizer.step()
            _loss = loss.cpu().item() if torch.cuda.is_available() else loss.item()
            _psnr = psnr.cpu().item() if torch.cuda.is_available() else psnr.item()
            epoch_total_loss += _loss
            if step % self.log_frq == 0 or step == len(train_dataloader) - 1:
                random_index = np.random.randint(0, inputs.shape[0])
                sampled_input = _inputs[random_index].cpu().numpy()
                sampled_target = _targets[random_index].cpu().numpy()
                sampled_sr = sr[random_index].cpu().detach().numpy()
                sampled_bicubic = _bicubics[random_index].cpu().numpy()
                if self.config['format'] == 'YCbCr':
                    sampled_input = ycbcr2rgb(sampled_input.transpose(1, 2, 0)).astype(np.float).transpose(2, 0, 1)
                    sampled_target = ycbcr2rgb(sampled_target.transpose(1, 2, 0)).astype(np.float).transpose(2, 0, 1)
                    _sr = np.zeros_like(sampled_target)
                    _sr[0, :, :] = sampled_sr[0, :, :]
                    _sr[1, :, :] = sampled_bicubic[1, :, :]
                    _sr[2, :, :] = sampled_bicubic[2, :, :]
                    sampled_sr = ycbcr2rgb(_sr.transpose(1, 2, 0)).astype(np.float).transpose(2, 0, 1)

                self._save_image(sampled_input,
                                 os.path.join(self.config['output_dir'], 'sample',
                                              '{}_{}_l.bmp'.format(epoch, step)))
                self._save_image(sampled_target,
                                 os.path.join(self.config['output_dir'], 'sample',
                                              '{}_{}_h.bmp'.format(epoch, step)))
                self._save_image(sampled_sr,
                                 os.path.join(self.config['output_dir'], 'sample',
                                              '{}_{}_pred.bmp'.format(epoch, step)))
                end = time.time()
                message = 'epoch {} {} / {} loss = {} psnr = {}, {:4f} min(s)'.format(epoch, step, len(train_dataloader), _loss, _psnr,  (end - start) / 60)
                self.logger.info(message)

        result['train_loss'] = epoch_total_loss / len(train_dataloader)
        if val_dataloader is not None:
            prediction, val_result = self._val_epoch(epoch, val_dataloader)
            result.update(val_result)

        if self.scheduler is not None:
            self.scheduler.step()
        return result

    def _eval_chop(self, input, bicubic, scale, stride, patch_size):
        c, h, w = input.shape
        # print(h, w)
        if isinstance(stride, int):
            stride = [stride, stride]
        predition = np.zeros((c, h * scale, w * scale))
        chops = []
        bicubic_chops = []
        for top in range(0, h, stride[0]):
            for left in range(0, w, stride[1]):
                chop = torch.zeros((c, patch_size[0], patch_size[1]), device=input.device)
                bicubic_chop = torch.zeros((c, scale * patch_size[0], scale * patch_size[1]), device=input.device)
                _crop = input[:, top:(top + patch_size[0]), left:(left + patch_size[1])]
                actual_h = _crop.shape[1]
                actual_w = _crop.shape[2]
                _bicubic = bicubic[:, scale * top:(scale * top + scale * actual_h), scale * left:(scale * left + scale * actual_w)]
                chop[:, :actual_h, :actual_w] = _crop
                bicubic_chop[:, :scale * actual_h, :scale * actual_w] = _bicubic
                bicubic_chops.append(bicubic_chop)
                chops.append(chop)
        chops = torch.stack(chops, dim=0)
        bicubic_chops = torch.stack(bicubic_chops, dim=0)
        sr = self.model((chops, bicubic_chops))
        # print(sr.shape)
        index = 0
        for top in range(0, h, stride[0]):
            for left in range(0, w, stride[1]):
                start_t = scale * top
                end_t = min(scale * top + scale * patch_size[0], predition.shape[1])
                start_l = scale * left
                end_l = min(scale * (left + patch_size[1]), predition.shape[2])
                _sr = sr[index, :, :end_t - start_t, :end_l - start_l]
                _sr = _sr.cpu().numpy() if torch.cuda.is_available() else _sr.numpy()
                # print(sr.shape, _sr.shape, start_t, end_t, start_l, end_l)
                predition[:, start_t:end_t, start_l:end_l] = _sr
                index += 1
        return predition

    def eval(self, test_dataloader, compute_score=False):
        self.model.eval()
        data_config = test_dataloader.dataset.config
        scale = data_config.get('upscale_factor')
        patch_size = data_config.get('patch_size')
        stride = data_config.get('sample_stride')
        save_dir = os.path.join(self.config['output_dir'], 'generated')
        ensure_path(save_dir)
        with torch.no_grad():
            prediction_list = []
            target_list = []
            for step, sample in enumerate(test_dataloader):
                inputs, targets, bicubics, tracks, frames = sample
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    bicubics = bicubics.cuda()
                _inputs = inputs
                _bicubics = bicubics
                if self.config['format'] == 'YCbCr':
                    inputs = inputs[:, 0].unsqueeze(1)
                    bicubics = bicubics[:, 0].unsuqeeze(1)
                for i, input in enumerate(inputs):
                    sr = self._eval_chop(input, bicubics[i], scale, stride, [patch_size[1], patch_size[0]])
                    track, frame = tracks[i], frames[i]
                    output_dir = os.path.join(save_dir, track)
                    ensure_path(output_dir)
                    bicubic_i = _bicubics[i]
                    if self.config['format'] == 'YCbCr':
                        _sr = np.zeros_like(bicubic_i)
                        _sr[0, :, :] = sr[0, :, :]
                        _sr[1, :, :] = bicubic_i[1, :, :]
                        _sr[2, :, :] = bicubic_i[2, :, :]
                        _sr = ycbcr2rgb(_sr.transpose(1, 2, 0)).transpose(2, 0, 1)
                    self._save_image(sr, os.path.join(output_dir, frame))
                    target_list.append(targets[i].numpy())
                    prediction_list.append(sr)
        prediction_list = np.array(prediction_list)
        target_list = np.array(target_list)
        if compute_score:
            psnr = psnr_fn(prediction_list, target_list)
            print('psnr is {}'.format(psnr))
            with open(os.path.join(self.config['output_dir'], 'val_result.json'), 'w') as writer:
                json.dump({'val_psnr': psnr}, writer)

    def _val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        prediction = None  ## frames * batch * channel
        truth = None
        with torch.no_grad():
            for step, sample in enumerate(val_dataloader):
                inputs, targets, bicubics, _, _ = sample  ## batch * channel * height * width
                # inputs = inputs.squeeze(1)
                # targets = targets.squeeze(1)
                # bicubics = bicubics.squeeze(1)
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    bicubics = bicubics.cuda()

                _inputs = inputs
                _bicubics = bicubics
                _targets = targets
                if self.config['format'] == 'YCbCr':
                    inputs = inputs[:, 0].unsqueeze(1)
                    bicubics = bicubics[:, 0].unsqueeze(1)
                    targets = targets[:, 0].unsqueeze(1)

                sr = self.model((inputs, bicubics))
                sr = sr.cpu().numpy() if torch.cuda.is_available() else sr.numpy()
                if prediction is None:
                    prediction = list(sr)
                    truth = list(targets.numpy())
                else:
                    prediction += list(sr)
                    truth += list(targets.numpy())
        prediction = np.array(prediction)  ## len(val_dataloader) * frame * c * h * w
        truth = np.array(truth)
        # print(prediction.shape, truth.shape)
        psnr = psnr_fn(prediction, truth, max_val=1.0)
        val_result = {'val_psnr': psnr}
        sampled_sr = prediction[:20]
        sampled_bicubic = bicubics[:20]
        if self.config['format'] == 'YCbCr':
            _sr = np.zeros_like(sampled_bicubic)
            _sr[:, 0, :, :] = prediction[:, 0, :, :]
            _sr[:, 1, :, :] = sampled_bicubic[:, 1, :, :]
            _sr[:, 2, :, :] = sampled_bicubic[:, 2, :, :]
            sampled_sr = ycbcr2rgb(_sr.transpose(1, 2, 0)).transpose(2, 0, 1)
        self._visualize(epoch, sampled_sr, os.path.join(self.config['output_dir'], 'vis', str(epoch)))
        return prediction, val_result
