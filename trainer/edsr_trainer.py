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
            # inputs = inputs.squeeze(1)
            # targets = targets.squeeze(1)
            # bicubics = bicubics.squeeze(1)
            ensure_path(os.path.join(self.config['output_dir'], 'sample'))
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
                bicubics = bicubics.cuda()
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
                self._save_image(inputs[random_index].cpu().numpy(),
                                 os.path.join(self.config['output_dir'], 'sample',
                                              '{}_{}_l.bmp'.format(epoch, step)))
                self._save_image(targets[random_index].cpu().numpy(),
                                 os.path.join(self.config['output_dir'], 'sample',
                                              '{}_{}_h.bmp'.format(epoch, step)))
                self._save_image(sr[random_index].cpu().detach().numpy(),
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

    def eval(self, test_dataloader):
        self.model.eval()
        data_config = test_dataloader.dataset.config
        scale = data_config.get('upscale_factor')
        patch_size = data_config.get('patch_size')
        stride = data_config.get('sample_stride')
        save_dir = os.path.join(self.config.get('result_dir'), 'test')
        ensure_path(save_dir)
        with torch.no_grad():
            for step, sample in enumerate(test_dataloader):
                inputs, _, bicubics, tracks, frames = sample
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    bicubics = bicubics.cuda()
                for i, input in enumerate(inputs):
                    sr = self._eval_chop(input, bicubics[i], scale, stride, [patch_size[1], patch_size[0]])
                    track, frame = tracks[i], frames[i]
                    output_dir = os.path.join(save_dir, track)
                    ensure_path(output_dir)
                    self._save_image(sr, os.path.join(output_dir, frame))

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
                sr = self.model((inputs, bicubics))
                # psnr = psnr_torch(sr, targets.cuda(), max_val=255.0)
                # print(psnr)
                sr = sr.cpu().numpy()  if torch.cuda.is_available() else sr.numpy()
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
        self._visualize(epoch, prediction[:20], os.path.join(self.config['output_dir'], 'vis', str(epoch)))
        return prediction, val_result
