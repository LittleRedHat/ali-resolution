# -*- coding: utf-8 -*-
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-24         #
__author__ = "zookeeper"
from base.base_trainer import BaseTrainer
import time
import torch
import torch.nn.functional as F
from model.metric import psnr as psnr_fn
import numpy as np
from utils.utils import ensure_path
import os
from PIL import Image


def total_variance(x, dims=(2, 3), reduction='mean'):
    tot_var = 0
    reduce = 1
    for dim in dims:
        row = x.split(1, dim=dim)
        reduce *= x.shape[dim]
        for i in range(len(row) - 1):
            tot_var += torch.abs(row[i] - row[i + 1]).sum()
    if reduction != 'mean':
        reduce = 1
    return tot_var / reduce


class FRVSRTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler,  config):
        super(FRVSRTrainer, self).__init__(model, optimizer, scheduler, config)
        self.log_frq = self.config.get('log_frq', 1000)
        self.log_dir = self.config.get('log_dir', 'logs')

    def _train_epoch(self, epoch, train_dataloader, val_dataloader = None, test_dataloader = None):
        self.model.train()
        result = {}
        start = time.time()
        epoch_total_loss = 0.0
        for step, sample in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            inputs, targets, bicubics, _ = sample ## inputs - > batch_size * frames * channel * height * width
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
            full_loss = 0.0
            flow_loss = 0.0
            image_loss = 0.0
            # last_lr = torch.zeros((inputs.size(0), inputs.size(2), inputs.size(3), inputs.size(4)), device = inputs.device)
            last_lr = inputs[:, 0]
            last_sr = F.interpolate(last_lr, scale_factor=self.config['up_scale'], mode='bilinear', align_corners=False)
            frames = inputs.size(1)
            for i in range(frames):
                lr = inputs[:, i]
                hr = targets[:, i]
                sr, hrw, lrw, flow = self.model((lr, last_lr, last_sr))
                last_lr = lr
                last_sr = sr
                l2_image = F.mse_loss(sr, hr)
                l2_warp = F.mse_loss(lrw, lr)
                tv_flow = total_variance(flow)
                loss = l2_image + self.config['flow_loss_weight'] * l2_warp + self.config['flow_var_weight'] * tv_flow
                full_loss += loss
                image_loss += l2_image.detach()
                flow_loss += l2_warp.detach()
            full_loss = full_loss / frames
            full_loss.backward()
            self.optimizer.step()
            end = time.time()
            _loss = full_loss.cpu().item() if torch.cuda.is_available() else full_loss.item()
            epoch_total_loss += _loss
            if step % self.log_frq == 0 or step == len(train_dataloader) - 1:
                message = 'epoch {} step {}/{} loss = {} {:4f}min(s)'.format(epoch, step, len(train_dataloader), _loss,
                                                                              (end - start) / 60)
                self.logger.info(message)
            # print(sr[0], targets[0][-1])
        result['train_loss'] = epoch_total_loss / len(train_dataloader)
        if val_dataloader is not None:
            prediction, val_result = self._val_epoch(epoch, val_dataloader)
            result.update(val_result)
        return result

    def _val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        prediction = None ## frames * batch * channel
        truth = None
        with torch.no_grad():
            for step, sample in enumerate(val_dataloader):
                inputs, targets,bicubics, _ = sample ## batch * frames * channel * height * width
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    # targets = targets.cuda()
                # last_lr = torch.zeros((inputs.size(0), inputs.size(2), inputs.size(3), inputs.size(4)),
                #                       device=inputs.device)
                last_lr = inputs[:, 0]
                last_sr = F.interpolate(last_lr, scale_factor=self.config['up_scale'], mode='bilinear',
                                        align_corners=False)
                frames_prediction = [] ## frames * batch * c * h * w
                frames = inputs.size(1)
                for i in range(frames):
                    lr = inputs[:, i]
                    # hr = targets[:, i]
                    sr, hrw, lrw, flow = self.model((lr, last_lr, last_sr)) ## batch * c * h * w
                    last_lr = lr
                    last_sr = sr
                    sr = sr.cpu().numpy() if torch.cuda.is_available() else sr.numpy()
                    frames_prediction.append(sr)
                frames_prediction = np.array(frames_prediction).transpose(1, 0, 2, 3, 4) ## batch * frames * c * h * w
                if prediction is None:
                    prediction = list(frames_prediction)
                    truth = list(targets.numpy())
                else:
                    prediction += list(frames_prediction)
                    truth += list(targets.numpy())
        prediction = np.array(prediction) ## len(val_dataloader) * frame * c * h * w
        truth = np.array(truth)
        psnr = psnr_fn(prediction, truth, max_val=1.0)
        val_result = {'val_psnr': psnr}
        self._visualize(epoch, prediction[0])
        return prediction, val_result

    def _visualize(self, epoch, prediction):
        frames = prediction.shape[0]
        output_dir = os.path.join(self.config['output_dir'], 'vis', str(epoch))
        ensure_path(output_dir)
        for i in range(frames):
            frame = np.uint8(prediction[i] * 255.0).transpose(1, 2, 0)
            image = Image.fromarray(frame)
            image.save(os.path.join(output_dir, '{}.bmp'.format(i)))

    def _eval(self, test_dataloader):
        pass

























