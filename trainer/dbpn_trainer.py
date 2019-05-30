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
import os
from utils.utils import ensure_path


class DBPNTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler, config):
        super(DBPNTrainer, self).__init__(model, optimizer, scheduler, config)
        self.log_frq = self.config.get('log_frq', 1000)
        self.log_dir = self.config.get('log_dir', 'logs')
        self.loss_fn = nn.L1Loss(size_average=True)

        # self.loss_fn = nn.MSELoss(size_average=True)

    def _train_epoch(self, epoch, train_dataloader, val_dataloader=None, test_dataloader=None):
        self.model.train()
        result = {}
        start = time.time()
        epoch_total_loss = 0.0
        for step, sample in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            inputs, targets, bicubics,  _ = sample  ## inputs - > batch_size * 1 * channel * height * width
            inputs = inputs.squeeze(1)
            targets = targets.squeeze(1)
            bicubics = bicubics.squeeze(1)
            ensure_path(os.path.join(self.config['output_dir'], 'sample'))
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
                bicubics = bicubics.cuda()
            sr = self.model((inputs, bicubics))
            loss = self.loss_fn(sr, targets)
            loss.backward()
            self.optimizer.step()
            _loss = loss.cpu().item() if torch.cuda.is_available() else loss.item()
            epoch_total_loss += _loss
            if step % self.log_frq == 0 or step == len(train_dataloader) - 1:
                random_index = np.random.randint(0, inputs.shape[0])
                self._save_image(inputs[random_index].cpu().numpy(),
                                 os.path.join(self.config['output_dir'], 'sample', '{}_{}_l.bmp'.format(epoch, step)))
                self._save_image(targets[random_index].cpu().numpy(),
                                 os.path.join(self.config['output_dir'], 'sample', '{}_{}_h.bmp'.format(epoch, step)))
                self._save_image(sr[random_index].cpu().detach().numpy(),
                                 os.path.join(self.config['output_dir'], 'sample', '{}_{}_pred.bmp'.format(epoch, step)))
                end = time.time()
                message = 'epoch {} {} / {} loss = {} {:4f} min(s)'.format(epoch, step, len(train_dataloader), _loss,  (end - start) / 60)
                self.logger.info(message)
        result['train_loss'] = epoch_total_loss / len(train_dataloader)
        if val_dataloader is not None:
            prediction, val_result = self._val_epoch(epoch, val_dataloader)
            result.update(val_result)
        return result

    def _val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        prediction = None  ## frames * batch * channel
        truth = None
        with torch.no_grad():
            for step, sample in enumerate(val_dataloader):
                inputs, targets, bicubics, _ = sample  ## batch * 1 * channel * height * width
                inputs = inputs.squeeze(1)
                targets = targets.squeeze(1)
                bicubics = bicubics.squeeze(1)
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    bicubics = bicubics.cuda()
                sr = self.model((inputs, bicubics))
                sr = sr.cpu().numpy()  if torch.cuda.is_available() else sr.numpy()
                if prediction is None:
                    prediction = list(sr)
                    truth = list(targets.numpy())
                else:
                    prediction += list(sr)
                    truth += list(targets.numpy())
        prediction = np.array(prediction)  ## len(val_dataloader) * frame * c * h * w
        truth = np.array(truth)
        psnr = psnr_fn(prediction, truth, max_val=1.0)
        val_result = {'val_psnr': psnr}
        self._visualize(epoch, prediction[:20], os.path.join(self.config['output_dir'], 'vis', str(epoch)))
        return prediction, val_result
