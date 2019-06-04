# -*- coding: utf-8 -*-
####################################
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-22         #
####################################
__author__ = "zookeeper"
from base.base_trainer import BaseTrainer
import torch
import time
import torch.nn as nn
from model.metric import psnr as psnr_fn, psnr_torch
import numpy as np
import os
from utils.utils import ensure_path


class RBPNTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler,  config):
        super(RBPNTrainer, self).__init__(model, optimizer, scheduler, config)
        self.loss_fn = nn.L1Loss(size_average=True)
        self.log_frq = self.config.get('log_frq', 1000)
        self.log_dir = self.config.get('log_dir', 'logs')

    def _train_epoch(self, epoch, train_dataloader, val_dataloader = None, test_dataloader = None):
        self.model.train()
        result = {}
        start = time.time()
        total_loss = 0.0
        ensure_path(os.path.join(self.config['output_dir'], 'sample'))
        for step, sample in enumerate(train_dataloader):
            inputs, targets, neighbors, flows, bicubics, _ = sample
            # print(inputs.shape, targets.shape, neighbors[0].shape, flows[0].shape, bicubics.shape)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
                bicubics = bicubics.cuda()
                neighbors = [j.cuda() for j in neighbors]
                flows = [j.cuda().float() for j in flows]
            self.optimizer.zero_grad()
            prediction = self.model((inputs, neighbors, flows, bicubics))
            loss = self.loss_fn(prediction, targets)
            psnr = psnr_torch(prediction, targets, max_val=1.0)
            loss.backward()
            self.optimizer.step()
            end = time.time()
            _loss = loss.cpu().item() if torch.cuda.is_available() else loss.item()
            _psnr = psnr.cpu().item() if torch.cuda.is_available() else psnr.item()
            total_loss += _loss
            if step % self.log_frq == 0 or step == len(train_dataloader) - 1:
                random_index = np.random.randint(0, inputs.shape[0])
                self._save_image(inputs[random_index].cpu().numpy(),
                                 os.path.join(self.config['output_dir'], 'sample',
                                              '{}_{}_l.bmp'.format(epoch, step)))
                self._save_image(targets[random_index].cpu().numpy(),
                                 os.path.join(self.config['output_dir'], 'sample',
                                              '{}_{}_h.bmp'.format(epoch, step)))
                self._save_image(prediction[random_index].cpu().detach().numpy(),
                                 os.path.join(self.config['output_dir'], 'sample',
                                              '{}_{}_pred.bmp'.format(epoch, step)))
                message = 'epoch {} {} / {} loss = {} psnr = {} {:4f} min(s)'.format(epoch, step, len(train_dataloader), _loss, _psnr, (end - start) / 60)
                self.logger.info(message)

        result['train_loss'] = total_loss / len(train_dataloader)
        if val_dataloader is not None:
            val_result = self._val_epoch(epoch, val_dataloader)
            result.update(val_result)
        return result

    def _val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        prediction = None
        truth = None
        ensure_path(os.path.join(self.config['output_dir'], 'vis'))
        with torch.no_grad():
            for step, sample in enumerate(val_dataloader):
                inputs, targets, neighbors, flows, bicubics, _ = sample
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    bicubics = bicubics.cuda()
                    neighbors = [j.cuda() for j in neighbors]
                    flows = [j.cuda().float() for j in flows]
                sr = self.model((inputs, neighbors, flows, bicubics))

                sr = sr.cpu().numpy().astype(np.uint8) if torch.cuda.is_available() else sr.numpy().astype(np.uint8)
                targets = targets.numpy().astype(np.uint8)
                if prediction is None:
                    prediction = list(sr)
                    truth = list(targets.numpy())
                else:
                    prediction += list(sr)
                    truth += list(targets.numpy())
        psnr = psnr_fn(prediction, truth, max_val=1.0)
        val_result = {'val_psnr': psnr}
        self._visualize(epoch, prediction[:20], os.path.join(self.config['output_dir'], 'vis', str(epoch)))

        return val_result








