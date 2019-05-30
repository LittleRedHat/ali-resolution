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
from model.metric import psnr
import numpy as np


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
            loss.backward()
            self.optimizer.step()
            end = time.time()
            _loss = loss.cpu().item() if torch.cuda.is_available() else loss.item()
            total_loss += _loss
            if step % self.log_frq == 0 or step == len(train_dataloader) - 1:
                message = 'epoch {} {} / {} loss = {} {:4f} min(s)'.format(epoch, step, len(train_dataloader), _loss,  (end - start) / 60)
                self.logger.info(message)

        result['train_loss'] = total_loss / len(train_dataloader)
        if val_dataloader is not None:
            val_result = self._val_epoch(epoch, val_dataloader)
            result.update(val_result)
        return result

    def _val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        pYs = []
        tYs = []
        with torch.no_grad():
            for step, sample in enumerate(val_dataloader):
                inputs, targets, neighbors, flows, bicubics, _ = sample
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    bicubics = bicubics.cuda()
                    neighbors = [j.cuda() for j in neighbors]
                    flows = [j.cuda().float() for j in flows]
                prediction = self.model((inputs, neighbors, flows, bicubics))

                prediction = prediction.cpu().numpy().astype(np.uint8) if torch.cuda.is_available() else prediction.numpy().astype(np.uint8)
                targets = targets.numpy().astype(np.uint8)


                for index in range(len(prediction)):
                    prediction_im = tensor2image(prediction[index].transpose(1, 2, 0))
                    target_im = tensor2image(targets[index].transpose(1, 2, 0))
                    pY, _, _ = prediction_im.convert('YCbCr').split()
                    tY, _, _ = target_im.convert('YCbCr').split()
                    pYs.append(pY)
                    tYs.append(tY)
            loss = psnr(pYs, tYs)
        return {'val_psnr': loss}








