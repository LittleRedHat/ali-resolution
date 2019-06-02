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
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.nn as nn
from model.metric import psnr as psnr_fn, psnr_torch
import numpy as np
import os
from utils.utils import ensure_path


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


class RBPNComposerTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler,  config):
        super(RBPNComposerTrainer, self).__init__(model, optimizer, scheduler, config)
        # self.loss_fn = nn.L1Loss(size_average=True)
        self.log_frq = self.config.get('log_frq', 1000)
        self.log_dir = self.config.get('log_dir', 'logs')

    def _train_epoch(self, epoch, train_dataloader, val_dataloader = None, test_dataloader = None):
        self.model.train()
        result = {}
        start = time.time()
        total_loss = 0.0
        ensure_path(os.path.join(self.config['output_dir'], 'sample'))
        for step, sample in enumerate(train_dataloader):
            inputs, targets, bicubics, _ = sample # inputs - > batch_size * frames * channel * height * width
            # if torch.cuda.is_available():
            #     inputs = inputs.cuda()
            #     targets = targets.cuda()
            #     bicubics = bicubics.cuda()

            self.optimizer.zero_grad()

            window = inputs.shape[1]
            frames = [x.squeeze(1) for x in inputs.split(1, dim=1)]
            labels = [target.squeeze(1) for target in targets.split(1, dim=1)]
            bicubics = [bicubic.squeeze(1) for bicubic in bicubics.split(1, dim=1)]



            inputs = frames.pop(window // 2)
            neighbors = frames
            bicubics = bicubics.pop(window // 2)
            targets = labels.pop(window // 2)

            print(inputs.shape, neighbors[0].shape, bicubics.shape, targets.shape)

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                neighbors = [x.cuda() for x in neighbors]
                targets = targets.cuda()
                bicubics = bicubics.cuda()

            prediction, flows, warps = self.model((inputs, neighbors, bicubics))

            image_loss = F.l1_loss(prediction, labels)
            warp_loss = [F.l1_loss(w, inputs) for w in warps]
            tv_loss = [total_variance(f) for f in flows]
            flow_loss = torch.stack(warp_loss).sum() * self.config['flow_loss_weight'] + \
                        torch.stack(tv_loss).sum() * self.config['tv_loss_weight']
            loss = image_loss + flow_loss
            psnr = psnr_torch(prediction, labels, max_val=1.0)
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
                inputs, targets, bicubics, _ = sample
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    bicubics = bicubics.cuda()

                window = inputs.shape(1)
                frames = [x.squeeze(1) for x in inputs.split(1, dim=1)]
                labels = [target.squeeze(1) for target in targets.split(1, dim=1)]
                bicubics = [bicubic.squeeze(1) for bicubic in bicubics.split(1, dim=1)]

                inputs = torch.stack(frames.pop(window // 2), dim=0)
                neighbors = frames
                bicubics = torch.stack(bicubics.pop(window // 2), dim=0)
                targets = torch.stack(labels.pop(window // 2), dim=0)

                sr, flows, warps = self.model((inputs, neighbors, bicubics))

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

    def eval(self, test_dataloader):
        raise NotImplementedError






