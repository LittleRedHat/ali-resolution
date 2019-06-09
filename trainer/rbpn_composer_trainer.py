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
import torch.nn.functional as F
import time
from model.metric import psnr as psnr_fn, psnr_torch
import numpy as np
import os
from utils.utils import ensure_path
import json


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
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                neighbors = [x.cuda() for x in neighbors]
                targets = targets.cuda()
                bicubics = bicubics.cuda()

            prediction, flows, warps = self.model((inputs, neighbors, bicubics))
            image_loss = F.l1_loss(prediction, targets)
            warp_loss = [F.l1_loss(w, inputs) for w in warps]
            tv_loss = [total_variance(f) for f in flows]
            flow_loss = torch.stack(warp_loss).sum() * self.config['flow_loss_weight'] + \
                        torch.stack(tv_loss).sum() * self.config['tv_loss_weight']
            loss = image_loss + flow_loss

            # loss = image_loss
            psnr = psnr_torch(prediction, targets, max_val=1.0)
            loss.backward()
            self.optimizer.step()
            end = time.time()
            _loss = loss.cpu().item() if torch.cuda.is_available() else loss.item()
            _image_loss = image_loss.cpu().item() if torch.cuda.is_available() else image_loss.item()
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
                message = 'epoch {} {} / {} loss = {} l1_loss = {} psnr = {} {:4f} min(s)'.format(epoch, step, len(train_dataloader), _loss, _image_loss, _psnr, (end - start) / 60)
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
                window = inputs.shape[1]

                frames = [x.squeeze(1) for x in inputs.split(1, dim=1)]
                labels = [target.squeeze(1) for target in targets.split(1, dim=1)]
                bicubics = [bicubic.squeeze(1) for bicubic in bicubics.split(1, dim=1)]

                inputs = frames.pop(window // 2)
                neighbors = frames
                bicubics = bicubics.pop(window // 2)
                targets = labels.pop(window // 2)

                if torch.cuda.is_available():
                    bicubics = bicubics.cuda()
                    inputs = inputs.cuda()
                    neighbors = [neigh.cuda() for neigh in neighbors]
                    # targets = targets.cuda()
                sr, flows, warps = self.model((inputs, neighbors, bicubics))
                sr = sr.cpu().numpy() if torch.cuda.is_available() else sr.numpy()
                targets = targets.numpy()
                if prediction is None:
                    prediction = list(sr)
                    truth = list(targets)
                else:
                    prediction += list(sr)
                    truth += list(targets)
        prediction = np.array(prediction)
        truth = np.array(truth)
        psnr = psnr_fn(prediction, truth, max_val=1.0)
        val_result = {'val_psnr': psnr}
        self._visualize(epoch, prediction[:20], os.path.join(self.config['output_dir'], 'vis', str(epoch)))
        return val_result

    def eval(self, test_dataloader, compute_score=False):
        self.model.eval()
        data_config = test_dataloader.dataset.config
        scale = data_config.get('upscale_factor')
        patch_size = data_config.get('patch_size')
        stride = data_config.get('sample_stride')
        if isinstance(stride, int):
            stride = [stride, stride]
        save_dir = os.path.join(self.config['output_dir'], 'generated')
        ensure_path(save_dir)
        prediction_list = None
        target_list = None
        with torch.no_grad():
            for step, sample in enumerate(test_dataloader):
                inputs, targets, bicubics, tracks, ids = sample
                batch_size, window, c, h, w = inputs.shape
                prediction = np.zeros((batch_size, c, h * scale, w * scale))
                frames = [x.squeeze(1) for x in inputs.split(1, dim=1)]
                bicubics = [bicubic.squeeze(1) for bicubic in bicubics.split(1, dim=1)]

                inputs = frames.pop(window // 2)  # b * c * h * w
                neighbors = frames
                bicubics = bicubics.pop(window // 2)

                if torch.cuda.is_available():
                    bicubics = bicubics.cuda()
                    inputs = inputs.cuda()
                    neighbors = [neigh.cuda() for neigh in neighbors]
                # note these hypothsise that all video has same resolution
                for top in range(0, h, stride[0]):
                    for left in range(0, w, stride[1]):
                        chop = torch.zeros((batch_size, c, patch_size[0], patch_size[1]), device=inputs.device)
                        bicubic_crop = torch.zeros((batch_size, c, patch_size[0] * scale, patch_size[1] * scale), device=inputs.device)
                        neighbor_crops = []

                        start_t = scale * top
                        end_t = min(scale * top + scale * patch_size[0], h * scale)
                        start_l = scale * left
                        end_l = min(scale * (left + patch_size[1]), w * scale)

                        _crop = inputs[:, :, top:(top + patch_size[0]), left:(left + patch_size[1])]
                        _bicubic_crop = bicubics[:, :, start_t:end_t, start_l:end_l]

                        for neigh in neighbors:
                            neighbor_crop = torch.zeros((batch_size, c, patch_size[0], patch_size[1]), device=inputs.device)
                            _neighbor_crop = neigh[:, :, top:(top + patch_size[0]), left:(left + patch_size[1])]
                            actual_h = _neighbor_crop.shape[-2]
                            actual_w = _neighbor_crop.shape[-1]
                            neighbor_crop[:, :, :actual_h, :actual_w] = _neighbor_crop
                            neighbor_crops.append(neighbor_crop)

                        actual_h = _crop.shape[-2]
                        actual_w = _crop.shape[-1]
                        chop[:, :, :actual_h, :actual_w] = _crop
                        bicubic_crop[:, :, :end_t - start_t, :end_l - start_l] = _bicubic_crop
                        sr, _, _ = self.model((chop, neighbor_crops, bicubic_crop))
                        sr = sr.cpu().numpy() if torch.cuda.is_available() else sr.numpy()
                        prediction[:, :, start_t:end_t, start_l:end_l] = sr[:, :, :end_t - start_t, :end_l - start_l]
                for i, sr in enumerate(prediction):
                    track, frame = tracks[i], ids[i]
                    output_dir = os.path.join(save_dir, str(track))
                    ensure_path(output_dir)
                    self._save_image(sr, os.path.join(output_dir, frame))
                targets = targets.numpy()
                if prediction_list is None:
                    prediction_list = list(prediction)
                    target_list = list(targets)
                else:
                    prediction_list += list(prediction)
                    target_list += list(targets)

            prediction_list = np.array(prediction_list)
            target_list = np.array(target_list)
            if compute_score:
                psnr = psnr_fn(prediction_list, target_list)
                print('psnr is {}'.format(psnr))
                with open(os.path.join(output_dir, 'val_result.json'), 'w') as writer:
                    json.dump({'val_psnr': psnr}, writer)







