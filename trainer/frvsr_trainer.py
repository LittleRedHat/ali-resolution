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
        ensure_path(os.path.join(self.config['output_dir'], 'sample'))
        for step, sample in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            inputs, targets, bicubics, _ = sample # inputs - > batch_size * frames * channel * height * width
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
            full_loss = 0.0
            flow_loss = 0.0
            image_loss = 0.0
            last_lr = inputs[:, 0]
            last_sr = F.interpolate(last_lr, scale_factor=self.config['up_scale'], mode='bilinear', align_corners=False)
            frames = inputs.size(1)
            predictions = []
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
                predictions.append(sr)
            predictions = torch.stack(predictions, dim=0)
            full_loss = full_loss / frames
            full_loss.backward()
            self.optimizer.step()
            end = time.time()
            _loss = full_loss.cpu().item() if torch.cuda.is_available() else full_loss.item()
            epoch_total_loss += _loss
            if step % self.log_frq == 0 or step == len(train_dataloader) - 1:
                message = 'epoch {} step {}/{} loss = {} {:4f}min(s)'.format(epoch, step, len(train_dataloader), _loss,
                                                                            (end - start) / 60)
                random_index = np.random.randint(0, inputs.size(0))
                random_frame_id = np.random.randint(0, frames)
                self._save_image(inputs[random_index][random_frame_id].cpu().numpy(),
                                 os.path.join(self.config['output_dir'], 'sample',
                                              '{}_{}_l.bmp'.format(epoch, step)))
                self._save_image(targets[random_index][random_frame_id].cpu().numpy(),
                                 os.path.join(self.config['output_dir'], 'sample',
                                              '{}_{}_h.bmp'.format(epoch, step)))
                self._save_image(predictions[random_frame_id][random_index].cpu().detach().numpy(),
                                 os.path.join(self.config['output_dir'], 'sample',
                                              '{}_{}_pred.bmp'.format(epoch, step)))

                self.logger.info(message)
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
                inputs, targets, bicubics, _ = sample ## batch * frames * channel * height * width
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                last_lr = inputs[:, 0]
                last_sr = F.interpolate(last_lr, scale_factor=self.config['up_scale'], mode='bilinear',
                                        align_corners=False)
                frames_prediction = [] ## frames * batch * c * h * w
                frames = inputs.size(1)
                for i in range(frames):
                    lr = inputs[:, i]
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
        self._visualize(epoch, prediction[0], os.path.join(self.config['output_dir'], 'vis', str(epoch)))
        return prediction, val_result

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
        with torch.no_grad():
            prediction_list = None
            target_list = None
            for step, sample in enumerate(test_dataloader):
                inputs, targets, bicubics, tracks, ids = sample
                batch_size, frames, c, h, w = inputs.shape
                prediction = np.zeros((batch_size, c, h * scale, w * scale))
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    # bicubics = bicubics.cuda()
                # note these hypothese that all video has same resolution
                for top in range(0, h, stride[0]):
                    for left in range(0, w, stride[1]):
                        chop = torch.zeros((batch_size, c, patch_size[0], patch_size[1]), device=inputs.device)
                        _crop = inputs[:, 0, :, top:(top + patch_size[0]), left:(left + patch_size[1])]
                        actual_h = _crop.shape[-2]
                        actual_w = _crop.shape[-1]
                        chop[:, :, :actual_h, :actual_w] = _crop
                        last_lr = chop
                        last_sr = F.interpolate(last_lr, scale_factor=self.config['up_scale'], mode='bilinear', align_corners=False)
                        for i in range(frames):
                            lr = torch.zeros((batch_size, c, patch_size[0], patch_size[1]), device=inputs.device)
                            _crop = inputs[:, i, :, top:(top + patch_size[0]), left:(left + patch_size[1])]
                            lr[:, :, :actual_h, :actual_w] = _crop
                            sr, _, _, _ = self.model((lr, last_lr, last_sr))
                        sr = sr.cpu().numpy() if torch.cuda.is_available() else sr.numpy()
                        start_t = scale * top
                        end_t = min(scale * top + scale * patch_size[0], h * scale)
                        start_l = scale * left
                        end_l = min(scale * (left + patch_size[1]), w * scale)
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

































