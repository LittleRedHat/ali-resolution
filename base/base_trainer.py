# -*- coding: utf-8 -*-
####################################
#  Copyright (c): zookeeper 2019.  #
#  Author: zookeeper               #
#  Email: 1817022566@qq.com        #
#  Update Date: 2019-05-19         #
####################################
__author__ = 'zookeeper'
from collections import defaultdict
import torch
import os
import json
from logger.logger import SummaryLogger
from abc import abstractmethod
import logging

logging.basicConfig(level=logging.INFO, format='')
import torch.nn as nn
import numpy as np
from PIL import Image
from utils.utils import ensure_path


class BaseTrainer:
    def __init__(self, model, optimizer, scheduler, config):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.summarizer = SummaryLogger(config.get('log_dir', 'logs'))
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("init logging")
        self.results = defaultdict(lambda: [])
        if torch.cuda.is_available():
            model = model.cuda()
        self.model = model
        self._model = model
        self._restore_ckpt()
        if torch.cuda.device_count() > 1:
            print("use data parallel")
            self.model = nn.DataParallel(model)

    def train(self, train_dataloader, val_dataloader=None, test_dataloader=None):
        epochs = self.config['epochs']
        for epoch in range(epochs):
            result = self._train_epoch(epoch, train_dataloader, val_dataloader, test_dataloader)
            for key, value in result.items():
                self.results[key].append(value)
                self.logger.info('{:15s}: {}'.format(key, value))
            self._save_ckpt(epoch)
            self._save_results()

    def _save_ckpt(self, epoch):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self._model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        filename = os.path.join(self.config['ckpt_dir'], 'model-epoch{}.pth.tar'.format(epoch))

        torch.save(state, filename)
        self._save_results()

    def _restore_ckpt(self):
        if self.config.get('restore_ckpt', None) is not None:
            self.logger.info("load model from {}".format(self.config.get("restore_ckpt")))
            ckpt = torch.load(self.config['restore_ckpt'])
            state = ckpt['state_dict']
            origin_state = self.model.state_dict().copy()
            for key, parameter in state.items():
                if key in origin_state.keys():
                    origin_state[key] = parameter
            self.model.load_state_dict(origin_state)

    def _save_results(self):
        # if not os.path.exists(self.config['log_dir']):
        #     os.makedirs(self.config['log_dir'])
        with open(os.path.join(self.config['log_dir'], 'metrics.json'), 'w') as fou:
            json.dump(self.results, fou)

    def _save_image(self, prediction, output):
        prediction = np.clip(prediction, 0, 1)
        frame = np.uint8(prediction * 255.0).transpose(1, 2, 0)
        # print(np.max(frame[:, :, 0]), np.max(frame[:, :, 1]), np.max(frame[:, :, 2]))
        image = Image.fromarray(frame)
        image.save(output)

    def _visualize(self, epoch, prediction, output_dir):
        frames = prediction.shape[0]
        ensure_path(output_dir)
        for i in range(frames):
            self._save_image(prediction[i], os.path.join(output_dir, '{}.bmp'.format(i)))

    @abstractmethod
    def _train_epoch(self, epoch, train_dataloader, val_dataloader=None, test_dataloader=None):
        raise NotImplementedError

    @abstractmethod
    def _val_epoch(self, epoch, val_dataloader):
        raise NotImplementedError
