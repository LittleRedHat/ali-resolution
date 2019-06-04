# -*- coding: utf-8 -*-
__author__ = 'zookeeper'
import argparse
import yaml
import torch
from model import model as model_factory
from dataloder import dataloder as data_factory
import trainer as trainer_factory
import torch.optim.lr_scheduler as scheduler_factory
import torch.optim as optim_factory

import torch.backends.cudnn as cudnn

from torchvision.transforms import ToTensor, Compose
import random
import os
from utils.utils import save_config, ensure_path, delete_path
from datetime import datetime
import pprint
import time

pp = pprint.PrettyPrinter(width=41, compact=True)


def get_model(config):
    model_name = list(config['net'].keys())[0]
    model_config = config['net'][model_name]
    model = getattr(model_factory, model_name)
    return model(model_config), model_name


def get_scheduler(optimizer, config):
    if 'scheduler' not in config:
        return None
    scheduler_name = config['scheduler']['name']
    if scheduler_name == 'none':
        return None
    scheduler = getattr(scheduler_factory, scheduler_name)
    scheduler_config = {k: v for k, v in config['scheduler'].items() if k != 'name'}
    return scheduler(optimizer, **scheduler_config)


def get_optimizer(params, config):
    optimizer_name = config['optimizer']['name']
    optimizer = getattr(optim_factory, optimizer_name)
    optimizer_config = {k: v for k, v in config['optimizer'].items() if k != 'name'}
    return optimizer(params, **optimizer_config)


def get_trainer(config):
    trainer_name = list(config['trainer'].keys())[0]
    trainer = getattr(trainer_factory, trainer_name)
    trainer_config = config['trainer'][trainer_name]

    return trainer, trainer_config


def get_dataset_cls(config):
    name = config['name']
    dataset_cls = getattr(data_factory, name)
    return dataset_cls


def get_dataloader(config, transform):
    dataset_cls = get_dataset_cls(config)
    dataset = dataset_cls(config, transform=transform)
    dataloader = dataset.get_dataloader(batch_size=config['batch_size'],
                                        shuffle=config['shuffle'],
                                        num_workers=config['num_workers']
                                        )
    return dataloader


def load_config(config_path):
    config = yaml.load(open(config_path))
    return config

def merge_config(config, args):
    for key, value in args.items():
        config[key] = value
    return config

def main():
    transform = Compose([
        ToTensor(),
    ])
    args = create_parser()
    torch.manual_seed(args['seed'])
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpus']
    config = load_config(args['config_path'])
    config = merge_config(config, args)
    model, model_name = get_model(config)
    optimizer = get_optimizer(model.parameters(), config)
    scheduler = get_scheduler(optimizer, config)
    trainer, trainer_config = get_trainer(config)
    trainer_config['mode'] = config['task']
    pp.pprint(config)
    print(model)
    print('# model parameters:', sum(param.numel() for param in model.parameters()))
    if args['task'] == 'train':
        now = datetime.now().strftime("%m-%d-%H")
        save_dir = config['save_dir'] + '/' + now
        delete_path(save_dir)
        ensure_path(save_dir)
        save_config(config, os.path.join(save_dir, 'config.yaml'))
        trainer_config['log_dir'] = os.path.join(save_dir, trainer_config['log_dir'])
        trainer_config['ckpt_dir'] = os.path.join(save_dir, trainer_config['ckpt_dir'])
        trainer_config['output_dir'] = os.path.join(save_dir, trainer_config['output_dir'])
        ensure_path(trainer_config['log_dir'])
        ensure_path(trainer_config['ckpt_dir'])
        ensure_path(trainer_config['output_dir'])
        trainer = trainer(model, optimizer, scheduler, trainer_config)
        train_dataloader = get_dataloader(config['data']['train'], transform)
        # val_dataloader = None
        val_dataloader = get_dataloader(config['data']['val'], transform)
        test_dataloader = None
        trainer.train(train_dataloader, val_dataloader=val_dataloader, test_dataloader=test_dataloader)
    if args['task'] == 'eval':
        start = time.time()
        ensure_path(trainer_config['result_dir'])
        save_config(config, os.path.join(trainer_config['result_dir'], 'config.yaml'))
        ensure_path(trainer_config['output_dir'])
        test_dataloader = get_dataloader(config['data']['test'], transform)
        trainer = trainer(model, optimizer, scheduler, trainer_config)
        trainer.eval(test_dataloader)
        end = time.time()
        print("eval use total {} min(s)".format((end - start) / 60.0))


def create_parser():
    parser = argparse.ArgumentParser("video super resolution parameters")
    parser.add_argument("--config_path", type=str, help="pipeline config file path")
    parser.add_argument("--gpus", type=str, help="gpu ids seperate by comma")
    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument("--task", type=str, default="train", help="which task to do")
    args, _ = parser.parse_known_args()
    if args.seed is None:
        args.seed = random.randint(-1e10, 1e10)

    return vars(args)


if __name__ == '__main__':
    main()