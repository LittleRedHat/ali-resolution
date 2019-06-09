#!/usr/bin/env bash
# -*- coding: utf-8 -*-
cd ..
gpus=$1
exp_dir=$2
python train.py --config_path ${exp_dir}/config.yaml --gpus ${gpus} --exp_dir ${exp_dir} --task val