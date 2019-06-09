#!/usr/bin/env bash
# -*- coding: utf-8 -*-
set -e
cd ..
config=$1
gpus=$2
exp_dir=$3
python train.py --config_path ${config} --gpus ${gpus} --exp_dir ${exp_dir} --task train
python train.py --config_path ${exp_dir}/config.yaml --gpus ${gpus} --exp_dir ${exp_dir} --task val
python train.py --config_path ${exp_dir}/config.yaml --gpus ${gpus} --task eval --exp_dir ${exp_dir}
cd ./scripts
bash generation.sh ../${exp_dir}/eval/generated ../${exp_dir}/eval