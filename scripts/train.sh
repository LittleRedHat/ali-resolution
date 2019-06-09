#!/usr/bin/env bash
cd ..
config=$1
gpus=$2
exp_dir=$3
python train.py --config_path ${config} --gpus ${gpus} --task train --exp_dir ${exp_dir}