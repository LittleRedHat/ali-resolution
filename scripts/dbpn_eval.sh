#!/usr/bin/env bash
cd ..
config=$1
gpus=$2
python train.py --config_path ./configs/${config}.yaml --gpus ${gpus} --task eval