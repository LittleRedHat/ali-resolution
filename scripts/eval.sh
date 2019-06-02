#!/usr/bin/env python
cd ..
config=$1
gpus=$2
python train.py --config_path ./configs/${config} --gpus ${gpus} --task eval