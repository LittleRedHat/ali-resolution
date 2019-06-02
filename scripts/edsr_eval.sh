#!/usr/bin/env bash
cd ..
python train.py --config_path ./configs/edsr_4x_eval.yaml --gpus 2 --task eval