#!/usr/bin/env bash
cd ..
python train.py --config_path ./configs/dbpn_4x_eval.yaml --gpus 4 --task eval