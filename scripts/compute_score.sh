#!/usr/bin/env bash
# -*- coding: utf-8 -*-
cd ..
dir=$1
conda activate vmaf
python compute_metric.py --pred_image_dir ${dir}/generated --pred_video_dir ${dir}/submit --ref_image_dir ./data/round1/val --ref_video_dir ./data/round1/val_raw/high --output_dir ${dir}