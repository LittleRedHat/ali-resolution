#!/usr/bin/env bash
gpus=$1
exp_dir=$2
./val.sh $gpus $exp_dir
#./val_generation.sh $exp_dir/val/generated $exp_dir/val 0
./compute_score.sh ${exp_dir}/val
