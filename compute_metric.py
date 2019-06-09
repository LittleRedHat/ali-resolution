#!/usr/bin/env bash
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
from utils.vmaf_utils import run_vmaf_in_batch
import os
import fnmatch
from PIL import Image
import numpy as np
import json


def psnr_fn(input, target, max_val=1.0, eps=1e-10):
    '''

    :param input: batch * channel * h * w
    :param target: batch * channel * h * w
    :param max_val: 1.0 or 255.0
    :return: psnr
    '''
    mse = np.square(input - target).mean(axis=(1, 2, 3))
    return 10 * np.log10(max_val ** 2 / (mse + eps)).mean()

class Metric:
    def __init__(self, config):
        self.config = config
        self.width = 480 * 4
        self.height = 270 * 4
        self.fmt = 'yuv420p'

    def compute_score(self):
        psnrs = self._compute_psnr()
        vmafs = self._compute_vmaf()
        mean_psnr = np.mean(list(psnrs.values()))
        mean_vmaf = np.mean(list(vmafs.values()))
        score = 0.8 * mean_psnr + 0.2 * mean_vmaf
        return score, psnrs, vmafs, mean_psnr, mean_vmaf

    def _read_image(self, file_path):
        im = Image.open(file_path).convert('RGB')
        im = np.array(im) / 255.0
        return im

    def _compute_psnr(self):
        predictions = []
        truths = []
        pred_tracks = sorted(fnmatch.filter(os.listdir(self.config['pred_image_dir']), 'Youku*'))
        ref_tracks = sorted(fnmatch.filter(os.listdir(self.config['ref_image_dir']), 'Youku*h_GT'))
        ref_tracks_map = {track[:11]: track for track in ref_tracks}
        pred_track_map = {track[:11]: track for track in pred_tracks}
        for track, pred_path in pred_track_map.items():
            ref_path = ref_tracks_map[track]
            frames = fnmatch.filter(os.listdir(os.path.join(self.config['pred_image_dir'], pred_path)), '*.bmp')
            for frame in frames:
                frame_id = int(os.path.splitext(frame)[0])
                predictions.append(os.path.join(self.config['pred_image_dir'], pred_path, '{:03d}.bmp'.format(frame_id)))
                truths.append(os.path.join(self.config['ref_image_dir'], ref_path, '{:03d}.bmp'.format(frame_id)))
        psnrs = {}
        for index in range(len(predictions)):
            p_image = self._read_image(predictions[index])
            p_image = np.expand_dims(p_image, axis=0)
            r_image = self._read_image(truths[index])
            r_image = np.expand_dims(r_image, axis=0)
            try:
                psnr_score = psnr_fn(p_image, r_image)
                key = predictions[index].split('/')[-2] + '/' + predictions[index].split('/')[-1]
                print(key, psnr_score)
                psnrs[key] = psnr_score
            except Exception as e:
                print(predictions[index], truths[index])
        psnrs = {key: psnrs[key] for key in sorted(psnrs.keys())}
        return psnrs

    def _compute_vmaf(self):
        predictions = []
        truths = []
        fmts = []
        widths = []
        heights = []
        tracks = self.config['vmaf_video_ids']
        ref_tracks = sorted(fnmatch.filter(os.listdir(self.config['ref_video_dir']), '*.y4m'))
        ref_tracks_map = {track[:11]:track for track in ref_tracks}

        pred_tracks = sorted(fnmatch.filter(os.listdir(self.config['pred_video_dir']), '*.y4m'))
        pred_tracks_map = {track[:11]: track for track in pred_tracks}
        for track in tracks:
            pred_file = os.path.join(self.config['pred_video_dir'], pred_tracks_map[track])
            ref_file = os.path.join(self.config['ref_video_dir'], ref_tracks_map[track])
            predictions.append(pred_file)
            truths.append(ref_file)
            widths.append(self.width)
            heights.append(self.height)
            fmts.append(self.fmt)
        vmaf_config = {'ref_files': truths, 'dis_files': predictions, 'fmts': fmts, 'widths': widths,
                       'heights': heights, 'parallelize': True, 'ci': '', 'model_path': '/home/xxc/workspace/vmaf/model/vmaf_4k_v0.6.1.pkl', 'pool_method': 'perc20'}
        scores = run_vmaf_in_batch(vmaf_config)
        scores = {predictions[i].split('/')[-1]: item['aggregate']['VMAF_score'] for i, item in enumerate(scores)}
        return scores


def parse_args():
    parser = argparse.ArgumentParser("metric computation parser")
    parser.add_argument('--pred_image_dir', type=str)
    parser.add_argument('--pred_video_dir', type=str)
    parser.add_argument('--ref_image_dir', type=str)
    parser.add_argument('--ref_video_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    return vars(args)


def main():
    args = parse_args()
    vmaf_video_ids = ['Youku_00194', 'Youku_00164', 'Youku_00189', 'Youku_00198', 'Youku_00155']
    args['vmaf_video_ids'] = vmaf_video_ids
    metric = Metric(args)
    score, psnr, vmaf, mean_psnr, mean_vmaf = metric.compute_score()
    result = {'score': score, 'psnr':psnr, 'vmaf':vmaf, 'mean_psnr': mean_psnr, 'mean_vmaf': mean_vmaf}
    with open(os.path.join(args['output_dir'], 'val_result.json'), 'w') as writer:
        json.dump(result, writer)


if __name__ == '__main__':
    main()
