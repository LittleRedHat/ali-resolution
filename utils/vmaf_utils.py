#!/usr/bin/env python
import os
import sys
import re
import numpy as np
from vmaf.config import VmafConfig
from vmaf.core.asset import Asset
from vmaf.core.executor import run_executors_in_parallel
from vmaf.core.quality_runner import VmafQualityRunner
from vmaf.tools.misc import cmd_option_exists, get_cmd_option
from vmaf.tools.stats import ListStats
import scipy.linalg
import json


def run_vmaf_in_batch(config):
    ref_files = config.get('ref_files')
    dis_files = config.get('dis_files')
    widths = config.get('widths')
    heights = config.get('heights')
    fmts = config.get('fmts')
    enable_conf_interval = config.get('ci', False)
    parallelize = config.get('parallelize', True)
    model_path = config.get('model_path', None)
    phone_model = config.get('phone_model', False)
    pool_method = config.get('pool_method')

    assets = []
    for i in range(len(ref_files)):
        ref_file = ref_files[i]
        dis_file = dis_files[i]
        width = widths[i]
        height = heights[i]
        fmt = fmts[i]
        asset = Asset(dataset="cmd",
                      content_id=0,
                      asset_id=i,
                      workdir_root=VmafConfig.workdir_path(),
                      ref_path=ref_file,
                      dis_path=dis_file,
                      asset_dict={'width': width, 'height': height, 'yuv_type': fmt}
                      )
        assets.append(asset)
    if enable_conf_interval:
        from vmaf.core.quality_runner import BootstrapVmafQualityRunner
        runner_class = BootstrapVmafQualityRunner
    else:
        runner_class = VmafQualityRunner

    if model_path is None:
        optional_dict = None
    else:
        optional_dict = {'model_filepath': model_path}

    if phone_model:
        if optional_dict is None:
            optional_dict = {}
        optional_dict['enable_transform_score'] = True
    runner = runner_class(
        assets,
        None, fifo_mode=True,
        delete_workdir=True,
        result_store=None,
        optional_dict=optional_dict,
        optional_dict2=None,
    )
    runner.run(parallelize=parallelize)
    results = runner.results
    scores = []
    for result in results:
        # pooling
        if pool_method == 'harmonic_mean':
            result.set_score_aggregate_method(ListStats.harmonic_mean)
        elif pool_method == 'min':
            result.set_score_aggregate_method(np.min)
        elif pool_method == 'median':
            result.set_score_aggregate_method(np.median)
        elif pool_method == 'perc5':
            result.set_score_aggregate_method(ListStats.perc5)
        elif pool_method == 'perc10':
            result.set_score_aggregate_method(ListStats.perc10)
        elif pool_method == 'perc20':
            result.set_score_aggregate_method(ListStats.perc20)
        else:
            raise NotImplementedError
        scores.append(json.loads(result.to_json()))
    return scores








