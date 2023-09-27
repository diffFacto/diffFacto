import argparse 
import torch
from anchor_diff.datasets.evaluation_utils import compute_all_metrics 
from anchor_diff.datasets.dataset_utils import load_ply_withNormals
import numpy as np
from anchor_diff.utils.misc import fps
import pickle
import json
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--datadir',
        type=str) 
    args = parser.parse_args()
    data = pickle.load(open(args.datadir, 'rb'))
    print(data.keys())
    metrics = []
    pred = data['pred']
    pred_mask = data.get('pred_mask', None)
    ref = data['ref']
    for i in range(4):
        print(f"Part {i} --------------------------------")
        _pred = torch.from_numpy(pred[i]).cuda().float()
        _ref = torch.from_numpy(ref[i]).cuda().float()
        if pred_mask is not None:
            _pred_mask = pred_mask[i].cuda()
        metric = compute_all_metrics(_pred, _ref, 32, mask=_pred_mask)
        print(metric)
        metrics.append(metric)
        
    avg_m = {k:0 for k in metrics[0].keys()}
    weights = [0.311, 0.312, 0.306, 0.057]
    # weights = [0.2788225675, 0.2755519215, 0.2722812756, 0.1733442355]
    for i, m in enumerate(metrics):
        for k, v in m.items():
            avg_m[k] += v * weights[i]
    print(avg_m)