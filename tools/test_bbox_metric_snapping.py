import argparse 
import torch
import os
from anchor_diff.datasets.evaluation_utils import _pairwise_EMD_CD_ 
from anchor_diff.datasets.dataset_utils import load_ply_withNormals, shapenet_part_normal_cat_to_id
import numpy as np
from anchor_diff.utils.misc import fps
import pickle
import json
import os
from tqdm import tqdm
from anchor_diff.datasets.evaluation_utils import compute_snapping_metric 


seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        type=str) 
    parser.add_argument(
        '--ref_dir',
        type=str) 
    parser.add_argument(
        '--save_dir',
        type=str
    )
    parser.add_argument(
        '--prefix',
        type=str
    )
    parser.add_argument(
        '--cat',
        type=str) 
    parser.add_argument(
        '--num_points_sample',
        type=int,
        default=2048) 
    parser.add_argument(
        '--num_points_compute',
        type=int,
        default=2048) 
    parser.add_argument(
        '--bs',
        type=int,
        default=32) 
    parser.add_argument(
        '--thresh',
        type=int,
        default=95) 
    args = parser.parse_args()
    thresh = args.thresh / 100
    data = pickle.load(open(args.data_dir, 'rb'))
    refs = pickle.load(open(args.ref_dir, 'rb'))
    cat_id = shapenet_part_normal_cat_to_id[args.cat]
    print(data.keys())
    assert 'pred' in data.keys() and ('pred_seg_mask' in data.keys() or 'seg_mask_ref' in data.keys())
    
    pred = torch.from_numpy(data['pred']).cuda()
    ref = torch.from_numpy(refs['ref']).cuda()
    # pred = torch.stack([-pred[..., 2], pred[..., 1], pred[..., 0]], dim=-1)
    pred_seg_mask = data.get('pred_seg_mask', None)
    ref_seg_mask = refs.get('ref_seg_mask', None)
    if pred_seg_mask is None:
        pred_seg_mask = data['seg_mask_ref']
    pred_seg_mask = torch.from_numpy(pred_seg_mask).cuda()
    ref_seg_mask = torch.from_numpy(ref_seg_mask).cuda()
    
    print("Normalizing---------")
    pred_max = pred.max(1)[0].reshape(-1, 1, 3) # (1, 3)
    pred_min = pred.min(1)[0].reshape(-1, 1, 3) # (1, 3)
    pred_shift = ((pred_min + pred_max) / 2).reshape(-1, 1, 3)
    pred_scale = (pred_max - pred_min).max(-1)[0].reshape(-1, 1, 1) / 2
    pred = (pred - pred_shift) / pred_scale
    ref_max = ref.max(1)[0].reshape(-1, 1, 3) # (1, 3)
    ref_min = ref.min(1)[0].reshape(-1, 1, 3) # (1, 3)
    ref_shift = ((ref_min + ref_max) / 2).reshape(-1, 1, 3)
    ref_scale = (ref_max - ref_min).max(-1)[0].reshape(-1, 1, 1) / 2
    ref = (ref - ref_shift) / ref_scale
    
    snapping_metrics = compute_snapping_metric(pred, pred_seg_mask, cls=args.cat)
    oracle_snapping_metrics = compute_snapping_metric(ref, ref_seg_mask, cls=args.cat)
    for k, v in snapping_metrics.items():
        print(f"{k}: {v}")
        print(f"oracle_{k}: {oracle_snapping_metrics[k]}")
    with open(os.path.join(args.save_dir, f"{args.prefix}_{args.cat}_snapping.txt"), 'w') as f:
        for k, v in snapping_metrics.items():
            f.write(f"{k}: {v}")
            f.write('\n')
            f.write(f"oracle_{k}: {oracle_snapping_metrics[k]}")
            f.write('\n')
    