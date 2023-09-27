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
from anchor_diff.datasets.evaluation_utils import compute_bbox_metric 


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
        '--cov_thresh',
        type=int,
        default=4) 
    parser.add_argument(
        '--thresh',
        type=int,
        default=95) 
    parser.add_argument(
        '--chamfer',
        action='store_true') 
    parser.add_argument(
        '--no_nn',
        action='store_true') 
    args = parser.parse_args()
    thresh = args.thresh / 100
    cov_thresh = [0.03, 0.04, 0.05, 0.06, 100][args.cov_thresh]
    data = pickle.load(open(args.data_dir, 'rb'))
    refs = pickle.load(open(args.ref_dir, 'rb'))
    cat_id = shapenet_part_normal_cat_to_id[args.cat]
    print(data.keys())
    assert 'pred' in data.keys() and ('pred_seg_mask' in data.keys() or 'seg_mask_ref' in data.keys())
    pred = torch.from_numpy(data['pred']).cuda()[:2800]
    ref = torch.from_numpy(refs['ref']).cuda()
    # pred = torch.stack([-pred[..., 2], pred[..., 1], pred[..., 0]], dim=-1)
    pred_seg_mask = data.get('pred_seg_mask', None)
    ref_seg_mask = refs.get('ref_seg_mask', None)
    if pred_seg_mask is None:
        pred_seg_mask = data['seg_mask_ref']
    pred_seg_mask = torch.from_numpy(pred_seg_mask).cuda()[:2800]
    ref_seg_mask = torch.from_numpy(ref_seg_mask).cuda()
    metric = 'chamfer' if args.chamfer else 'iou'
    bbox_metrics = compute_bbox_metric(pred, pred_seg_mask, ref, ref_seg_mask, 32, n_class=4, thresh=thresh, metric=metric, no_nn=args.no_nn, cov_thresh=cov_thresh)
    for k, v in bbox_metrics.items():
        print(f"{k}: {v}")
    with open(os.path.join(args.save_dir, f"{args.prefix}_{args.cat}_robust_thresh{args.thresh}_cov_thresh_{args.cov_thresh}.txt"), 'w') as f:
        for k, v in bbox_metrics.items():
            f.write(f"{k}: {v}")
            f.write('\n')
    