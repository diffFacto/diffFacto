import argparse 
import torch
from anchor_diff.datasets.evaluation_utils import _pairwise_EMD_CD_ 
from anchor_diff.datasets.dataset_utils import load_ply_withNormals, shapenet_part_normal_cat_to_id
import numpy as np
from anchor_diff.utils.misc import fps
import pickle
import json
import os
from tqdm import tqdm
from anchor_diff.datasets.evaluation_utils import compute_all_metrics 


seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--datadir',
        type=str) 
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
    
    args = parser.parse_args()
    data = pickle.load(open(args.datadir, 'rb'))
    cat_id = shapenet_part_normal_cat_to_id[args.cat]
    print(data.keys())
    assert 'pred' in data.keys() and ('pred_seg_mask' in data.keys() or 'seg_mask_ref' in data.keys())
    pred = torch.from_numpy(data['pred']).cuda()
    # pred = torch.stack([-pred[..., 2], pred[..., 1], pred[..., 0]], dim=-1)
    pred_seg_mask = data.get('pred_seg_mask', None)
    if pred_seg_mask is None:
        pred_seg_mask = data['seg_mask_ref']
    pred_seg_mask = torch.from_numpy(pred_seg_mask).cuda()
    size = pred.shape[0]
    pred_part_pointcloud = [[], [], [], []]
    ref_part_pointcloud = [[], [], [], []]
    root = '/orion/u/w4756677/datasets/coalace_data_processed' if args.cat != 'Car' else '/orion/u/w4756677/datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal'
    with open(f"{root}/train_test_split/shuffled_test_file_list.json", 'r') as f:
        test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
    
    fns = sorted(os.listdir(f"{root}/{cat_id}"))  # Get each txt file under the folder according to the path and put it in the list
    if args.cat != 'Car':
        densePoints = [f for f in fns if (f[-7:] == 'ori.ply')]
        labels = [f for f in fns if (f[-10:] == 'ori.labels')]
        
        labels = [os.path.join(f"{root}/{cat_id}", fn) for fn in labels if ((fn[0:-11] in test_ids))]
        densePoints = [os.path.join(f"{root}/{cat_id}", fn) for fn in densePoints if ((fn[0:-8] in test_ids) )]
        print(len(labels), len(densePoints))
    else:
        densePoints = [f for f in fns if (f[-3:] == 'txt')]
        densePoints = [os.path.join(f"{root}/{cat_id}", fn) for fn in densePoints if ((fn[0:-4] in test_ids) )]
        print(len(densePoints))
        
    print(size)
    pred_params = []
    ref_params = []
    for i in tqdm(range(size)):
        _pred = pred[i]
        _pred_seg_mask = pred_seg_mask[i]
            
        _pred_max = _pred.max(0)[0].reshape(1,3) # (1, 3)
        _pred_min = _pred.min(0)[0].reshape(1,3) # (1, 3)
        _pred_shift = ((_pred_min + _pred_max) / 2).reshape(1, 3)
        _pred_scale = (_pred_max - _pred_min).max(-1)[0].reshape(1, 1) / 2
        _pred = (_pred - _pred_shift) / _pred_scale
        one_pred_params = []
        for j in range(4):
            idx = _pred_seg_mask == j 
            if torch.any(idx):
                points = _pred[idx]
                if points.shape[0] > 100:
                    part_mean = (points.max(0, keepdim=True)[0] - points.min(0, keepdim=True)[0]) / 2.0
                    part_std = (points.max(0, keepdim=True)[0] - points.min(0, keepdim=True)[0]) / 2.0
                    one_pred_params.append(torch.rand(args.num_points_sample, 3).cuda() * part_std + part_mean)
        one_pred_params = torch.cat(one_pred_params, dim=0).unsqueeze(0)
        if one_pred_params.shape[1] > args.num_points_compute:
            one_pred_params = fps(one_pred_params, args.num_points_compute)
        pred_params.append(one_pred_params)
        
    for i in tqdm(range(len(densePoints))):
        if args.cat != 'Car':
            _ref = torch.from_numpy(load_ply_withNormals(densePoints[i]).astype(np.float32)).cuda()
            _ref_seg_mask = torch.from_numpy(np.loadtxt(labels[i]).astype(np.int32)).cuda() - 1
        else:
            data = torch.from_numpy(np.loadtxt(densePoints[i]).astype(np.float32)).cuda()
            _ref_seg_mask = data[:, -1].to(torch.int32) - seg_classes[args.cat][0] # Get the label of the sub category
            _ref = data[:, 0:3]
            
        _ref_max = _ref.max(0)[0].reshape(1,3) # (1, 3)
        _ref_min = _ref.min(0)[0].reshape(1,3) # (1, 3)
        _ref_shift = ((_ref_min + _ref_max) / 2).reshape(1, 3)
        _ref_scale = (_ref_max - _ref_min).max(-1)[0].reshape(1, 1) / 2
        _ref = (_ref - _ref_shift) / _ref_scale
        one_ref_params = []
        for j in range(4):
            ref_idx = _ref_seg_mask == j
            if args.cat != "Car":
                ref_points = _ref[ref_idx]
                if ref_points.abs().sum() != 0:
                    ref_part_mean = (ref_points.max(0, keepdim=True)[0] - ref_points.min(0, keepdim=True)[0]) / 2.0
                    ref_part_std = (ref_points.max(0, keepdim=True)[0] - ref_points.min(0, keepdim=True)[0]) / 2.0
                    one_ref_params.append(torch.rand(args.num_points_sample, 3).cuda() * ref_part_std + ref_part_mean)
            else:
                if torch.any(ref_idx):
                    ref_points = _ref[ref_idx]
                    if ref_points.shape[0] > 100:
                        ref_part_mean = (ref_points.max(0, keepdim=True)[0] - ref_points.min(0, keepdim=True)[0]) / 2.0
                        ref_part_std = (ref_points.max(0, keepdim=True)[0] - ref_points.min(0, keepdim=True)[0]) / 2.0
                        one_ref_params.append(torch.rand(args.num_points_sample, 3).cuda() * ref_part_std + ref_part_mean)
        one_ref_params = torch.cat(one_ref_params, dim=0).unsqueeze(0)
        if one_ref_params.shape[1] > args.num_points_compute:
            one_ref_params = fps(one_ref_params, args.num_points_compute)
        ref_params.append(one_ref_params)
    
    pred_params = torch.cat(pred_params,dim=0)
    ref_params = torch.cat(ref_params,dim=0)
            
    print(pred_params.shape, ref_params.shape)
    metric = compute_all_metrics(pred_params, ref_params, 32)
    print(metric)
    # filename = os.path.basename(args.datadir)[:-4]
    # print(f'saved at /orion/u/w4756677/diffusion/anchorDIff/weights/{filename}_param_{args.cat}.pkl')
    # pickle.dump(dict(ref=ref_params, pred=pred_params, metric=metric), open(f"/orion/u/w4756677/diffusion/anchorDIff/weights/{filename}_param_{args.cat}.pkl", "wb"))
