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
        '--canonical',
        action='store_true'
    )
    parser.add_argument(
        '--npoint',
        type=int,
        default=2048
    )
    args = parser.parse_args()
    data = dict(np.load(args.datadir))
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
    pred_part_mask = [[], [], [], []]
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
    for i in tqdm(range(size)):
        _pred = pred[i]
        _pred_seg_mask = pred_seg_mask[i]
        for j in range(4):
            idx = _pred_seg_mask == j 
            if torch.any(idx):
                points = _pred[idx].unsqueeze(0)
                if points.shape[1] > 100:
                    mask = torch.ones([1, args.npoint])
                    if points.shape[1] < args.npoint:
                        mask[:, points.shape[1]:] = 0
                    while points.shape[1] < args.npoint:
                        points = torch.cat([points, points], dim=1)
                    if points.shape[1] > args.npoint:
                        points = points[:, :args.npoint]
                    pred_part_pointcloud[j].append(points)
                    assert (mask.shape == (1, args.npoint))
                    pred_part_mask[j].append(mask)
    for i in tqdm(range(len(densePoints))):
        if args.cat != 'Car':
            _ref = torch.from_numpy(load_ply_withNormals(densePoints[i]).astype(np.float32)).cuda()
            _ref_seg_mask = torch.from_numpy(np.loadtxt(labels[i]).astype(np.int32)).cuda() - 1
        else:
            data = torch.from_numpy(np.loadtxt(densePoints[i]).astype(np.float32)).cuda()
            _ref_seg_mask = data[:, -1].to(torch.int32) - seg_classes[args.cat][0] # Get the label of the sub category
            _ref = data[:, 0:3]
            
        for j in range(4):
            ref_idx = _ref_seg_mask == j
            if args.cat != "Car":
                ref_points = _ref[ref_idx].unsqueeze(0)
                if ref_points.abs().sum() != 0:
                    if ref_points.shape[1] > args.npoint:
                        ref_points = fps(ref_points, args.npoint)
                    ref_part_pointcloud[j].append(ref_points)
            else:
                if torch.any(ref_idx):
                    ref_points = _ref[ref_idx].unsqueeze(0)
                    if ref_points.shape[1] > 100:
                        while ref_points.shape[1] < args.npoint:
                            ref_points = torch.cat([ref_points, ref_points], dim=1)
                        if ref_points.shape[1] > args.npoint:
                            ref_points = fps(ref_points, args.npoint)
                        ref_part_pointcloud[j].append(ref_points)
            
    print(len(pred_part_pointcloud), len(ref_part_pointcloud), len(pred_part_mask))
    pred_part_pointcloud, ref_part_pointcloud, pred_part_mask = map(lambda l: [torch.cat(ll, dim=0) for ll in l], [pred_part_pointcloud, ref_part_pointcloud, pred_part_mask])
    for l, s, k in zip(pred_part_pointcloud, ref_part_pointcloud, pred_part_mask):
        print(s.shape, l.shape, k.shape)
    print('Normalizing------------------')
    if args.canonical:
        for i in range(4):
            _pred = pred_part_pointcloud[i]
            _pred_max = _pred.max(1)[0].reshape(-1, 1,3) # (1, 3)
            _pred_min = _pred.min(1)[0].reshape(-1, 1,3) # (1, 3)
            _pred_shift = ((_pred_min + _pred_max) / 2).reshape(-1, 1, 3)
            _pred_scale = (_pred_max - _pred_min).reshape(-1, 1, 3) / 2
            _pred = (_pred - _pred_shift) / _pred_scale
            pred_part_pointcloud[i] = _pred.detach().cpu().numpy()
            print(_pred.max(), _pred.min())
            _ref = ref_part_pointcloud[i]
            _ref_max = _ref.max(1)[0].reshape(-1, 1,3) # (1, 3)
            _ref_min = _ref.min(1)[0].reshape(-1, 1,3) # (1, 3)
            _ref_shift = ((_ref_min + _ref_max) / 2).reshape(-1, 1, 3)
            _ref_scale = (_ref_max - _ref_min).reshape(-1, 1, 3) / 2
            _ref = (_ref - _ref_shift) / _ref_scale
            ref_part_pointcloud[i] = _ref.detach().cpu().numpy()
    else:
        for i in range(4):
            _pred = pred_part_pointcloud[i]
            _pred_max = _pred.max(1)[0].reshape(-1, 1,3) # (1, 3)
            _pred_min = _pred.min(1)[0].reshape(-1, 1,3) # (1, 3)
            _pred_shift = ((_pred_min + _pred_max) / 2).reshape(-1, 1, 3)
            _pred_scale = (_pred_max - _pred_min).max(-1)[0].reshape(-1, 1, 1) / 2
            _pred = (_pred - _pred_shift) / _pred_scale
            pred_part_pointcloud[i] = _pred.detach().cpu().numpy()
            _ref = ref_part_pointcloud[i]
            _ref_max = _ref.max(1)[0].reshape(-1, 1,3) # (1, 3)
            _ref_min = _ref.min(1)[0].reshape(-1, 1,3) # (1, 3)
            _ref_shift = ((_ref_min + _ref_max) / 2).reshape(-1, 1, 3)
            _ref_scale = (_ref_max - _ref_min).max(-1)[0].reshape(-1, 1, 1) / 2
            _ref = (_ref - _ref_shift) / _ref_scale
            ref_part_pointcloud[i] = _ref.detach().cpu().numpy()
        
    filename = os.path.basename(args.datadir)[:-4]
    s = 'uncan' if not args.canonical else 'can'
    print(f'saved at /orion/u/w4756677/diffusion/anchorDIff/weights/{filename}_part_{args.cat}_{s}.pkl')
    pickle.dump(dict(ref=ref_part_pointcloud, pred=pred_part_pointcloud, pred_mask=pred_part_mask), open(f"/orion/u/w4756677/diffusion/anchorDIff/weights/{filename}_part_{args.cat}_{s}.pkl", "wb"))
