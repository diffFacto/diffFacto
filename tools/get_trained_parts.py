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
        '--cat',
        type=str) 
    args = parser.parse_args()
    cat_id = shapenet_part_normal_cat_to_id[args.cat]
    
    ref_part_pointcloud = [[], [], [], []]
    root = '/orion/u/w4756677/datasets/coalace_data_processed' if args.cat != 'Car' else '/orion/u/w4756677/datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal'
    with open(f"{root}/train_test_split/shuffled_test_file_list.json", 'r') as f:
        test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
    with open(os.path.join(root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
        train_ids = set([str(d.split('/')[2]) for d in json.load(f)]) # '928c86eabc0be624c2bf2dc31ba1713' this is the first value
    with open(os.path.join(root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
        val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
    fns = sorted(os.listdir(f"{root}/{cat_id}"))  # Get each txt file under the folder according to the path and put it in the list
    if args.cat != 'Car':
        densePoints = [f for f in fns if (f[-7:] == 'ori.ply')]
        labels = [f for f in fns if (f[-10:] == 'ori.labels')]
        
        labels = [os.path.join(f"{root}/{cat_id}", fn) for fn in labels if ((fn[0:-11] in train_ids) or (fn[0:-11] in val_ids))]
        densePoints = [os.path.join(f"{root}/{cat_id}", fn) for fn in densePoints if ((fn[0:-8] in train_ids) or (fn[0:-8] in val_ids))]
        print(len(labels), len(densePoints))
    else:
        densePoints = [f for f in fns if (f[-3:] == 'txt')]
        densePoints = [os.path.join(f"{root}/{cat_id}", fn) for fn in densePoints if ((fn[0:-4] in test_ids) )]
        print(len(densePoints))
        
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
                    if ref_points.shape[1] > 2048:
                        ref_points = fps(ref_points, 2048)
                    ref_part_pointcloud[j].append(ref_points)
            else:
                if torch.any(ref_idx):
                    ref_points = _ref[ref_idx].unsqueeze(0)
                    if ref_points.shape[1] > 100:
                        while ref_points.shape[1] < 2048:
                            ref_points = torch.cat([ref_points, ref_points], dim=1)
                        if ref_points.shape[1] > 2048:
                            ref_points = fps(ref_points, 2048)
                        ref_part_pointcloud[j].append(ref_points)
            
    print(len(ref_part_pointcloud))
    ref_part_pointcloud =  [torch.cat(ll, dim=0) for ll in ref_part_pointcloud]
    for s in ref_part_pointcloud:
        print(s.shape)
    print('Normalizing------------------')
    for i in range(4):
        _ref = ref_part_pointcloud[i]
        _ref_shift = _ref.mean(1).reshape(-1, 1, 3)
        _ref_scale = _ref.std(1).reshape(-1, 1, 3)
        _ref = (_ref - _ref_shift) / _ref_scale
        ref_part_pointcloud[i] = _ref.detach().cpu().numpy()
    print(f'saved at /orion/u/w4756677/diffusion/anchorDIff/weights/ref_part_{args.cat}.pkl')
    pickle.dump(dict(ref=ref_part_pointcloud), open(f"/orion/u/w4756677/diffusion/anchorDIff/weights/ref_part_{args.cat}.pkl", "wb"))
