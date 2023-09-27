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
    cat_id = shapenet_part_normal_cat_to_id[args.cat]
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
        
    ref = []
        
    # pred_params = torch.cat(pred_params,dim=0)
    # pickle.dump(dict(pred=torch.cat(pred_params,dim=0).cpu().numpy(), ref=pred.cpu().numpy()), open("/orion/u/w4756677/diffusion/anchorDIff/weights/out.pkl", 'wb'))
    refs_out = []
    masks_out = []
    for i in tqdm(range(len(densePoints))):
        if args.cat != 'Car':
            _ref = torch.from_numpy(load_ply_withNormals(densePoints[i]).astype(np.float32)).cuda()
            _ref_seg_mask = torch.from_numpy(np.loadtxt(labels[i]).astype(np.int32)).cuda() - 1
        else:
            data = torch.from_numpy(np.loadtxt(densePoints[i]).astype(np.float32)).cuda()
            _ref_seg_mask = data[:, -1].to(torch.int32) - seg_classes[args.cat][0] # Get the label of the sub category
            _ref = data[:, 0:3]
            
        
        ref_out = torch.zeros(4, 512, 3)
        mask_out = torch.zeros(4, 512)
        valid_id = [0 for _ in range(4)]
        for j in range(4):
            ref_idx = _ref_seg_mask == j
            ref_points = _ref[ref_idx]
            if ref_points.abs().sum() != 0:
                valid_id[j] = 1
                ref_out[j] = ref_points[:512]
                mask_out[j] = j
        valid_part = torch.tensor(valid_id).argmax()
        valid_point = ref_out[valid_part]
        for j in range(4):
            if valid_id[j] == 0:
                mask_out[j] = valid_part
                ref_out[j] = valid_point
        ref_out = ref_out.reshape(2048, 3)
        mask_out = mask_out.reshape(2048)
        ref_out_max = ref_out.max(0)[0].reshape(1,3) # (1, 3)
        ref_out_min = ref_out.min(0)[0].reshape(1,3) # (1, 3)
        ref_out_shift = ((ref_out_min + ref_out_max) / 2).reshape(1, 3)
        ref_out_scale = (ref_out_max - ref_out_min).max(-1)[0].reshape(1, 1) / 2
        ref_out = (ref_out - ref_out_shift) / ref_out_scale
        refs_out.append(ref_out)
        masks_out.append(mask_out)
    
    refs_out = torch.cat(refs_out, dim=0).reshape(-1, 2048, 3)
    masks_out = torch.cat(masks_out, dim=0).reshape(-1, 2048).int()
    print(refs_out.shape[0], masks_out.shape[0])
    assert (masks_out.shape[0] == refs_out.shape[0])
    pickle.dump(dict(ref=refs_out.cpu().numpy(), ref_seg_mask=masks_out.cpu().numpy()), open("/orion/u/w4756677/diffusion/anchorDIff/weights/out_ref_airplane.pkl", 'wb'))
            
    # filename = os.path.basename(args.datadir)[:-4]
    # print(f'saved at /orion/u/w4756677/diffusion/anchorDIff/weights/{filename}_param_{args.cat}.pkl')
    # pickle.dump(dict(ref=ref_params, pred=pred_params, metric=metric), open(f"/orion/u/w4756677/diffusion/anchorDIff/weights/{filename}_param_{args.cat}.pkl", "wb"))
