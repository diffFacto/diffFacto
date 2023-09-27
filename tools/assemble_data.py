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
from pointnet2_ops import pointnet2_utils


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
        '--prefix',
        type=str) 
    args = parser.parse_args()
    # import torch
    # data1 = torch.load("/orion/u/w4756677/diffusion/anchorDIff/weights/chair-separate-v2.pt")
    # data2 = torch.load("/orion/u/w4756677/diffusion/anchorDIff/weights/chair-separate.pt")
    # ratio = [700*4, 700*4, 690*4, 140*4]
    # data = torch.cat([data1, data2], dim=0)[:2800]
    # print(data.shape)
    # out_data = []
    # out_mask = []
    # for i in tqdm(range(data.shape[0])):
    #     _data = data[i][..., :3].cuda()
    #     _mask = data[i][..., 3].cuda()
    #     if i > ratio[3]:
    #         wo_arm = _mask != 3 
    #         _data = _data[wo_arm]
    #         _mask = _mask[wo_arm]
    #     if i > ratio[2]:
    #         wo_leg = _mask != 2
    #         _data = _data[wo_leg]
    #         _mask = _mask[wo_leg]
    #     _data, ids = fps(_data[None], 2048, True)
    #     _mask = pointnet2_utils.gather_operation(_mask.reshape(1, 1, -1).contiguous(), ids).reshape(2048)
    #     print(_data.shape, _mask.shape)
    #     out_data.append(_data[0].cpu().numpy())
    #     out_mask.append(_mask.cpu().numpy())
    # out_data = np.stack(out_data, axis=0)
    # out_mask = np.stack(out_mask, axis=0)
    # pickle.load(open())
    # print(out_data.shape, out_mask.shape)
    # pickle.dump(dict(pred=out_data, pred_seg_mask=out_mask), open("/orion/u/w4756677/diffusion/anchorDIff/weights/lion_part/lion_2800_chair.pkl", "wb"))
    # exit()
    # data1 = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/weights/lion_Chair_2048.pkl", 'rb'))
    # print(data.keys(), data1.keys())
    # n = data1['pred'].shape[0]
    # pickle.dump( dict(pred=np.concatenate([data1['pred'], data['pred'][:2800 - n]], axis=0), pred_seg_mask=np.concatenate([data1['pred_seg_mask'], data['pred_seg_mask'][:2800 - n]], axis=0)), open(f"/orion/u/w4756677/diffusion/anchorDIff/weights/lion_2800_chair.pkl", 'wb'))
    # exit()
    # data_out = []
    # for i in range(704):
    #     _data = data[i]
    #     perm = torch.randperm(8192)
    #     idx = perm[:2048]
    #     data_out.append(_data[idx])
    # data_out = torch.stack(data_out, dim=0)
    # print(data_out.shape)
    # pickle.dump(dict(pred=data_out[..., :3].numpy().astype(float), pred_seg_mask=data_out[..., 3].numpy().astype(int)), open("/orion/u/w4756677/diffusion/anchorDIff/weights/lion_chair_v2.pkl", 'wb'))
    # exit()
    # print(data['parts'].shape)
    # data = data['parts'].cpu().numpy()
    # np.save("/orion/u/w4756677/diffusion/anchorDIff/weights/pruned_part_2000_part_2.npy", data)
    # num = 2800 if args.cat == "Chair" else 1360
    data = pickle.load(open(args.datadir, 'rb'))
    print(data.keys())
    # print(data.keys())
    pred = [v[:700] for k, v in data.items() if f"pred_sample" in k][:4]
    pred = np.concatenate(pred, axis=0)
    # print(len(pred))
    # print(pred.shape)
    # del data['pred']
    mask = data['pred_seg_mask'][:700, None].repeat(4, axis=1).reshape(2800, 2048)

    print(pred.shape, mask.shape)
    pickle.dump(dict(pred=pred, pred_seg_mask=mask), open(f"/orion/u/w4756677/diffusion/anchorDIff/weights/variant_data/{args.prefix}.pkl", 'wb'))
    # print(pred.shape)
    exit()
    # pickle.dump( dict(pred=pred[:2800], pred_seg_mask=mask[:2800]), open(f"/orion/u/w4756677/diffusion/anchorDIff/weights/v2_dpm_{num}_{args.cat}.pkl", 'wb'))
    # exit()
    preds = [v for k, v in data.items() if 'pred' in k]
    # print(preds)
    print(len(preds))
    N = 704 if args.cat == 'Chair' else 340
    # factors = [(N, 1), (N, 2), (N, 3), (N, 4), (N, 5)]
    factors = [(N, 4)]
    # pred = torch.stack([-pred[..., 2], pred[..., 1], pred[..., 0]], dim=-1)
    
    for size, m in factors:
        _preds = [preds[p] for p in range(m)]
        _preds = [p[:size] for p in _preds]
        _mask = mask[None, :size].repeat(m, axis=0).reshape(-1, 2048)
        _preds = np.concatenate(_preds, axis=0)
        print(_preds.shape)
        pickle.dump( dict(pred=_preds, seg_mask_ref=_mask), open(f"/orion/u/w4756677/diffusion/anchorDIff/weights/624_n100_{size}_{m}.pkl", 'wb'))
                        
