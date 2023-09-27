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
import h5py

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
    args = parser.parse_args()
    
    data = h5py.File("/orion/u/w4756677/datasets/partglot_data/subsampled_partglot_data_triplet.h5", 'r')
    distractors = data['distractor_pcds']
    targets = data['target_pcds']
    print(distractors.shape, targets.shape)
    print(data.keys())
    exit()
    
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
    seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
    # dir_to_ds = "/orion/u/w4756677/datasets/colasce_data_txt/02691156"
    # dirs = os.listdir(dir_to_ds)
    # parts = [[], [], [], []]
    # parts_fns = [[], [], [], []]
    # permed_parts = []
    # permed_parts_fns = []
    # for d in tqdm(dirs):
    #     print(d[:-4])        
    #     out = np.loadtxt(os.path.join(dir_to_ds, d)).astype(np.float32)
    #     # out[..., 3] -= 12
    #     mask = out[..., 3]
    #     for k in range(4):
    #         part_mask = mask == k
    #         if np.any(part_mask):
    #             part_pts = out[part_mask, :3]
    #             parts[k].append(part_pts)
    #             parts_fns[k].append(d[:-4])
    # for k in range(4):
    #     assert (len(parts[k]) == len(parts_fns[k]))
    #     print(len(parts[k]) )
    #     parts[k] = np.stack(parts[k], axis=0)
    # for k in range(4):
    #     part = parts[k]
    #     part_fn = parts_fns[k]
    #     l = part.shape[0]
    #     perm = np.random.permutation(np.arange(l))
    #     part_fn = [part_fn[i] for i in np.arange(l).tolist()]
    #     part = part[perm]
    #     assert part.shape[0] == l
    #     if l > 2000:
    #         part = part[:2000]
    #         part_fn = part_fn[:2000]
    #     permed_parts.append(part)
    #     permed_parts_fns.append(part_fn)
    # for i in range(4):
    #     print(permed_parts[i].shape, len(permed_parts_fns[i]))
    # pickle.dump(dict(parts=permed_parts, permed_parts_fns=permed_parts_fns), open("/orion/u/w4756677/diffusion/anchorDIff/weights/ref_part/permuted_parts_airplane.pkl", "wb"))
    # exit()
    part0 = np.load("/orion/u/w4756677/diffusion/anchorDIff/weights/sgf_part/mixing_chair_two/mixing0.npy")
    part1 = np.load("/orion/u/w4756677/diffusion/anchorDIff/weights/sgf_part/mixing_chair_two/mixing1.npy")
    part2 = np.load("/orion/u/w4756677/diffusion/anchorDIff/weights/sgf_part/mixing_chair_two/mixing2.npy")
    part3 = np.load("/orion/u/w4756677/diffusion/anchorDIff/weights/sgf_part/mixing_chair_two/mixing3.npy")
    mask = np.array([0, 1, 2, 3]).repeat(2048, axis=0)
    allpart = np.concatenate([part0, part1, part2, part3], axis=0).reshape(-1, 3)
    print(mask.shape, allpart.shape)
    np.save("/orion/u/w4756677/diffusion/anchorDIff/weights/sgf_part/mixing_chair_two/total.npy", np.concatenate([allpart, mask[...,None]], axis=-1))
    exit()
    from anchor_diff.utils.misc import fps
    from pointnet2_ops import pointnet2_utils
    sgf_leg_interp = torch.from_numpy(np.load("/orion/u/w4756677/diffusion/anchorDIff/weights/interpolation/sgf_leg_interp.npy"))
    sgf_leg_back = torch.from_numpy(np.load("/orion/u/w4756677/diffusion/anchorDIff/weights/interpolation/sgf_back_for_leg_interp.npy"))
    sgf_leg_seat = torch.from_numpy(np.load("/orion/u/w4756677/diffusion/anchorDIff/weights/interpolation/sgf_seat_for_leg_interp.npy"))
    sgf_leg_arm = torch.from_numpy(np.load("/orion/u/w4756677/diffusion/anchorDIff/weights/interpolation/sgf_arm_for_leg_interp.npy"))
    data = pickle.load(open("/orion/u/w4756677/diffusion/anchorDIff/weights/interpolation/leg_set.pkl", "rb"))
    set1 = torch.from_numpy(data['from']).cuda().float()
    out = []
    mask = []
    arm_id = 0
    for i in tqdm(range(set1.shape[0])):
        _out = []
        m = []
        _data1 = set1[i]
        pts1 = _data1[:, :3]
        mask1 = _data1[:, 3]
        em1 = mask1 == 3
        _out.append(sgf_leg_back[i])
        _out.append(sgf_leg_seat[i])
        m = [torch.zeros([2048]), torch.ones([2048])]
        if torch.any(em1): 
            _out.append(sgf_leg_arm[arm_id])
            arm_id += 1
            m.append(torch.ones([2048]) * 3)
        _out = torch.cat(_out, dim=1).repeat_interleave(10, dim=0)
        print(_out.shape, sgf_leg_interp.shape)
        _out = torch.cat([_out, sgf_leg_interp[i]], dim=1)
        # m.append(2)
        m.append(torch.ones([2048]) * 2)
        _mask = torch.cat(m).reshape(1, -1).repeat_interleave(10, dim=0)
        print(m)
        print(_mask.shape, _out.shape)
        print(_out.shape)
        if _out.shape[1] > 2048:
            _out, idx = fps(_out.cuda(), 2048, True)
            print(_out.shape)
            _mask = pointnet2_utils.gather_operation(_mask.reshape(10, 1, -1).contiguous().float().cuda(), idx.int()).view(10, 2048, 1)
            print(_out.shape, _mask.shape)
        _out = np.concatenate([_out.cpu().numpy(), _mask.cpu().numpy()], axis=-1)
        out.append(_out)
    out = np.stack(out, axis=0)
    print(out.shape)
    np.save("/orion/u/w4756677/diffusion/anchorDIff/weights/interpolation/sgf_interp.npy", out)
    exit()
    
    for p in path:
        data = np.loadtxt(fn[1]).astype(np.float32) # size 20488,7 read in this txt file, a total of 20488 points, each point xyz rgb + sub category label
        token = (os.path.splitext(os.path.basename(fn[1]))[0])
        data.append()
        
    data = pickle.load(open(args.datadir, 'rb'))
    
    # print(data.keys())
    pred = data['ref']
    # print(len(pred))
    # print(pred.shape)
    # del data['pred']
    mask = data['ref_seg_mask']
    out = np.concatenate([pred, mask[..., None]], axis=-1)
    # del data['pred_seg_mask']
    perm_out = np.random.permutation(out)
    print(out.shape, perm_out.shape)
    pickle.dump(dict(set1=out, set2=perm_out), open("/orion/u/w4756677/diffusion/anchorDIff/weights/interpolation/interpolation_set_airplanes.pkl", 'wb'))
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
                        
