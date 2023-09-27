import os
import os.path as osp
import torch
import torch.nn.functional as F 
from difffacto.utils.registry import DATASETS
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm
from .dataset_utils import pc_norm, shapenet_part_normal_cat_to_id, build_loss_funcs, DataLoaderWrapperOne, worker_init_fn, augment
from .evaluation_utils import EMD_CD, compute_all_metrics
import json
import pickle
import random

import os


@DATASETS.register_module()
def ShapeNetSegSuperSegment(batch_size,data_root, part='pn_aware', split='train', n_class=4, scale_mode='shape_unit', num_workers=0, distributed=False, augment=False, contrastive_learning=False, vertical_only=False, eval_mode='ae', shift_only=False, augment_attn=False, normalize_attn=False, augment_prob=0.5, global_shift_prob=0.0):
    assert not distributed
    dataset = _ShapeNetSegSuperSegment(data_root, split, part, scale_mode, n_class, augment=augment,contrastive_learning=contrastive_learning, vertical_only=vertical_only, eval_mode=eval_mode, shift_only=shift_only, augment_attn=augment_attn, normalize_attn=normalize_attn, augment_prob=augment_prob, global_shift_prob=global_shift_prob)
    loader = DataLoaderWrapperOne(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    return loader, None


class _ShapeNetSegSuperSegment(Dataset):
    def __init__(self, data_root, split, part='pn_aware',scale_mode='shape_unit', n_class=4, augment=False, contrastive_learning=False, vertical_only=False, eval_mode='ae', shift_only=False, augment_attn=False, normalize_attn=False, augment_prob=0.5, global_shift_prob=0.0):
        super().__init__()
        assert eval_mode in ['ae', 'gen']
        self.eval_mode = eval_mode
        self.segs_data = pickle.load(open(os.path.join(data_root, f"shapenet_pointcloud_{part}.pkl"), "rb"))
        self.attn_map = pickle.load(open(os.path.join(data_root, f"shapenet_label_{part}.pkl"), "rb"))
        self.scale_mode = scale_mode
        self.shift_only=shift_only
        self.augment_attn=augment_attn
        self.augment_prob=augment_prob
        self.normalize_attn=normalize_attn
        self.global_shift_prob=global_shift_prob
        self.n_class=n_class
        self.augment=augment
        self.contrastive_learning=contrastive_learning
        self.vertical_only=vertical_only
        self.get_stats()
        if split == 'test':
            assert not (augment or contrastive_learning)
        num_train = int(len(self.segs_data) * 0.8)
        if split == 'train':
            self.segs_data = self.segs_data[:num_train]
            self.attn_map = self.attn_map[:num_train]
        if split == 'test':
            self.segs_data = self.segs_data[num_train:]
            self.attn_map = self.attn_map[num_train:]
        print(self.segs_data.shape, self.attn_map.shape)
        
        
        
    def get_stats(self):
        self.stats = dict() 
        self.stats['mean'] = self.segs_data.reshape(-1, 3).mean(0)
        self.stats['std'] = self.segs_data.reshape(-1).std()
        part_means = [[] for _ in range(self.n_class)]
        # print(len(part_means))
        for i in range(self.segs_data.shape[0]):
            geos = self.segs_data[i]
            attn_map = self.attn_map[i]
            seg_mask = attn_map.argmax(1)
            # print(seg_mask.shape)
            # print(i)
            for j in range(self.n_class):
                part = geos[seg_mask == j]
                
                if part.shape[0] > 0:
                    # print(part.mean(0))
                    part_means[j].append(part.mean(0))
                    # print(part.mean(0))
                    # print(sum(part_means[j]))
        self.stats["part_means"] = np.stack([np.stack(part_means[i], axis=0).mean(0) for i in range(self.n_class)], axis=0)
        print("#####PART MEANS ######")
        print(self.stats["part_means"])
        print("#####PART MEANS ######")
            

    def __getitem__(self, idx):
        out_dict=dict()
        geos = self.segs_data[idx]
        geos, shift, scale = pc_norm(geos, self.scale_mode, to_torch=True, stats=self.stats)
        attn_map = torch.from_numpy(self.attn_map[idx])
        if random.random() < self.global_shift_prob:
            rand_shift = torch.rand(1, 3) - 0.5
            if self.vertical_only:
                rand_shift[:, [0,2]] = 0.
            geos = geos + rand_shift
            shift -= rand_shift / scale
        if self.normalize_attn:
            attn_map = F.softmax(attn_map, 1)
        seg_mask = attn_map.max(1)[1]
        if self.augment_attn:
            attn_map = attn_map + torch.randn_like(attn_map) * 0.2 - 0.1
            attn_map= attn_map.clamp(0., 1.)
        
        if self.augment:
            input, part_scale, part_shifts = augment(geos, seg_mask, self.n_class, vertical_only=self.vertical_only, shift_only=self.shift_only)
            out_dict['input'] = input
            out_dict['attn_map'] = attn_map
            out_dict['ref'] = geos
            out_dict['seg_mask'] = seg_mask
            out_dict['shift'] = shift 
            out_dict['scale'] = scale
            out_dict['part_scale'] = part_scale
            out_dict['part_shift'] = part_shifts
            

        elif self.contrastive_learning:
            if random.random() < self.augment_prob:
                pos_input = augment(geos, seg_mask, self.n_class, vertical_only=self.vertical_only)
            neg_id = random.randrange(0, len(self.segs_data))
            if neg_id == idx: 
                neg_id = (idx + 1) % len(self.segs_data)
            neg_input, neg_shift, neg_scale = pc_norm(self.segs_data[neg_id], self.scale_mode, to_torch=True, stats=self.stats)
            neg_attn_map = torch.from_numpy(self.attn_map[neg_id])

            combined_attn_map = torch.cat([attn_map.unsqueeze(0).expand(2, -1, -1), neg_attn_map.unsqueeze(0)], dim=0)
            combined_seg_mask = combined_attn_map.max(-1)[1]
            out_dict['shift'] = torch.stack([shift, shift, neg_shift], dim=0)
            out_dict['scale'] = torch.stack([scale, scale, neg_scale], dim=0)
            out_dict['attn_map'] = combined_attn_map
            out_dict['input'] = torch.stack([geos, pos_input, neg_input], dim=0)
            out_dict['ref'] = torch.stack([geos, geos, neg_input], dim=0)
            out_dict['seg_mask'] = combined_seg_mask
            return out_dict
        else:
            out_dict['input'] = geos
            out_dict['ref'] = geos
            out_dict['attn_map'] = attn_map
            out_dict['seg_mask'] = seg_mask
            out_dict['shift'] = shift 
            out_dict['scale'] = scale
            out_dict['part_scale'] = torch.ones([self.n_class, 3])
            out_dict['part_shift'] = torch.zeros([self.n_class, 3])
        
        out_dict['global_anchor_mean'] = (torch.from_numpy(self.stats["part_means"]) - shift) / scale

        return out_dict
    def __len__(self):
        return len(self.segs_data)

    def evaluate(self, results, save_num_batch, device):
        save_dict = dict()
        preds = []
        refs = []
        for idx, pred_dict in enumerate(tqdm(results)):
            shift = pred_dict['shift']
            scale = pred_dict['scale']
            del pred_dict['shift']
            del pred_dict['scale']
            if self.eval_mode == 'ae':
                pred = pred_dict['pred'] * scale + shift 
                ref = pred_dict['input_ref'] * scale + shift 
            preds.append(pred)
            refs.append(ref)
            
            pred_dict = {k: v * scale + shift if v.shape[-1] == 3 else v for k, v in pred_dict.items()}               
            if idx < save_num_batch:
                for k, v in pred_dict.items():
                    if k not in save_dict.keys():
                        save_dict[k] = []
                    save_dict[k].append(v.detach().cpu().numpy())
        preds = torch.cat(preds, dim=0).cuda()
        refs = torch.cat(refs, dim=0).cuda()
        print(preds.shape, refs.shape)
        if self.eval_mode == 'ae':
            metrics = EMD_CD(preds, refs, 32)
        elif self.eval_mode == 'gen':
            metrics = compute_all_metrics(preds, refs, 32)
        save_dict = {k: np.concatenate(v, axis=0) for k, v in save_dict.items()}
        return save_dict, metrics
