import os
import os.path as osp
import torch
from difffacto.utils.registry import DATASETS
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm
from .dataset_utils import pc_norm, shapenet_part_normal_cat_to_id, build_loss_funcs, DataLoaderWrapperOne, worker_init_fn, augment
import json
import pickle
import random
import os


@DATASETS.register_module()
def ShapeNetSegSuperSegmentParglot(batch_size,data_root, part='pn_aware', split='train', n_class=4, scale_mode='shape_unit', num_workers=0, distributed=False, augment=False, vertical_only=False):
    assert not distributed
    dataset = _ShapeNetSegSuperSegmentPartglot(data_root, split, part, scale_mode, n_class, augment=augment, vertical_only=vertical_only)
    loader = DataLoaderWrapperOne(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    return loader, None


class _ShapeNetSegSuperSegmentPartglot(Dataset):
    def __init__(self, data_root, split, part='pn_aware',scale_mode='shape_unit', n_class=4, augment=False, vertical_only=False):
        super().__init__()
        data = pickle.load(open(os.path.join(data_root, f"partglot_shapenet_seg_out_{part}_raw.pkl"), "rb"))
        self.segs_data = data["geos"].astype(np.float32)
        self.geo_mask = data["geos_masks"].astype(np.float32)
        self.attn_map = data['attn_maps'].astype(np.float32)
        self.scale_mode = scale_mode
        self.n_class=n_class
        self.augment=augment
        self.vertical_only=vertical_only
        if split == 'test':
            assert not augment
        num_train = int(len(self.segs_data) * 0.8)
        if split == 'train':
            self.segs_data = self.segs_data[:num_train]
            self.attn_map = self.attn_map[:num_train]
        if split == 'test':
            self.segs_data = self.segs_data[num_train:]
            self.attn_map = self.attn_map[num_train:]
        print(self.segs_data.shape, self.attn_map.shape)
        self.get_stats()
        
        
    def get_stats(self):
        self.stats = dict() 
        self.stats['mean'] = self.segs_data.reshape(-1, 3).mean(0)
        self.stats['std'] = self.segs_data.reshape(-1).std()

    def __getitem__(self, idx):
        out_dict=dict()
        geos = self.segs_data[idx]
        npoint = geos.shape[1]
        geo_mask = self.geo_mask[idx]
        part_geos, shift, scale = pc_norm(geos[geo_mask == 1].reshape(-1, 3), self.scale_mode, to_torch=True, stats=self.stats)
        geos[geo_mask == 1] = part_geos.reshape(-1, npoint, 3)
        attn_map = torch.from_numpy(self.attn_map[idx])
        seg_mask = attn_map.max(0)[1]
        
        if self.augment:
            # not working
            input = augment(geos, seg_mask, self.n_class, vertical_only=self.vertical_only)
            out_dict['input'] = input
            out_dict['attn_map'] = attn_map
            out_dict['ref'] = geos
            out_dict['seg_mask'] = seg_mask
            out_dict['shift'] = shift 
            out_dict['scale'] = scale
        #else:
        out_dict['input'] = geos
        out_dict['ref'] = geos
        out_dict['geo_mask'] = geo_mask
        out_dict['attn_map'] = attn_map
        out_dict['seg_mask'] = seg_mask
        out_dict['shift'] = shift 
        out_dict['scale'] = scale
        


        return out_dict
    def __len__(self):
        return len(self.segs_data)

    def evaluate(self, results, save_num_batch, device):
        save_dict = dict()
        fine_metrics = {'metric':0.}
        for idx, pred_dict in enumerate(tqdm(results)):
            shift = pred_dict['shift'][11].to(device)
            scale = pred_dict['scale'][11].to(device)
            del pred_dict['shift']
            del pred_dict['scale']
            for k, v in pred_dict.items():
                if v.shape[-1] == 3:
                    v = v.to(device) * scale.unsqueeze(0) + shift.unsqueeze(0)
            if idx < save_num_batch:
                for k, v in pred_dict.items():
                    if k not in save_dict.keys():
                        save_dict[k] = []
                    save_dict[k].append(v.detach().cpu().numpy())
            
        save_dict = {k: np.concatenate(v, axis=0) for k, v in save_dict.items()}
        return save_dict, fine_metrics
