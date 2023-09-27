import os
import torch
import torch.nn.functional as F
from difffacto.utils.registry import DATASETS
from difffacto.utils.misc import fps
from torch.utils.data import Dataset
import numpy as np
from tqdm.auto import tqdm
from .dataset_utils import pc_norm, shapenet_part_normal_cat_to_id, build_loss_funcs, DataLoaderWrapper, DataLoaderWrapperOne
from .evaluation_utils import EMD_CD, compute_all_metrics
import json
import pickle
import random

@DATASETS.register_module()
class CustomDataset(Dataset):

    def __init__(
        self,
        data_dir = '../datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal',
        npoints=2500, 
        scale_mode='shape_unit',
        part_scale_mode='shape_canonical',
        eval_mode='ae',
        clip=True,
        n_class=4
    ):
        self.scale_mode=scale_mode 
        self.part_scale_mode=part_scale_mode 
        self.npoints = npoints # Sampling points
        self.root = data_dir # File root path
        self.eval_mode=eval_mode
        self.n_class=n_class
        self.clip=clip

        d = pickle.load(open(self.root, 'rb'))
        self.data = d['pred']
        self.label =d['pred_seg_mask']

    def __getitem__(self, index):
        # resample
        point_set = self.data[index] # Sample by index
        label = self.label[index] # Sample by index
        
        shifts = np.zeros((self.n_class, 3))
        scales = np.ones((self.n_class, 3))
        present = torch.zeros(self.n_class)

        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice]
        label = label[choice]
        point_set, shift, scale = pc_norm(point_set, self.scale_mode, to_torch=False) 
        out = np.zeros_like(point_set)
        for i in range(self.n_class):
            idx = label == i
            if idx.sum() >= 10: # if less than 10 points we ignore
                part_point_set = point_set[idx]
                part_point_set_mean = part_point_set.mean(0)
                part_point_set_std = part_point_set.std(0)
                if np.any(part_point_set_std == 0.0):
                    present[i] = 0
                    part_point_set_std = np.ones_like(part_point_set_std)
                else:
                    present[i] = 1
                print(self.part_scale_mode)
                shifted_chosen, pshift, pscale = pc_norm(part_point_set, self.part_scale_mode, to_torch=False, clip=self.clip)
                print(pshift, pscale)
                shifts[i] = pshift[0]
                scales[i] = pscale[0]
                out[idx] = shifted_chosen
            elif np.any(idx):
                # we change its label to their nearest neighbors
                part_point = point_set[idx] # n, 3
                rest_point = point_set[~idx]
                rest_seg = label[~idx]
                dist = (part_point[:, None] - rest_point[None]) ** 2
                label_idx = dist.sum(-1).argmin(1)
                label[idx] = np.take_along_axis(rest_seg, label_idx, axis=0)
        



        
        return {
            'input':out, 
            'seg_mask':label,
            'shift':shift,
            'scale':scale, 
            'id':index,
            } # pointset is point cloud data, cls has 16 categories, and seg is a small category corresponding to different points in the data

    def __len__(self):
        return self.data.shape[0]

    

    def evaluate(self, results, save_num_batch, device):
        save_dict = dict()
        preds = []
        refs = []
        for idx, pred_dict in enumerate(tqdm(results)):
            shift = pred_dict['shift']
            scale = pred_dict['scale']
            del pred_dict['shift']
            del pred_dict['scale']
            pred = pred_dict['pred']
            ref = pred_dict['input_ref']
            if pred.shape[1] > 2048:
                pred = fps(pred.to('cuda').contiguous(), 2048).cpu()
            if ref.shape[1] > 2048:
                ref = fps(ref.to('cuda').contiguous(), 2048).cpu()
            if self.eval_mode == 'ae':
                pred = pred * scale + shift
                ref = ref * scale + shift
            else:
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
                if self.using_whole_chair_only:
                    present = pred_dict['present'][:, :3].sum(1) == 3 
                    pred = pred[present]
                    ref = ref[present]
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
        if self.eval_mode == 'ae':
            metrics = EMD_CD(preds, refs, 32)
        elif self.eval_mode == 'gen':
            # metrics = dict(l=torch.zeros(1))
            print(preds.shape, refs.shape)
            metrics = compute_all_metrics(preds, refs, 32)
        ssave_dict = dict()
        for k, v in save_dict.items():
            ssave_dict[k] = np.concatenate(v, axis=0)
        return ssave_dict, metrics
