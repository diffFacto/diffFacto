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
import random


@DATASETS.register_module()
def ShapeNetPart(
    batch_size, 
    root, 
    npoints, 
    num_workers=0, 
    scale_mode=None,
    eval_mode='ae',
    distributed=False, 
    shuffle=True, 
    drop_last=True,
    ):
    return ShapeNetParts(root, npoints, scale_mode=scale_mode, eval_mode=eval_mode)





@DATASETS.register_module()
class ShapeNetParts(Dataset):

    def __init__(
        self,
        root = '../datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal',
        npoints=2500, 
        scale_mode=None,
        eval_mode='ae'
    ):
        self.scale_mode=scale_mode 
        self.npoints = npoints # Sampling points
        self.root = root # File root path
        self.eval_mode=eval_mode

        self.data = torch.load(self.root).numpy()

    def __getitem__(self, index):
        # resample
        point_set = self.data[index] # Sample by index
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice] 
        point_set, shift, scale = pc_norm(point_set, self.scale_mode, to_torch=True)
        
        
        return {
            'input':point_set, 
            'ref':point_set, 
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
