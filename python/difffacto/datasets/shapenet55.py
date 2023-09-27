import numpy as np
import torch
from torch.utils.data import Dataset
from difffacto.utils.registry import DATASETS 
from difffacto.utils.misc import *
import os
import pickle
from tqdm import tqdm
from pathlib import Path
from .dataset_utils import pc_norm, shapenet55_id_to_cat, build_loss_funcs, DataLoaderWrapper

@DATASETS.register_module()
def ShapeNet(batch_size, root, npoints, split, crop=[.25, .75], num_workers=0, scale_mode='shape_unit', mode='median', loss=None, distributed=False, shuffle=True, cats=['all']):
    mode={
            'easy': 1/4,
            'median' :1/2,
            'hard':3/4,
            'complete': 1.
        }[mode]
    if split == 'test':
        crop_range = int(mode * npoints)
    
    elif isinstance(crop, list):
        crop_range = [int(crop[0] * npoints), int(crop[1] * npoints)]
    elif isinstance(crop, float):
        crop_range = int(crop * npoints)
    else:
        raise NotImplementedError
    dataset = _ShapeNet(root, npoints, split, scale_mode=scale_mode, fine_loss=loss, cats=cats)
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
    else:
        sampler = None
    return DataLoaderWrapper(dataset, batch_size, num_workers, True, crop_range, npoints, sampler=sampler), sampler



    

    

class _ShapeNet(Dataset):
    def __init__(self, root, npoints, split, scale_mode='shape_unit', fine_loss=None, cats=['all']):
        if 'all' in cats:
            cats = list(shapenet55_id_to_cat.values())
        for cat in cats:
            if cat not in shapenet55_id_to_cat.values():
                assert False, "Unknown category"
        self.cats = cats
        self.data_root = root
        self.subset = split
        self.npoints = npoints
        self.scale_mode=scale_mode
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        self.fine_loss_func = build_loss_funcs(fine_loss)
        self.parent_path = Path(root).parent.absolute()
        

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            if shapenet55_id_to_cat[taxonomy_id] in self.cats:
                model_id = line.split('-')[1].split('.')[0]
                self.file_list.append({
                    'taxonomy_id': taxonomy_id,
                    'model_id': model_id,
                    'file_path': os.path.join(self.parent_path, 'shapenet_pc', line)
                })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    


    
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = np.load(sample['file_path']).astype(np.float32)
        data = filter_points(data, self.npoints)
        data, shift, scale = pc_norm(data, self.scale_mode, to_torch=True)

        return {'taxonomy_id':sample['taxonomy_id'], 'model_id':sample['model_id'], 'shift':shift, 'scale': scale, 'pointcloud':data}

    def __len__(self):
        return len(self.file_list)
    
    def evaluate(self, results, save_num_batch, device):
        save_dict = dict()
        fine_metrics = {k:0. for k in self.fine_loss_func.keys()}
        for idx, pred_dict in enumerate(tqdm(results)):
            ref = pred_dict['ref'].to(device)
            pred = pred_dict['pred'].to(device)
            shift = pred_dict['shift'].to(device)
            scale = pred_dict['scale'].to(device)
            del pred_dict['shift']
            del pred_dict['scale']
            ref, pred = map(lambda p: p * scale + shift, [ref, pred])
            for k, v in self.fine_loss_func.items():
                fine_metrics[k] += v(ref, pred)
            if idx < save_num_batch:
                for k, v in pred_dict.items():
                    if k not in save_dict.keys():
                        save_dict[k] = []
                    save_dict[k].append(v.detach().cpu().numpy())
            
        fine_metrics = {'fine_' + str(k): v / len(results) for k, v in fine_metrics.items()}
        save_dict = {k: np.concatenate(v, axis=0) for k, v in save_dict.items()}
        return save_dict, fine_metrics
        
            