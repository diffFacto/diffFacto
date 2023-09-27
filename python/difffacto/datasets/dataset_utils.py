import numpy as np
import torch
from torch.utils.data import DataLoader
from difffacto.utils.misc import separate_point_cloud
from difffacto.utils.registry import build_from_cfg, METRICS
import random
from plyfile import  PlyData

def load_ply_withNormals(file_name ):
    
    ply_data = PlyData.read(file_name)
    vertices = ply_data['vertex']
    
    points  = np.vstack([vertices['x'],  vertices['y'],  vertices['z']]).T

    return points

class IterWrapper:
    def __init__(self, loader):
        self.loader = loader
        self.loader_iter = iter(loader.loader)
    def __next__(self):
        d = self.loader_iter.next()
        partials, _ = separate_point_cloud(d['ref'].cuda(), self.loader.npoints, self.loader.crop_range)
        d['partial'] = partials 
        return d

class DataLoaderWrapper:
    def __init__(self, train_dataset, batch_size, num_workers, drop_last, crop_range, npoints, sampler=None):
        self.loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last, sampler=sampler, worker_init_fn=worker_init_fn)
        self.batch_size = self.loader.batch_size
        self.crop_range=crop_range
        self.npoints=npoints
    def __iter__(self):
        return IterWrapper(self)

    def __len__(self):
        return len(self.loader)

    def evaluate(self, results, save_num_batch, device):
        return self.loader.dataset.evaluate(results, save_num_batch, device)


class DataLoaderWrapperOne(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def evaluate(self, results, save_num_batch, device):
        return self.dataset.evaluate(results, save_num_batch, device)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def pc_norm(pc, scale_mode, to_torch=True, stats=None, clip=True):
        if scale_mode == 'global_unit':
            shift = stats['mean'].reshape(1, 3)
            scale = stats['std'].reshape(1, 1)
        elif scale_mode == 'shape_unit':
            shift = pc.mean(0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif scale_mode == 'shape_canonical':
            shift = pc.mean(0).reshape(1, 3)
            scale = pc.std(0).reshape(1, 3)
            if clip:
                scale = scale.clip(1e-2, 1)
            if np.any(scale == 0.0):
                scale[0, scale[0] == 0.0] = 1.0
        elif scale_mode == 'shape_canonical_bbox':
            pc_max = pc.max(0).reshape(1,3) # (1, 3)
            pc_min = pc.min(0).reshape(1,3) # (1, 3)
            shift = ((pc_min + pc_max) / 2).reshape(1, 3)
            scale = (pc_max - pc_min).reshape(1, 3) / 2
            if clip:
                scale = scale.clip(1e-2, 1)
            if np.any(scale == 0.0):
                scale[0, scale[0] == 0.0] = 1.0
        elif scale_mode == 'shape_half':
            shift = pc.mean(0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.5)
        elif scale_mode == 'shape_34':
            shift = pc.mean(0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.75)
        elif scale_mode == 'shape_bbox':
            pc_max = pc.max(0).reshape(1,3) # (1, 3)
            pc_min = pc.min(0).reshape(1,3) # (1, 3)
            shift = ((pc_min + pc_max) / 2).reshape(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        else:
            shift = np.zeros([1, 3])
            scale = np.ones([1, 1])
        pc = (pc - shift) / scale
        if to_torch:
            pc, shift, scale = map(lambda t: torch.from_numpy(t).float(), [pc, shift, scale])
        return pc, shift, scale

def build_loss_funcs(loss_dict):
    loss = dict()
    if isinstance(loss_dict, dict):
        loss[loss_dict['type']] = build_from_cfg(loss_dict, METRICS)
    elif isinstance(loss_dict, list):
        for val_l in loss_dict:
            loss[val_l['type']] = build_from_cfg(val_l, METRICS)
    else:
        loss = loss_dict
    return loss

def augment(input, seg_mask, n_class, vertical_only=True, shift_only=False):
    out = torch.zeros_like(input)
    scales = []
    shifts = []
    for i in range(n_class):
        part = input[seg_mask == i]
        rand_scale = torch.rand(1,3) / 2 + 0.7 if not shift_only else torch.ones(1,3)
        rand_shift = torch.rand(1, 3) - 0.5
        if vertical_only:
            rand_shift[:, [0,2]] = 0.
        scales.append(rand_scale)
        shifts.append(rand_shift)
        part = (part + rand_shift) * rand_scale
        out[seg_mask == i] += part
    scales = torch.cat(scales, dim=0)
    shifts = torch.cat(shifts, dim=0)
    return out, scales, shifts

shapenet55_id_to_cat={
    "02691156": "airplane", "02747177": "trash bin", 
    "02773838": "bag", "02801938": "basket", 
    "02808440": "bathtub", "02818832": "bed", 
    "02828884": "bench", "02843684": "birdhouse", 
    "02871439": "bookshelf", "02876657": "bottle", 
    "02880940": "bowl", "02924116": "bus", 
    "02933112": "cabinet", "02942699": "camera", 
    "02946921": "can", "02954340": "cap", 
    "02958343": "car", "02992529": "cellphone", 
    "03001627": "chair", "03046257": "clock", 
    "03085013": "keyboard", "03207941": "dishwasher", 
    "03211117": "display", "03261776": "earphone", 
    "03325088": "faucet", "03337140": "file cabinet",
    "03467517": "guitar", "03513137": "helmet",
    "03593526": "jar", "03624134": "knife", 
    "03636649": "lamp", "03642806": "laptop", 
    "03691459": "loudspeaker", "03710193": "mailbox", 
    "03759954": "microphone", "03761084": "microwaves", 
    "03790512": "motorbike", "03797390": "mug", 
    "03928116": "piano", "03938244": "pillow", 
    "03948459": "pistol", "03991062": "flowerpot", 
    "04004475": "printer", "04074963": "remote", 
    "04090263": "rifle", "04099429": "rocket", 
    "04225987": "skateboard", "04256520": "sofa", 
    "04330267": "stove", "04379243": "table", 
    "04401088": "telephone", "04460130": "tower", 
    "04468005": "train", "04530566": "watercraft", 
    "04554684": "washer"
    }


shapenet_part_normal_cat_to_id={
    'Airplane': '02691156', 'Bag': '02773838', 
    'Cap': '02954340', 'Car': '02958343', 
    'Chair': '03001627', 'Earphone': '03261776', 
    'Guitar': '03467517', 'Knife': '03624134', 
    'Lamp': '03636649', 'Laptop': '03642806', 
    'Motorbike': '03790512', 'Mug': '03797390', 
    'Pistol': '03948459', 'Rocket': '04099429', 
    'Skateboard': '04225987', 'Table': '04379243'
    }


shapenet_chair_part_distribution={
    '1110': 0.7209302325581395, 
    '1111': 0.2630199803471995, 
    '1101': 0.009498853586636095, 
    '1001': 0.00032754667540124465, 
    '1100': 0.002947920078611202, 
    '0111': 0.0013101867016049786, 
    '0110': 0.0016377333770062235, 
    '1011': 0.00032754667540124465
}