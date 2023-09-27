import os
import torch
import torch.nn.functional as F
from difffacto.utils.registry import DATASETS
from difffacto.utils.misc import fps
from torch.utils.data import Dataset
import numpy as np
from tqdm.auto import tqdm
from .dataset_utils import pc_norm, shapenet_part_normal_cat_to_id, build_loss_funcs, DataLoaderWrapper, DataLoaderWrapperOne
from .evaluation_utils import EMD_CD, compute_all_metrics, compute_bbox_metric, compute_part_metric, compute_snapping_metric
import json
import random
from pointnet2_ops import pointnet2_utils


@DATASETS.register_module()
def ShapeNetSegPart(
    batch_size, 
    root, 
    npoints, 
    split, 
    num_workers=0, 
    eval_mode='ae',
    scale_mode='shape_unit', 
    part_scale_mode=None,
    distributed=False, 
    shuffle=True, 
    drop_last=True,
    class_choice='Chair',
    save_only=False,
    augment=False,
    augment_shift=False,
    augment_scale=False,
    using_whole_chair_only=False,
    clip=True,
    dropout_part=0.0):
    assert not distributed
    dataset = _ShapeNetSegParts(root, npoints, split, scale_mode=scale_mode, class_choice=class_choice, save_only=save_only, eval_mode=eval_mode, augment=augment, augment_shift=augment_shift, augment_scale=augment_scale, dropout_part=dropout_part, using_whole_chair_only=using_whole_chair_only, part_scale_mode=part_scale_mode, clip=clip)

    return DataLoaderWrapperOne(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last, shuffle=shuffle), None






@DATASETS.register_module()
def ShapeNetSeg(
    batch_size, 
    root, 
    npoints, 
    split, 
    num_workers=0, 
    scale_mode='shape_unit', 
    eval_mode='ae',
    distributed=False, 
    drop_last=True,
    shuffle=True, 
    class_choice='Chair',
    save_only=False,
    augment=False, 
    augment_shift=False,
    augment_scale=False):
    assert not distributed
    dataset = _ShapeNetSeg(root, npoints, split, scale_mode=scale_mode, class_choice=class_choice, save_only=save_only, eval_mode=eval_mode, augment=augment, augment_shift=augment_shift, augment_scale=augment_scale)
    
    return DataLoaderWrapperOne(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last, shuffle=shuffle), None




class _ShapeNetSeg(Dataset):

    def __init__(
        self,
        root = '../datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal',
        npoints=2500, 
        split='train', 
        class_choice='Chair', 
        normal_channel=False, 
        scale_mode='shape_unit',
        eval_mode='ae',
        save_only=False,
        augment=False,
        augment_shift=False,
        augment_scale=False,
        dropout_part=0.0,
        using_whole_chair_only=False,
        part_scale_mode=None,
        clip=True,
    ):
        self.eval_mode=eval_mode
        self.part_scale_mode=part_scale_mode if part_scale_mode is not None else scale_mode
        self.augment=augment
        self.clip=clip
        if augment:
            augment_shift = augment_scale = True
        self.augment_shift = augment_shift
        self.augment_scale = augment_scale
        self.using_whole_chair_only=using_whole_chair_only
        assert not (self.augment and split == 'test')
        self.npoints = npoints # Sampling points
        self.root = root # File root path
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt') # Path corresponding to category and folder name
        self.cat = {}
        self.normal_channel = normal_channel # Whether to use rgb information
        self.scale_mode = scale_mode

        self.cat = shapenet_part_normal_cat_to_id #{'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343', 'Chair': '03001627', 'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134', 'Lamp': '03636649', 'Laptop': '03642806', 'Motorbike': '03790512', 'Mug': '03797390', 'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987', 'Table': '04379243'}
        self.classes_original = dict(zip(self.cat, range(len(self.cat)))) #{'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}

        self.cat = {class_choice:shapenet_part_normal_cat_to_id[class_choice]} if class_choice is not None else shapenet_part_normal_cat_to_id
        self.class_choice = class_choice
        self.save_only=save_only
        print(self.cat)
        self.noises = dict()


        self.meta = {} # Read the classified folder jason file and put their names in the list
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)]) # '928c86eabc0be624c2bf2dc31ba1713' this is the first value
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item]) # # Get the path corresponding to a folder, such as the first folder 02691156
            fns = sorted(os.listdir(dir_point))  # Get each txt file under the folder according to the path and put it in the list
            
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids] # Judge whether the txt file in the folder is in the training TXT. If so, the txt file obtained in fns is the file to be trained in all txt files in this category, and put it into fns
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            elif split == 'all':
                fns = [fn for fn in fns if ((fn[0:-4] in test_ids) or (fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)
            # print(os.path.basename(fns))
            for fn in fns:
                "The first i Secondary cycle  fns What you get is the third i Training eligible in folders txt Folder name"
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))  # Generate a dictionary and combine the category name and the training path as the training data in a large class
                #After the above code is executed, you can put all the data to be trained or verified into a dictionary. The key of the dictionary is the category to which the data belongs, such as aircraft. Value is the full path of its corresponding data
                #{aircraft: [path 1, path 2.....]}

            
        # exit()
        # print(fns[0][0:-4])
        #####################################################################################################################################################
        self.datapath = []
        for item in self.cat: # self.cat is the dictionary corresponding to the category name and folder
            for fn in self.meta[item]:
                self.datapath.append((item, fn)) # Generate tuples of labels and point cloud paths, and convert the dictionary in self.met into a tuple
        print(len(self.datapath))
        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        ## self.classes corresponds the name and index of the category, such as aircraft < --- > 0
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        """
        shapenet There are 16 categories, and then each category has some components, such as aircraft 'Airplane': [0, 1, 2, 3] Among them, the four sub categories labeled 0 1 2 3 belong to the category of aircraft
        self.seg_classes Is to correspond the major class to the minor class
        """
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
        self.num_class = len(self.seg_classes[class_choice])
        self.dropout_part=dropout_part
        # self.compute_part_distribution()

    def store_noise(self, all_noises, all_ids):
        self.noises = dict()
        for noises, ids in zip(all_noises, all_ids):
            assert noises.shape[0] == len(ids)
            for i, id in enumerate(ids):
                self.noises[id.item()] = noises[i].detach().cpu().numpy()
    def compute_part_distribution(self):
        distribution = dict()
        for fn in self.datapath:
            data = np.loadtxt(fn[1]).astype(np.float32) # size 20488,7 read in this txt file, a total of 20488 points, each point xyz rgb + sub category label
            present = [0]*self.num_class
            
            seg = data[:, -1].astype(np.long) - self.seg_classes[self.class_choice][0] # Get the label of the sub category
            for i in range(self.num_class):
                idx = seg == i 
                if idx.any():
                    present[i] = 1
            if ''.join(map(str, present)) not in distribution.keys():
                distribution[''.join(map(str, present))] = 0
            distribution[''.join(map(str, present))] += 1
        print(distribution)
        print({k: v / len(self.datapath) for k, v in distribution.items()})
    def __getitem__(self, index):
        if index in self.cache: # The initial slef.cache is an empty dictionary, which is used to store the retrieved data and place it according to (point_set, cls, seg) while avoiding repeated sampling
            point_set, cls, seg,  = self.cache[index]
        else:
            fn = self.datapath[index] # The path to get the training data according to the index self.datepath is a tuple (class name, path)
            cat = self.datapath[index][0] # Get the class name
            cls = self.classes[cat] # Convert class name to index
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32) # size 20488,7 read in this txt file, a total of 20488 points, each point xyz rgb + sub category label
            token = (os.path.splitext(os.path.basename(fn[1]))[0])
            
            if not self.normal_channel:  # Determine whether to use rgb information
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.long) - self.seg_classes[self.class_choice][0] # Get the label of the sub category


            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg, token)
        
        noise = self.noises.get(index, np.zeros(1)) #for cimle
        
        choice = np.random.choice(len(seg), self.npoints, replace=True) # Random sampling of data in a category returns the index, allowing repeated sampling
        # resample
        point_set = point_set[choice, :] # Sample by index
        seg = seg[choice]
        seg_onehot = F.one_hot(torch.from_numpy(seg), num_classes=self.num_class)
        # print(seg.shape, seg_onehot.shape)
        present = torch.zeros(self.num_class)
        part_means = np.zeros((self.num_class, 3), dtype=np.float32)
        part_scale = np.ones((self.num_class, 3), dtype=np.float32)

        point_set, shift, scale = pc_norm(point_set, self.scale_mode, to_torch=True)
        for i in range(self.num_class):
            idx = seg == i 
            if idx.any():
                present[i] = 1
                part_point_set = point_set[idx]
                part_point_set_mean = part_point_set.mean(0)
                part_point_set_std = part_point_set.std(0)
                part_means[i] = part_point_set_mean
                part_scale[i] = part_point_set_std
                
        if self.augment_shift or self.augment_scale:
            if self.augment_scale:
                rand_scale = torch.rand(1,3) / 2 + 0.7
            else:
                rand_scale = torch.ones(1,3) 
            if self.augment_shift:
                rand_shift = torch.rand(1, 3) - 0.5
            else:
                rand_shift = torch.zeros(1, 3)
            # ((o - s  - ss * scale ) / ssc * scale
            point_set = (point_set - rand_shift) / rand_scale
            shift = shift + scale * rand_shift
            scale = rand_scale * scale
        


        part_scale = part_scale.transpose(1,0)
        part_means = part_means.transpose(1,0)
        return {
            'present': present,
            'part_scale': part_scale, 
            'part_shift': part_means, 
            'input':point_set, 
            'attn_map': seg_onehot, 
            'ref_attn_map': seg_onehot, 
            'ref':point_set, 
            'class':cls, 
            'seg_mask':seg, 
            'ref_seg_mask':seg, 
            'shift':shift,
            'scale':scale, 
            'id':index,
            'token':token,
            'noise':torch.from_numpy(noise),
            } # pointset is point cloud data, cls has 16 categories, and seg is a small category corresponding to different points in the data

    def __len__(self):
        return len(self.datapath)

    
    # def evaluate(self, results, save_num_batch, device):
    #     save_dict = dict()
    #     for idx, pred_dict in enumerate(tqdm(results)):
    #         idxx = pred_dict['id']
    #         mean, logvar = pred_dict['mean'], pred_dict['logvar']
    #         for id in idxx:
    #             save_dict[id.item] = (mean, logvar)
    #     print(len(save_dict.keys()))
    #     return save_dict, dict(l=0)
    def evaluate(self, results, save_num_batch, device):
        save_dict = dict()
        preds = []
        preds_mask = []
        refs_mask = []
        refs = []
        for idx, pred_dict in enumerate(tqdm(results)):
            if self.save_only:
                # pred_dict = {k: v * scale + shift if v.shape[-1] == 3 else v for k, v in pred_dict.items()}    
                if idx < save_num_batch:
                    for k, v in pred_dict.items():
                        if k not in save_dict.keys():
                            save_dict[k] = []
                        if isinstance(v, torch.Tensor):
                            save_dict[k].append(v.detach().cpu().numpy())
                        else:
                            save_dict[k].append(v)
                continue
            shift = pred_dict['shift']
            scale = pred_dict['scale']
            del pred_dict['shift']
            del pred_dict['scale']
            
            pred = pred_dict['pred']
            pred_mask = pred_dict['pred_seg_mask']
            ref = pred_dict['input_ref']
            ref_mask = pred_dict['ref_seg_mask']
            if pred.shape[1] > 2048:
                pred, idx = fps(pred.to('cuda').contiguous(), 2048, ret_id=True).cpu()
                pred_mask = pointnet2_utils.gather_operation(pred_mask.unsqueeze(1).contiguous(), idx).reshape(-1, 2048).cpu()
            if ref.shape[1] > 2048:
                ref, ref_idx = fps(ref.to('cuda').contiguous(), 2048).cpu()
                ref_mask = pointnet2_utils.gather_operation(ref_mask.unsqueeze(1).contiguous(), ref_idx).reshape(-1, 2048).cpu()
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
            preds_mask.append(pred_mask)
            refs_mask.append(ref_mask)
            pred_dict = {k: v * scale + shift if v.shape[-1] == 3 else v for k, v in pred_dict.items()}    
            # pred_dict['pred'] = pred 
            # pred_dict['input_ref'] = ref  
                     
            if idx < save_num_batch:
                for k, v in pred_dict.items():
                    if k not in save_dict.keys():
                        save_dict[k] = []
                    save_dict[k].append(v.detach().cpu().numpy())
        if self.save_only: 
            ssave_dict = dict()
            for k, v in save_dict.items():
                ssave_dict[k] = np.concatenate(v, axis=0)
            return ssave_dict, dict(l=0)
        preds = torch.cat(preds, dim=0).cuda()
        refs = torch.cat(refs, dim=0).cuda()
        preds_mask = torch.cat(preds_mask, dim=0).cuda()
        refs_mask = torch.cat(refs_mask, dim=0).cuda()
        if self.eval_mode == 'ae':
            metrics = EMD_CD(preds, refs, 32)
        elif self.eval_mode == 'gen_part':
            # metrics = dict(l=torch.zeros(1))
            print(preds.shape, refs.shape)
            snapping_metric = compute_snapping_metric(preds, preds_mask, cls=self.class_choice)
            for k, v in snapping_metric.items():
                print(f'{k}: {v}')
            oracle_snapping_metric = compute_snapping_metric(refs, refs_mask, cls=self.class_choice)
            bbox_metric = compute_bbox_metric(preds, preds_mask, refs, refs_mask, 32, n_class=self.num_class, metric='chamfer')
            part_metric = compute_part_metric(preds, preds_mask, refs, refs_mask, 32, n_class=self.num_class)
            metrics = compute_all_metrics(preds, refs, 32)
            metrics.update(snapping_metric)
            metrics.update({f"oracle_{k}":v for k, v in oracle_snapping_metric.items()})
            metrics.update(part_metric)
            metrics.update(bbox_metric)
        elif self.eval_mode == 'gen':
            # metrics = dict(l=torch.zeros(1))
            print(preds.shape, refs.shape)
            metrics = compute_all_metrics(preds, refs, 32)
        ssave_dict = dict()
        for k, v in save_dict.items():
            ssave_dict[k] = np.concatenate(v, axis=0)
        return ssave_dict, metrics

class _ShapeNetSegParts(_ShapeNetSeg):

    def __init__(
        self,
        root = '../datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal',
        npoints=2500, 
        split='train', 
        class_choice='Chair', 
        normal_channel=False, 
        scale_mode='shape_unit',
        save_only=False,
        eval_mode='ae',
        augment=False,
        augment_shift=False,
        augment_scale=False,
        dropout_part=0.0,
        using_whole_chair_only=False,
        part_scale_mode=None,
        clip=True,
    ):
        super().__init__(
            root = root,
            npoints=npoints, 
            split=split, 
            class_choice=class_choice, 
            normal_channel=normal_channel, 
            scale_mode=scale_mode,
            save_only=save_only,
            eval_mode=eval_mode,
            augment=augment,
            augment_shift=augment_shift,
            augment_scale=augment_scale,
            dropout_part=dropout_part,
            using_whole_chair_only=using_whole_chair_only,
            part_scale_mode=part_scale_mode,
            clip=clip
        )
        
    def __getitem__(self, index):
        if index in self.cache: # The initial slef.cache is an empty dictionary, which is used to store the retrieved data and place it according to (point_set, cls, seg) while avoiding repeated sampling
            point_set, cls, seg, token = self.cache[index]
        else:
            fn = self.datapath[index] # The path to get the training data according to the index self.datepath is a tuple (class name, path)
            cat = self.datapath[index][0] # Get the class name
            cls = self.classes[cat] # Convert class name to index
            noise = self.noises.get(index, np.zeros(1))#for cimle
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32) # size 20488,7 read in this txt file, a total of 20488 points, each point xyz rgb + sub category label
            token = (os.path.splitext(os.path.basename(fn[1]))[0])
            
            if not self.normal_channel:  # Determine whether to use rgb information
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.long) - self.seg_classes[self.class_choice][0] # Get the label of the sub category


            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg, token)
        
        # Random sampling of data in a category returns the index, allowing repeated sampling
        # resample
        noise = self.noises.get(index, np.zeros(1))#for cimle
        
        shifts = np.zeros((self.num_class, 3))
        scales = np.ones((self.num_class, 3))
        present = torch.zeros(self.num_class)
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        ori_out = point_set[choice] 
        
        seg = seg[choice]
        
        ori_out, shift, scale = pc_norm(ori_out, self.scale_mode, to_torch=False) 
        out = np.zeros_like(ori_out)
        for i in range(self.num_class):
            idx = seg == i
            if idx.sum() >= 10: # if less than 10 points we ignore
                part_point_set = ori_out[idx]
                part_point_set_mean = part_point_set.mean(0)
                part_point_set_std = part_point_set.std(0)
                if np.any(part_point_set_std == 0.0):
                    present[i] = 0
                    part_point_set_std = np.ones_like(part_point_set_std)
                else:
                    present[i] = 1
                shifted_chosen, pshift, pscale = pc_norm(part_point_set, self.part_scale_mode, to_torch=False, clip=self.clip) 
                shifts[i] = pshift[0]
                scales[i] = pscale[0]
                out[idx] = shifted_chosen
            elif np.any(idx):
                # we change its label to their nearest neighbors
                part_point = ori_out[idx] # n, 3
                rest_point = ori_out[~idx]
                rest_seg = seg[~idx]
                dist = (part_point[:, None] - rest_point[None]) ** 2
                seg_idx = dist.sum(-1).argmin(1)
                seg[idx] = np.take_along_axis(rest_seg, seg_idx, axis=0)
                
        seg_onehot = F.one_hot(torch.from_numpy(seg), num_classes=self.num_class)
        part_idx = np.random.rand(self.num_class) < self.dropout_part
        drop_outed_present = present.clone()
        drop_outed_present[part_idx] = 0
        ori_out, shift, scale = map(lambda t: torch.from_numpy(t).float(), [ori_out, shift, scale])
        out, shifts, scales = map(lambda t: torch.from_numpy(t).float(), [out, shifts, scales])
        if self.augment_shift or self.augment_scale:
            if self.augment_scale:
                rand_scale = torch.rand(1,3) / 2 + 0.7
            else:
                rand_scale = torch.ones(1,3) 
            if self.augment_shift:
                rand_shift = torch.rand(1, 3) - 0.5
            else:
                rand_shift = torch.zeros(1, 3)
            # (o + s) * scale
            ori_out = (ori_out + rand_shift) * rand_scale
            shift = shift  + scale * rand_shift
            scale = rand_scale * scale

        # print(seg.shape, seg_onehot.shape)
        assert (shifts.shape == shifts.shape == (self.num_class, 3))
        assert ori_out.shape == (self.npoints, 3) == out.shape
        assert seg_onehot.shape == (self.npoints, self.num_class)
        
        




        return {
            'present': present,
            'dp_present': drop_outed_present,
            'part_scale':scales.transpose(0,1), 
            'part_shift':shifts.transpose(0,1), 
            'input':out, 
            'ref_attn_map': seg_onehot, 
            'attn_map': seg_onehot, 
            'ref':ori_out, 
            'class':cls, 
            'ref_seg_mask':seg, 
            'token':token,
            'seg_mask':seg, 
            'shift':shift, 
            'scale':scale, 
            'id':index,
            'noise':torch.from_numpy(noise),
            } # pointset is point cloud data, cls has 16 categories, and seg is a small category corresponding to different points in the data
