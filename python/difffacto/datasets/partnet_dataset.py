import os
import torch
import torch.nn.functional as F
from difffacto.utils.registry import DATASETS
from torch.utils.data import Dataset
from difffacto.utils.misc import fps
import numpy as np
from tqdm.auto import tqdm
from .dataset_utils import pc_norm, shapenet_part_normal_cat_to_id, build_loss_funcs, DataLoaderWrapper, DataLoaderWrapperOne, load_ply_withNormals
from .evaluation_utils import EMD_CD, compute_all_metrics
import json


@DATASETS.register_module()
def Partnet(
    batch_size, 
    root, 
    n_part, 
    npoints, 
    part_npoints,
    split, 
    num_workers=0, 
    eval_mode='ae',
    scale_mode='shape_unit', 
    part_scale_mode=None,
    distributed=False, 
    shuffle=True, 
    class_choice='Chair',
    save_only=False,
    augment=False):
    assert not distributed
    dataset = _Partnet(root, npoints, part_npoints, split, scale_mode=scale_mode, class_choice=class_choice, save_only=save_only, eval_mode=eval_mode, augment=augment, n_part=n_part,part_scale_mode=part_scale_mode)

    return DataLoaderWrapperOne(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=shuffle), None




class _Partnet(Dataset):

    def __init__(
        self,
        root = '/orion/u/w4756677/datasets/coalace_data_processed/',
        npoints=2048,
        part_npoints=512,
        split='train', 
        n_part=4, 
        class_choice='Chair',
        normal_channel=False, 
        scale_mode='shape_bbox',
        part_scale_mode=None,
        eval_mode='ae',
        save_only=False,
        augment=False,
    ):
        self.eval_mode=eval_mode
        self.augment=augment
        assert not (self.augment and split == 'test')
        self.npoints = npoints # Sampling points per part
        self.root = root # File root path
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt') # Path corresponding to category and folder name
        self.cat = {}
        self.normal_channel = normal_channel # Whether to use rgb information
        self.scale_mode = scale_mode
        self.part_scale_mode=part_scale_mode
        self.part_npoints=part_npoints

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
            # print(fns[0][0:-4])
            densePoints = [f for f in fns if (f[-7:] == 'ori.ply')]
            labels = [f for f in fns if (f[-10:] == 'ori.labels')]
            assert len(densePoints) == len(labels)
            if split == 'trainval':
                labels = [fn for fn in labels if ((fn[0:-11] in train_ids) or (fn[0:-11] in val_ids))]
                densePoints = [fn for fn in densePoints if ((fn[0:-8] in train_ids) or (fn[0:-8] in val_ids))]
            elif split == 'train':
                labels = [fn for fn in labels if ((fn[0:-11] in train_ids))]
                densePoints = [fn for fn in densePoints if ((fn[0:-8] in train_ids))] # Judge whether the txt file in the folder is in the training TXT. If so, the txt file obtained in fns is the file to be trained in all txt files in this category, and put it into fns
            elif split == 'val':
                labels = [fn for fn in labels if ((fn[0:-11] in val_ids))]
                densePoints = [fn for fn in densePoints if ((fn[0:-8] in val_ids))]
            elif split == 'test':
                labels = [fn for fn in labels if ((fn[0:-11] in test_ids))]
                densePoints = [fn for fn in densePoints if ((fn[0:-8] in test_ids) )]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in densePoints:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.ply'), os.path.join(dir_point, token + '.labels')))  # Generate a dictionary and combine the category name and the training path as the training data in a large class
                #After the above code is executed, you can put all the data to be trained or verified into a dictionary. The key of the dictionary is the category to which the data belongs, such as aircraft. Value is the full path of its corresponding data
                #{aircraft: [path 1, path 2.....]}
        #####################################################################################################################################################
        self.datapath = []
        for item in self.cat: # self.cat is the dictionary corresponding to the category name and folder
            for fn in self.meta[item]:
                self.datapath.append((item, *fn)) # Generate tuples of labels and point cloud paths, and convert the dictionary in self.met into a tuple
        print(len(self.datapath))
        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        ## self.classes corresponds the name and index of the category, such as aircraft < --- > 0
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
        self.num_class = n_part

    def store_noise(self, all_noises, all_ids):
        self.noises = dict()
        for noises, ids in zip(all_noises, all_ids):
            assert noises.shape[0] == len(ids)
            for i, id in enumerate(ids):
                self.noises[id.item()] = noises[i].detach().cpu().numpy()

    def __getitem__(self, index):
        if index in self.cache: # The initial slef.cache is an empty dictionary, which is used to store the retrieved data and place it according to (point_set, cls, seg) while avoiding repeated sampling
            valid_point_set, cls, valid_seg, valid_seg_onehot, shift, scale = self.cache[index]
        else:
            fn = self.datapath[index] # The path to get the training data according to the index self.datepath is a tuple (class name, path)
            cat = self.datapath[index][0] # Get the class name
            cls = self.classes[cat] # Convert class name to index
            cls = np.array([cls]).astype(np.int32)
            point_set = load_ply_withNormals(fn[1]).astype(np.float32) # size 20488,7 read in this txt file, a total of 20488 points, each point xyz rgb + sub category label
            
            seg = np.loadtxt(fn[2]).astype(np.long) - 1 # Get the label of the sub category
            seg_onehot = F.one_hot(torch.from_numpy(seg), num_classes=self.num_class)
            
            # get rid of dummy points
            mask = np.abs(point_set).sum(1) != 0
            valid_point_set = point_set[mask]
            valid_seg = seg[mask]
            valid_seg_onehot = seg_onehot[mask]
            valid_point_set, shift, scale = pc_norm(valid_point_set, self.scale_mode, to_torch=False) 
            if len(self.cache) < self.cache_size:
                self.cache[index] = (valid_point_set, cls, valid_seg, valid_seg_onehot, shift, scale)
        
        noise = self.noises.get(index, np.zeros(1))#for cimle
        choice = np.random.choice(len(valid_seg), self.npoints, replace=False) # Random sampling of data in a category returns the index, allowing repeated sampling
        # resample
        ref_point_set = valid_point_set[choice, :] # Sample by index
        ref_seg = valid_seg[choice]
        ref_seg_onehot = valid_seg_onehot[choice]
        shifts = np.zeros((self.num_class, 3))
        scales = np.zeros((self.num_class, 3))
        present = torch.zeros(self.num_class)
        out = np.zeros((self.num_class, self.part_npoints, 3))
        out_mask = np.zeros((self.num_class, self.part_npoints))
        out_onehot = np.zeros((self.num_class, self.part_npoints, self.num_class))
        # print(seg.shape, seg_onehot.shape)
        for i in range(self.num_class):
            idx = valid_seg == i
            if idx.any(): # if less than 10 points we ignore
                present[i] = 1
                part_point_set = valid_point_set[idx]
                shifted_chosen, pshift, pscale = pc_norm(part_point_set, self.part_scale_mode, to_torch=False, clip=True) 
                if self.part_npoints < shifted_chosen.shape[0]:
                    choice = np.random.choice(shifted_chosen.shape[0], self.part_npoints, replace=False) # Random sampling of data in a category returns the index, allowing repeated sampling
                    shifted_chosen = shifted_chosen[choice]
                    mask_chosen = valid_seg[choice]
                    onehot_chosen = valid_seg_onehot[choice]
                    
                shifts[i] = part_point_set.mean(0)
                scales[i] = part_point_set.std(0).clip(1e-2, 1)
                out[i] = shifted_chosen
                out_mask[i] = mask_chosen
                out_onehot[i] = onehot_chosen
        ref_point_set, shift, scale = map(lambda t: torch.from_numpy(t).float(), [ref_point_set, shift, scale])
        out, shifts, scales = map(lambda t: torch.from_numpy(t).float(), [out, shifts, scales])
        if self.augment:
            rand_scale = torch.rand(1,3) / 2 + 0.7
            rand_shift = torch.rand(1, 3) - 0.5
            # (o + s) * scale
            ori_out = (ori_out + rand_shift) * rand_scale
            shift = shift  + scale * rand_shift
            scale = rand_scale * scale


        return {'part_scale': scales.transpose(0, 1), 
                'part_shift': shifts.transpose(0, 1), 
                'input':out, 
                "present": present,
                'attn_map': out_onehot, 
                'ref':ref_point_set,
                'ref_attn_map': ref_seg_onehot,
                'ref_seg_mask': ref_seg,
                'class':cls, 
                'seg_mask':out_mask, 
                'shift':shift, 
                'scale':scale, 
                'id':index,
                'noise':torch.from_numpy(noise)
                } # pointset is point cloud data, cls has 16 categories, and seg is a small category corresponding to different points in the data

    def __len__(self):
        return len(self.datapath)

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
                pred = fps(pred.to('cuda'), 2048)
            if ref.shape[1] > 2048:
                ref = fps(ref.to('cuda'), 2048)
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
            B, N, C = ref.shape
            pred_dict = {k: v.reshape(B, N, C) * scale + shift if v.shape[-1] == 3 else v for k, v in pred_dict.items()}    
            pred_dict['pred'] = pred 
            pred_dict['input_ref'] = ref  
                     
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
