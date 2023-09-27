import torch 
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
import numpy as np
import pickle
from tqdm import tqdm
import os.path as osp 
import h5py
from data_utils import *

def preprocess():
    path = "/mnt/disk/wang/diffusion/PartGlot/logs/pre_trained/pn_agnostic/10-22_02-28-36/pred_label/final/out.pkl"
    data = pickle.load(open(path, 'rb'))
    segs_data = torch.from_numpy(data["geos"].astype(np.float32)).to('cuda')
    segs_mask = torch.from_numpy(data["geos_masks"].astype(np.float32)).to('cuda')
    attn_maps = torch.from_numpy(data['attn_maps'].astype(np.float32)).to('cuda')
    out_segs_data = []
    out_attn_maps = []
    for i in tqdm(range(segs_data.shape[0])):
        seg_data, mask, attn_map = segs_data[i], segs_mask[i], attn_maps[i]
        used_points, used_attn_map = seg_data[mask == 1].reshape(-1, 3), attn_map[:, mask==1].unsqueeze(2).expand(-1, -1, 512).reshape(4, -1).transpose(1,0)
        if used_points.shape[0] > 2048:
            idx = furthest_point_sample(used_points.unsqueeze(0), 2048)
            used_points = gather_operation(used_points.unsqueeze(0).transpose(1,2).contiguous(), idx).squeeze(0).transpose(1,0)
            used_attn_map = gather_operation(used_attn_map.unsqueeze(0).transpose(1,2).contiguous(), idx).squeeze(0).transpose(1,0)
        if used_points.shape[0] < 2048:
            continue
        
        out_segs_data.append(used_points)
        out_attn_maps.append(used_attn_map)
    out_segs_data = torch.stack(out_segs_data, dim=0)
    out_attn_maps = torch.stack(out_attn_maps, dim=0)
    print(out_segs_data.shape, out_attn_maps.shape)
    pickle.dump(out_segs_data.cpu().numpy(), open("/mnt/disk/wang/diffusion/datasets/partglot_data/shapenet_pointcloud_pn_agnostic.pkl", 'wb'))
    pickle.dump(out_attn_maps.cpu().numpy(), open("/mnt/disk/wang/diffusion/datasets/partglot_data/shapenet_label_pn_agnostic.pkl", "wb"))

def pre_process_partglot_data():
    (game_data,
    word2int,
    int2word,
    int2sn,
    sn2int,
    sorted_sn
    ) = unpickle_data(osp.join("/mnt/disk3/wang/diffusion/datasets/partglot_data", "game_data.pkl"))
    gt_attn_maps = pickle.load(open("/mnt/disk3/wang/diffusion/PartGlot/logs/pre_trained/pn_aware/11-01_10-46-46/pred_label/final/ground_truth_attnmaps.pkl", "rb"))['gt_attn_maps']
    distractor_attn_maps = pickle.load(open("/mnt/disk3/wang/diffusion/PartGlot/logs/pre_trained/pn_aware/11-01_11-06-43/pred_label/final/distractor_attnmaps.pkl", "rb"))['gt_attn_maps']
    h5_data = h5py.File(osp.join("/mnt/disk3/wang/diffusion/datasets/partglot_data", "cic_bsp.h5"), "r")
    segs_data = h5_data["data"][:].astype(np.float32)
    print(type(segs_data))
    segs_mask = h5_data["mask"][:].astype(np.float32)
    padded_text, seq_len = pad_text_symbols_with_zeros(
            game_data["text"], 33, force_zero_end=True
        )
    labels = convert_labels_to_one_hot(game_data["target_chair"])
    geo_ids = np.array(game_data[["chair_a", "chair_b", "chair_c"]], dtype=np.int32)
    mask, part_indicator = get_mask_of_game_data(
            game_data,
            word2int,
            True,
            True,
            33,
            True,
        )
    masked_geo_ids = geo_ids[mask]
    masked_labels = labels[mask]
    masked_padded_text = padded_text[mask]
    masked_text = game_data["text"][mask]
    masked_part_indicator = part_indicator[mask]
    l = len(masked_geo_ids)
    print(l)
    out_gt = []
    out_distractor = []
    out_gt_mask = []
    out_distractor_mask = []
    out_texts = []
    out_part_indicator = []
    for idx in tqdm(range(l)):
        geo_id = torch.from_numpy(masked_geo_ids[idx])
        target = torch.tensor(np.argmax(masked_labels[idx]))
        text = [torch.from_numpy(masked_padded_text[idx])] * 2
        part_ind = [torch.from_numpy(masked_part_indicator[idx])] * 2

        geos = torch.from_numpy(segs_data[geo_id])
        geos_mask = torch.from_numpy(segs_mask[geo_id])
        gt = [geos[target], geos[target]]
        gt_mask = [geos_mask[target]] * 2
        distractor_mask = [geos_mask[i] for i in range(3) if i != target]
        distractors = [geos[i] for i in range(3) if i != target]
        out_gt += gt
        out_gt_mask += gt_mask
        out_distractor += distractors
        out_distractor_mask += distractor_mask
        out_texts += text
        out_part_indicator += part_ind
    
    out_gt, out_distractor, out_gt_mask, out_distractor_mask, out_texts, out_part_indicator = map(lambda p: torch.stack(p, dim=0).numpy(), [out_gt, out_distractor, out_gt_mask, out_distractor_mask, out_texts, out_part_indicator])
    print(out_gt.shape, out_distractor.shape, out_gt_mask.shape, out_distractor_mask.shape, out_texts.shape, out_part_indicator.shape)
    pickle.dump(dict(gt_geos=out_gt, distractor_geos=out_distractor, gt_mask=out_gt_mask, distractor_mask=out_distractor_mask, texts=out_texts, part_indicator=out_part_indicator), open("/mnt/disk3/wang/diffusion/datasets/partglot_data/processed_partglot_data.pkl", "wb"))

def pre_process_partglot_data_triplet():
    (game_data,
    word2int,
    int2word,
    int2sn,
    sn2int,
    sorted_sn
    ) = unpickle_data(osp.join("/mnt/disk3/wang/diffusion/datasets/partglot_data", "game_data.pkl"))
    gt_attn_maps = pickle.load(open("/mnt/disk3/wang/diffusion/PartGlot/logs/pre_trained/pn_aware/11-01_10-46-46/pred_label/final/ground_truth_attnmaps.pkl", "rb"))['gt_attn_maps']
    distractor_attn_maps = pickle.load(open("/mnt/disk3/wang/diffusion/PartGlot/logs/pre_trained/pn_aware/11-01_11-06-43/pred_label/final/distractor_attnmaps.pkl", "rb"))['gt_attn_maps']
    h5_data = h5py.File(osp.join("/mnt/disk3/wang/diffusion/datasets/partglot_data", "cic_bsp.h5"), "r")
    segs_data = h5_data["data"][:].astype(np.float32)
    print(type(segs_data))
    segs_mask = h5_data["mask"][:].astype(np.float32)
    padded_text, seq_len = pad_text_symbols_with_zeros(
            game_data["text"], 33, force_zero_end=True
        )
    labels = convert_labels_to_one_hot(game_data["target_chair"])
    geo_ids = np.array(game_data[["chair_a", "chair_b", "chair_c"]], dtype=np.int32)
    mask, part_indicator = get_mask_of_game_data(
            game_data,
            word2int,
            True,
            True,
            33,
            True,
        )
    masked_geo_ids = geo_ids[mask]
    masked_labels = labels[mask]
    masked_padded_text = padded_text[mask]
    masked_text = game_data["text"][mask]
    masked_part_indicator = part_indicator[mask]
    l = len(masked_geo_ids)
    print(l)
    out_gt = []
    out_distractor = []
    out_gt_mask = []
    out_distractor_mask = []
    out_texts = []
    out_part_indicator = []
    for idx in tqdm(range(l)):
        geo_id = torch.from_numpy(masked_geo_ids[idx])
        target = torch.tensor(np.argmax(masked_labels[idx]))
        text = torch.from_numpy(masked_padded_text[idx])
        part_ind = torch.from_numpy(masked_part_indicator[idx])
        used_gt_points, used_gt_attn_map = gt[gt_mask == 1].reshape(-1, 3), gt_attn_map[:, gt_mask==1].unsqueeze(2).expand(-1, -1, 512).reshape(4, -1).transpose(1,0)
        used_distractor_points, used_distractor_attn_map = distractor[distractor_mask == 1].reshape(-1, 3), distractor_attn_map[:, distractor_mask==1].unsqueeze(2).expand(-1, -1, 512).reshape(4, -1).transpose(1,0)
        

        geos = torch.from_numpy(segs_data[geo_id])
        geos_mask = torch.from_numpy(segs_mask[geo_id])
        gt = geos[target]
        gt_mask = geos_mask[target]
        distractor_mask = [geos_mask[i] for i in range(3) if i != target]
        distractors = [geos[i] for i in range(3) if i != target]
        out_gt += gt
        out_gt_mask += gt_mask
        out_distractor += distractors
        out_distractor_mask += distractor_mask
        out_texts += text
        out_part_indicator += part_ind
    
    out_gt, out_distractor, out_gt_mask, out_distractor_mask, out_texts, out_part_indicator = map(lambda p: torch.stack(p, dim=0).numpy(), [out_gt, out_distractor, out_gt_mask, out_distractor_mask, out_texts, out_part_indicator])
    print(out_gt.shape, out_distractor.shape, out_gt_mask.shape, out_distractor_mask.shape, out_texts.shape, out_part_indicator.shape)
    pickle.dump(dict(gt_geos=out_gt, distractor_geos=out_distractor, gt_mask=out_gt_mask, distractor_mask=out_distractor_mask, texts=out_texts, part_indicator=out_part_indicator), open("/mnt/disk3/wang/diffusion/datasets/partglot_data/processed_partglot_data.pkl", "wb"))


def process_partglot_data():
    out = pickle.load(open("/mnt/disk3/wang/diffusion/datasets/partglot_data/processed_partglot_data.pkl", "rb"))
    gt_attn_maps = pickle.load(open("/mnt/disk3/wang/diffusion/PartGlot/logs/pre_trained/pn_aware/11-01_10-46-46/pred_label/final/ground_truth_attnmaps.pkl", "rb"))['gt_attn_maps']
    distractor_attn_maps = pickle.load(open("/mnt/disk3/wang/diffusion/PartGlot/logs/pre_trained/pn_aware/11-01_11-06-43/pred_label/final/distractor_attnmaps.pkl", "rb"))['gt_attn_maps']
    gt = out['gt_geos']
    distractor = out['distractor_geos'] 
    gt_mask = out['gt_mask']
    distractor_mask = out['distractor_mask']
    texts = out['texts']
    part_indicator = out['part_indicator']
    print(gt.shape, distractor.shape, gt_mask.shape, distractor_mask.shape, texts.shape, part_indicator.shape, gt_attn_maps.shape, distractor_attn_maps.shape)
    hf = h5py.File('/mnt/disk3/wang/diffusion/datasets/partglot_data/processed_partglot_data.h5', 'w')
    hf.create_dataset("target_geos", data=gt)
    hf.create_dataset("distractor_geos", data=distractor)
    hf.create_dataset("target_geo_masks", data=gt_mask)
    hf.create_dataset("distractor_geo_masks", data=distractor_mask)
    hf.create_dataset("texts", data=texts)
    hf.create_dataset("part_indicator", data=part_indicator)
    hf.create_dataset("target_attn_maps", data=gt_attn_maps)
    hf.create_dataset("distractor_attn_maps", data=distractor_attn_maps)
    hf.close()

def subsample_partglot_data():
    data = h5py.File("/mnt/disk3/wang/diffusion/datasets/partglot_data/processed_partglot_data.h5", "r")
    
    gts = data['target_geos'][:]
    l = gts.shape[0]
    assert l % 2 == 0
    B = l //2
    print(B)
    gt_attn_maps = data['target_attn_maps'][:].reshape(B, 2, 4, 50)
    distractor_attn_maps = data['distractor_attn_maps'][:].reshape(B, 2, 4, 50)
    gts = gts.reshape(B, 2, 50, 512, 3)
    distractors = data['distractor_geos'][:].reshape(B, 2, 50, 512, 3)
    gt_masks = data['target_geo_masks'][:].reshape(B, 2, 50)
    distractor_masks = data['distractor_geo_masks'][:].reshape(B, 2, 50)
    texts = data['texts'][:].reshape(B, 2, -1)
    print(texts.shape)
    part_indicators = data['part_indicator'][:].reshape(B, 2, 4)
    
    subsampled_part_indicators = []
    subsampled_texts = []
    subsampled_gts = []
    subsampled_distractors = []
    subsampled_gt_attn_maps = []
    subsampled_distractor_attn_maps = []
    for i in tqdm(range(B)):
        gt = torch.from_numpy(gts[i]).cuda()
        assert torch.all(gt[0] == gt[1])
        gt = gt[0]
        distractor0, distractor1 = torch.split(torch.from_numpy(distractors[i]).cuda(), 1, dim=0)
        distractor0, distractor1 = distractor0.squeeze(0), distractor1.squeeze(0)
        gt_mask = torch.from_numpy(gt_masks[i, 0]).cuda()
        distractor0_mask, distractor1_mask =  torch.split(torch.from_numpy(distractor_masks[i]).cuda(), 1, dim=0)
        distractor0_mask, distractor1_mask = distractor0_mask.squeeze(0), distractor1_mask.squeeze(0)
        gt_attn_map = torch.from_numpy(gt_attn_maps[i, 0]).cuda()
        distractor0_attn_map, distractor1_attn_map = torch.split(torch.from_numpy(distractor_attn_maps[i]).cuda(), 1, dim=0)
        distractor0_attn_map, distractor1_attn_map = distractor0_attn_map.squeeze(0), distractor1_attn_map.squeeze(0)
        used_gt_points, used_gt_attn_map = gt[gt_mask == 1].reshape(-1, 3), gt_attn_map[:, gt_mask==1].unsqueeze(2).expand(-1, -1, 512).reshape(4, -1).transpose(1,0)
        used_distractor0_points, used_distractor0_attn_map = distractor0[distractor0_mask == 1].reshape(-1, 3), distractor0_attn_map[:, distractor0_mask==1].unsqueeze(2).expand(-1, -1, 512).reshape(4, -1).transpose(1,0)
        used_distractor1_points, used_distractor1_attn_map = distractor1[distractor1_mask == 1].reshape(-1, 3), distractor1_attn_map[:, distractor1_mask==1].unsqueeze(2).expand(-1, -1, 512).reshape(4, -1).transpose(1,0)
        
        if used_gt_points.shape[0] > 2048:
            idx = furthest_point_sample(used_gt_points.unsqueeze(0), 2048)
            used_gt_points = gather_operation(used_gt_points.unsqueeze(0).transpose(1,2).contiguous(), idx).squeeze(0).transpose(1,0)
            used_gt_attn_map = gather_operation(used_gt_attn_map.unsqueeze(0).transpose(1,2).contiguous(), idx).squeeze(0).transpose(1,0)
        elif used_gt_points.shape[0] < 2048:
            continue
        
        if used_distractor0_points.shape[0] > 2048:
            idxx = furthest_point_sample(used_distractor0_points.unsqueeze(0), 2048)
            used_distractor0_points = gather_operation(used_distractor0_points.unsqueeze(0).transpose(1,2).contiguous(), idxx).squeeze(0).transpose(1,0)
            used_distractor0_attn_map = gather_operation(used_distractor0_attn_map.unsqueeze(0).transpose(1,2).contiguous(), idxx).squeeze(0).transpose(1,0)
        elif used_distractor0_points.shape[0] < 2048:
            continue
        
        if used_distractor1_points.shape[0] > 2048:
            idxxx = furthest_point_sample(used_distractor1_points.unsqueeze(0), 2048)
            used_distractor1_points = gather_operation(used_distractor1_points.unsqueeze(0).transpose(1,2).contiguous(), idxxx).squeeze(0).transpose(1,0)
            used_distractor1_attn_map = gather_operation(used_distractor1_attn_map.unsqueeze(0).transpose(1,2).contiguous(), idxxx).squeeze(0).transpose(1,0)
        elif used_distractor1_points.shape[0] < 2048:
            continue

        subsampled_gts.append(used_gt_points.cpu().numpy())
        subsampled_gt_attn_maps.append(used_gt_attn_map.cpu().numpy())
        subsampled_distractors.append(torch.stack([used_distractor0_points, used_distractor1_points], dim=0).cpu().numpy())
        subsampled_distractor_attn_maps.append(torch.stack([used_distractor0_attn_map, used_distractor1_attn_map], dim=0).cpu().numpy())   
        subsampled_part_indicators.append(part_indicators[i, 0])
        subsampled_texts.append(texts[i, 0])         

    subsampled_gts = np.stack(subsampled_gts, axis=0)
    subsampled_gt_attn_maps = np.stack(subsampled_gt_attn_maps, axis=0)
    subsampled_distractors = np.stack(subsampled_distractors, axis=0)
    subsampled_distractor_attn_maps = np.stack(subsampled_distractor_attn_maps, axis=0)
    subsampled_part_indicators = np.stack(subsampled_part_indicators, axis=0)
    subsampled_texts = np.stack(subsampled_texts, axis=0)
    
    print(subsampled_gts.shape, subsampled_distractors.shape, subsampled_texts.shape, subsampled_part_indicators.shape, subsampled_gt_attn_maps.shape, subsampled_distractor_attn_maps.shape)
    
    hf = h5py.File('/mnt/disk3/wang/diffusion/datasets/partglot_data/subsampled_partglot_data_triplet.h5', 'w')
    hf.create_dataset("target_pcds", data=subsampled_gts)
    hf.create_dataset("distractor_pcds", data=subsampled_distractors)
    hf.create_dataset("texts", data=texts)
    hf.create_dataset("part_indicators", data=subsampled_part_indicators)
    hf.create_dataset("target_attn_maps", data=subsampled_gt_attn_maps)
    hf.create_dataset("distractor_attn_maps", data=subsampled_distractor_attn_maps)
    hf.close()


# preprocess()


subsample_partglot_data()
