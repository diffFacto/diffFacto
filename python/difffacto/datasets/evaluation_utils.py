import torch
import numpy as np
import torch.nn.functional as F
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from tqdm.auto import tqdm
# Import CUDA version of approximate EMD,
# from https://github.com/zekunhao1995/pcgan-pytorch/
from difffacto.metrics.chamfer_dist import ChamferDistanceL2_split
from difffacto.metrics.emd.emd_module import EMD
from difffacto.utils.misc import fps
from functools import partial
from .iou import get_3d_box, box3d_iou

def distChamferCUDA(x, y):
    func = ChamferDistanceL2_split(reduce=False)
    return func(x, y)



def part_chamfer(n_class, A, B, accelerated=False):
    dist = []
    for i in range(n_class):
        _A = A.get(i, None)
        _B = B.get(i, None)
        if (_A is not None) != (_B is not None):
            return torch.ones(1) * float('inf')
        if _A is None:
            continue
        _A_points = torch.rand([512, 3]) * (_A[1] - _A[0]) + _A[0]
        _B_points = torch.rand([512, 3]) * (_B[1] - _B[0]) + _B[0]
        if accelerated:
            _dl, _dr = distChamferCUDA(_A_points[None].cuda(), _B_points[None].cuda())
        else:
            _dl, _dr = distChamfer(_A_points[None], _B_points[None])
        dist.append(_dl.mean() + _dr.mean())
    dist = torch.tensor(dist).mean()
    return dist 

def part_l2(n_class, A, B, accelerated=False):
    dist = []
    for i in range(n_class):
        _A = A.get(i, None)
        _B = B.get(i, None)
        if (_A is not None) != (_B is not None):
            return torch.ones(1) * float('inf')
        if _A is None:
            continue
        A_max = _A[1][0]
        A_min = _A[0][0]
        B_max = _B[1][0]
        B_min = _B[0][0]
        scale_A = (A_max - A_min) / 2.0 
        scale_B = (B_max - B_min) / 2.0
        shift_A = (A_max + A_min) / 2.0
        shift_B = (B_max + B_min) / 2.0 
        d = F.mse_loss(torch.cat([scale_A, shift_A], dim=0), torch.cat([scale_B, shift_B], dim=0))
        dist.append(d)
    dist = torch.tensor(dist).mean()
    return dist 

def part_miou(n_class, A, B, accelerated=True):
    dist = []
    for i in range(n_class):
        _A = A.get(i, None)
        _B = B.get(i, None)
        if (_A is not None) != (_B is not None):
            return torch.ones(1) * float('inf')
        if _A is None:
            continue
        A_max = _A[1][0].numpy()
        A_min = _A[0][0].numpy()
        B_max = _B[1][0].numpy()
        B_min = _B[0][0].numpy()
        A_box  = get_3d_box(A_max - A_min, 0, (A_max + A_min) / 2.0) 
        B_box  = get_3d_box(B_max - B_min, 0, (B_max + B_min) / 2.0) 
        iou_3d, _ = box3d_iou(A_box, B_box)
        dist.append(iou_3d)
    dist = torch.tensor(dist).mean()
    return 1.0 - dist 

def emd_approx(sample, ref):
    B, N, N_ref = sample.size(0), sample.size(1), ref.size(1)
    assert N == N_ref, "Not sure what would EMD do in this case"
    emd = EMD(0.002, 10000, True)
    cost = emd(sample, ref)  # (B,)
    return cost


# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def EMD_CD(sample_pcs, ref_pcs, batch_size, accelerated_cd=False, reduced=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)

    for b_start in tqdm(iterator, desc='EMD-CD'):
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        if accelerated_cd:
            dl, dr = distChamferCUDA(sample_batch, ref_batch)
        else:
            dl, dr = distChamfer(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

        emd_batch = emd_approx(sample_batch, ref_batch)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)

    results = {
        'MMD-CD': cd,
        'MMD-EMD': emd,
    }
    return results


def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size,
                      accelerated_cd=True, verbose=True, mask_sample=None, mask_ref=None):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)
    if verbose:
        iterator = tqdm(iterator, desc='Pairwise EMD-CD')
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]
        if mask_sample is not None:
            _mask_sample = mask_sample[sample_b_start]
        cd_lst = []
        emd_lst = []
        sub_iterator = range(0, N_ref, batch_size)
        if verbose:
            sub_iterator = tqdm(sub_iterator, leave=False)
        for ref_b_start in sub_iterator:
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]
            if mask_ref is not None:
                _mask_ref = mask_ref[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            point_dim = ref_batch.size(2)
            sample_batch_exp = sample_batch.view(1, -1, point_dim).expand(
                batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            if accelerated_cd:
                dl, dr = distChamferCUDA(sample_batch_exp, ref_batch)
                    
            else:
                dl, dr = distChamfer(sample_batch_exp, ref_batch)
            if mask_sample is not None:
                dl_mean = (dl * _mask_sample.unsqueeze(0)).sum(1) / _mask_sample.sum()
            else:
                dl_mean = dl.mean(1)
            if mask_ref is not None:
                dr_mean = (dr * _mask_ref).sum(1) / _mask_ref.sum(1)
            else:
                dr_mean = dr.mean(1)
            cd_lst.append((dl_mean + dr_mean).view(1, -1))

            emd_batch = emd_approx(sample_batch_exp, ref_batch)
            emd_lst.append(emd_batch.view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)
        # print(cd_lst.min(1)[0])

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


# Adapted from https://github.com/xuqiantong/
# GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False, one_way=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    print(n0, n1)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat([
        torch.cat((Mxx, Mxy), 1),
        torch.cat((Mxy.transpose(0, 1), Myy), 1)], 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(
        k, 0, False)
    print(val.shape, idx.shape)
    print(val[0, :n0].mean(), val[0, n0:].mean())
    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()
    if one_way:
        pred = pred[:n0]
        label = pred[:n0]

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


def lgan_mmd_cov(all_dist, thresh=1000):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, idx = torch.min(all_dist, dim=0)
    min_val, idxx = torch.sort(min_val)
    print("Min dist of test shape to generated sample")
    print(min_val, idx[idxx], idxx)
    min_val_fromsmp, idxxx = torch.sort(min_val_fromsmp)
    print("Min dist of generated sample to test shape")
    torch.set_printoptions(profile="full")
    print(min_val_fromsmp, min_idx[idxxx], idxxx)
    torch.set_printoptions(profile="default")
    sorted_idx = idx[idxx]
    outlier_mask = min_val > thresh 
    if torch.any(outlier_mask):
        sorted_idx[outlier_mask] = sorted_idx[0]
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(sorted_idx.unique().view(-1).size(0)) / float(N_ref)
    print(sorted_idx.unique().view(-1))
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }


def lgan_mmd_cov_match(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }, min_idx.view(-1)

def compute_bbox_metric(preds, preds_mask, refs, refs_mask, batch_size, n_class=4, thresh=1.0, metric='chamfer', no_nn=False, cov_thresh=100):
    print("Bbox EMD CD")
    pred_size = preds.shape[0]
    ref_size = refs.shape[0]
    assert pred_size == preds_mask.shape[0]
    assert ref_size == refs_mask.shape[0]
    pred_params = []
    ref_params = []
    for i in range(pred_size):
        _pred = preds[i]
        _pred_seg_mask = preds_mask[i]
            
        _pred_max = _pred.max(0)[0].reshape(1,3) # (1, 3)
        _pred_min = _pred.min(0)[0].reshape(1,3) # (1, 3)
        _pred_shift = ((_pred_min + _pred_max) / 2).reshape(1, 3)
        _pred_scale = (_pred_max - _pred_min).max(-1)[0].reshape(1, 1) / 2
        _pred = (_pred - _pred_shift) / _pred_scale
        one_pred_params = dict()
        for j in range(n_class):
            idx = _pred_seg_mask == j 
            if torch.any(idx):
                points = _pred[idx]
                if points.shape[0] > 100:
                    part_max = torch.quantile(points, thresh, dim=0, keepdim=True).cpu()
                    part_min =  torch.quantile(points, 1 - thresh, dim=0, keepdim=True).cpu()
                    one_pred_params[j] = (part_min, part_max)
        pred_params.append(one_pred_params)
        
    for i in range(ref_size):
        _ref = refs[i]
        _ref_max = _ref.max(0)[0].reshape(1,3) # (1, 3)
        _ref_min = _ref.min(0)[0].reshape(1,3) # (1, 3)
        _ref_shift = ((_ref_min + _ref_max) / 2).reshape(1, 3)
        _ref_scale = (_ref_max - _ref_min).max(-1)[0].reshape(1, 1) / 2
        _ref = (_ref - _ref_shift) / _ref_scale
        _ref_seg_mask = refs_mask[i]
        one_ref_params = dict()
        for j in range(n_class):
            ref_idx = _ref_seg_mask == j
            if torch.any(ref_idx):
                ref_points = _ref[ref_idx]
                if ref_points.shape[0] > 100:
                    ref_part_max = torch.quantile(ref_points, thresh, dim=0, keepdim=True).cpu()
                    ref_part_min =  torch.quantile(ref_points, 1 - thresh, dim=0, keepdim=True).cpu()
                    one_ref_params[j] = (ref_part_min, ref_part_max)

        ref_params.append(one_ref_params)
    if metric == 'chamfer':
        metric = compute_all_metrics_cust_func(pred_params, ref_params, partial(part_chamfer, n_class), 'bbox_chamfer', accelerated_cd=True, no_nn=no_nn, thresh=cov_thresh)
    elif metric == 'iou':
        metric = compute_all_metrics_cust_func(pred_params, ref_params, partial(part_miou, n_class), 'bbox_iou', accelerated_cd=True, no_nn=no_nn, thresh=cov_thresh)
    elif metric == 'l2':
        metric = compute_all_metrics_cust_func(pred_params, ref_params, partial(part_l2, n_class), 'bbox_l2', accelerated_cd=True, no_nn=no_nn, thresh=cov_thresh)
    else:
        raise NotImplementedError
    metric = {f"bbox_{k}":v for k, v in metric.items()}
    
    return metric

def compute_all_metrics_cust_func(sample_pcs, ref_pcs, dist_func, dist_name, accelerated_cd=False, no_nn=False, thresh=1000):
    results = {}
    
    print("Pairwise EMD CD")
    M = len(sample_pcs)
    N = len(ref_pcs)
    rs_dists = torch.zeros([N, M])
    for i in tqdm(range(N)):
        for j in range(M):
            rs_dists[i, j] = dist_func(ref_pcs[i], sample_pcs[j], accelerated=accelerated_cd)
            
    res_cd = lgan_mmd_cov(rs_dists.t(), thresh=cov_thresh)
    results.update({
        f"{k}-{dist_name}": v for k, v in res_cd.items()
    })
    
    for k, v in results.items():
        print('[%s] %.8f' % (k, v.item()))
    if no_nn:
        return results
    rr_dists = torch.zeros([N, N])
    for i in tqdm(range(N)):
        for j in range(N):
            rr_dists[i, j] = dist_func(ref_pcs[i], ref_pcs[j], accelerated=accelerated_cd)
            
    ss_dists = torch.zeros([M, M])
    for i in tqdm(range(M)):
        for j in range(M):
            ss_dists[i, j] = dist_func(sample_pcs[i], sample_pcs[j], accelerated=accelerated_cd)
            
       
    # 1-NN results
    one_nn_cd_res = knn(rr_dists, rs_dists, ss_dists, 1, sqrt=False)
    results.update({
        f"1-NN-{dist_name}-{k}": v for k, v in one_nn_cd_res.items() if 'acc' in k
    })

    return results

def compute_snapping_metric(preds, preds_mask, cls='Chair'):
    print("Snapping Metric")
    if cls == 'Chair':
        connected = [(0, [1,2]), (1, [2]), (3, [0, 1])]
    elif cls == 'Airplane':
        connected = [(1, [0]), (2, [0]), (3, [0, 1])]
    dists = {p[0]:[] for p in connected}
    for k in tqdm(range(preds.shape[0])):
        pred = preds[k]
        pred_mask = preds_mask[k]
        min_d = []
        for i, js in connected:
            id_part_A = pred_mask == i 
            min_d = []
            for j in js:
                id_part_B = pred_mask == j 
                if torch.any(id_part_A) and torch.any(id_part_B):
                    point_A = pred[id_part_A]
                    point_B = pred[id_part_B]
                    dist = ((point_A[:, None] - point_B[None]) ** 2).sum(-1)
                    min_point_A_val, id_min_point_A = dist.min(1)[0].topk(50,dim=0, largest=False)
                    min_point_B_val, id_min_point_B = dist.min(0)[0].topk(50,dim=0, largest=False)
                    min_point_A = point_A[id_min_point_A]
                    min_point_B = point_B[id_min_point_B]
                    dl, dr = distChamfer(min_point_A[None], min_point_B[None])
                    min_d.append(dl.mean(1) + dr.mean(1))
            if len(min_d):
                min_d = torch.min(torch.tensor(min_d)).item()
                dists[i].append((k, min_d)) 
                # print(dist)
    d_array = torch.tensor([d[1] for d in dists[1]])
    d_ind_array = torch.tensor([d[0] for d in dists[1]])
    print(d_array.sort()[0], d_ind_array[d_array.sort()[1]])
    dists = {f"snapping_{cls}_{k}":torch.tensor(list(zip(*v))[1]).mean() for k, v in dists.items()}
    return dists
                    
            

def compute_part_metric(preds, preds_mask, refs, refs_mask, batch_size, n_class=4):
    print('part level Pairwise EMD CD')
    pred_size = preds.shape[0]
    ref_size = refs.shape[0]
    pred_part_pointcloud = [[] for _ in range(n_class)]
    pred_part_mask = [[] for _ in range(n_class)]
    ref_part_pointcloud = [[] for _ in range(n_class)]
    assert pred_size == preds_mask.shape[0]
    assert ref_size == refs_mask.shape[0]
    for i in tqdm(range(pred_size)):
        _pred = preds[i]
        _pred_seg_mask = preds_mask[i]
        for j in range(n_class):
            idx = _pred_seg_mask == j 
            if torch.any(idx):
                points = _pred[idx].unsqueeze(0)
                if points.shape[1] > 100:
                    mask = torch.ones([1, 512])
                    if points.shape[1] < 512:
                        mask[:, points.shape[1]:] = 0
                    while points.shape[1] < 512:
                        points = torch.cat([points, points], dim=1)
                    if points.shape[1] > 512:
                        points = points[:, :512]
                    pred_part_pointcloud[j].append(points)
                    assert (mask.shape == (1, 512))
                    pred_part_mask[j].append(mask)
    for i in tqdm(range(ref_size)):
        _ref = refs[i]
        _ref_seg_mask = refs_mask[i]
        for j in range(n_class):
            idx = _ref_seg_mask == j 
            if torch.any(idx):
                points = _ref[idx].unsqueeze(0)
                if points.shape[1] > 100:
                    while points.shape[1] < 512:
                        points = torch.cat([points, points], dim=1)
                    if points.shape[1] > 512:
                        points = points[:, :512]
                    ref_part_pointcloud[j].append(points)
    pred_part_pointcloud, ref_part_pointcloud, pred_part_mask = map(lambda l: [torch.cat(ll, dim=0) for ll in l], [pred_part_pointcloud, ref_part_pointcloud, pred_part_mask])
    weight = []
    for l, s, k in zip(pred_part_pointcloud, ref_part_pointcloud, pred_part_mask):
        print(s.shape, l.shape, k.shape)
        weight.append(s.shape[0])
    N = sum(weight)
    weight = [w / N for w in weight]
    print('Normalizing------------------')
    for i in range(4):
        _pred = pred_part_pointcloud[i]
        _pred_max = _pred.max(1)[0].reshape(-1, 1,3) # (1, 3)
        _pred_min = _pred.min(1)[0].reshape(-1, 1,3) # (1, 3)
        _pred_shift = ((_pred_min + _pred_max) / 2).reshape(-1, 1, 3)
        _pred_scale = (_pred_max - _pred_min).reshape(-1, 1, 3) / 2
        _pred = (_pred - _pred_shift) / _pred_scale
        pred_part_pointcloud[i] = _pred
        _ref = ref_part_pointcloud[i]
        _ref_max = _ref.max(1)[0].reshape(-1, 1,3) # (1, 3)
        _ref_min = _ref.min(1)[0].reshape(-1, 1,3) # (1, 3)
        _ref_shift = ((_ref_min + _ref_max) / 2).reshape(-1, 1, 3)
        _ref_scale = (_ref_max - _ref_min).reshape(-1, 1, 3) / 2
        _ref = (_ref - _ref_shift) / _ref_scale
        ref_part_pointcloud[i] = _ref
    metrics = []
    for i in range(n_class):
        print(f"Part {i} --------------------------------")
        _pred = pred_part_pointcloud[i]
        _ref = ref_part_pointcloud[i]
        _pred_mask = pred_part_mask[i].cuda()
        metric = compute_all_metrics(_pred, _ref, 32, mask=_pred_mask)
        metrics.append(metric)
    avg_m = {f"part_weighted_{k}":0 for k in metrics[0].keys()}
    for i, m in enumerate(metrics):
        for k, v in m.items():
            avg_m[f"part_weighted_{k}"] += v * weight[i]
    return avg_m

def compute_all_metrics(sample_pcs, ref_pcs, batch_size, accelerated_cd=True, one_way=False, mask=None):
    results = {}
    
    print("Pairwise EMD CD")
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(
        ref_pcs, sample_pcs, batch_size, accelerated_cd=accelerated_cd, mask_ref=mask)
    print(M_rs_cd.min(1)[0].mean(), M_rs_cd.min(0)[0].mean())
    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({
        "%s-CD" % k: v for k, v in res_cd.items()
    })

    res_emd = lgan_mmd_cov(M_rs_emd.t())
    results.update({
        "%s-EMD" % k: v for k, v in res_emd.items()
    })

    for k, v in results.items():
        print('[%s] %.8f' % (k, v.item()))

    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(
        ref_pcs, ref_pcs, batch_size, accelerated_cd=accelerated_cd)
    
    if not one_way:
        M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(
            sample_pcs, sample_pcs, batch_size, accelerated_cd=accelerated_cd, mask_ref=mask, mask_sample=mask)
    else:
        INFINITY = float('inf')
        M_ss_cd =  torch.zeros(M_rs_cd.shape[1], M_rs_cd.shape[1]).cuda() + INFINITY
        M_ss_emd = torch.zeros(M_rs_cd.shape[1], M_rs_cd.shape[1]) + INFINITY
        
    # 1-NN results
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False, one_way=one_way)
    results.update({
        "1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k
    })
    one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False, one_way=one_way)
    results.update({
        "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
    })

    return results


#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with
    resolution^3 cells, that is placed in the unit-cube. If clip_sphere it True
    it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jsd_between_point_cloud_sets(
        sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds,
       as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(
        sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(
        ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(
        pclouds, grid_resolution, in_sphere=False, verbose=False):
    """Given a collection of point-clouds, estimate the entropy of
    the random variables corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in tqdm(pclouds, desc='JSD'):
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    """another way of computing JSD"""

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))


if __name__ == '__main__':
    a = torch.randn([16, 2048, 3]).cuda()
    b = torch.randn([16, 2048, 3]).cuda()
    print(EMD_CD(a, b, batch_size=8))