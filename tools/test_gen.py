import torch 
from tqdm import tqdm
from anchor_diff.utils.misc import fps
import numpy as np
from  anchor_diff.datasets.evaluation_utils import compute_all_metrics
import argparse
import pickle
import json
import os
from datetime import datetime
def evaluate(pred, ref, batch_size):
    preds = []
    refs = []
    print(pred.shape, ref.shape)
    for i in tqdm(range(pred.shape[0])):
        _pred = pred[i]
        if _pred.shape[0] > 2048:
            _pred = fps(_pred, 2048)
        pred_max = _pred.max(0).reshape(1, 3) # (1, 3)
        pred_min = _pred.min(0).reshape(1, 3) # (1, 3)
        pred_shift = ((pred_min + pred_max) / 2).reshape(1, 3)
        pred_scale = (pred_max - pred_min).max(-1)[0].reshape(1, 1) / 2
        _pred = (_pred - pred_shift) / pred_scale
        preds.append(_pred)
    for i in tqdm(range(ref.shape[0])):
        _ref = ref[i]
        if _ref.shape[0] > 2048:
            _ref = fps(_ref, 2048)
        ref_max = _ref.max(0).reshape(1, 3) # (1, 3)
        ref_min = _ref.min(0).reshape(1, 3) # (1, 3)
        ref_shift = ((ref_min + ref_max) / 2).reshape(1, 3)
        ref_scale = (ref_max - ref_min).max(-1)[0].reshape(1, 1) / 2
        _ref = (_ref - ref_shift) / ref_scale
        refs.append(_ref)

                     
    preds = torch.from_numpy(np.stack(preds, axis=0)).cuda()
    refs = torch.from_numpy(np.stack(refs, axis=0)).cuda()
    print(preds.shape, refs.shape)
    metrics = compute_all_metrics(preds, refs, batch_size)
    print(metrics)
    return metrics
    
    
def get_data(data, num_sample, num_part_sample):
    assert num_sample % num_part_sample == 0
    ref = data['input_ref']
    out = []
    preds = []
    for i in range(10):
        preds.append(data[f'pred_sample {i}'])
    for i in range(num_sample // num_part_sample):
        out.append(preds[i][:num_part_sample])
    return np.concatenate(out, axis=0), ref
def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")



    parser.add_argument(
        "--data_dir",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
    )
    parser.add_argument(
        "--tag",
        default='exp',
        type=str,
    )
    parser.add_argument(
        "--num_sample",
        default=600,
        type=int,
    )
    parser.add_argument(
        "--num_part_sample",
        default=200,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
    )
    
    args = parser.parse_args()
    data = pickle.load(open(args.data_dir, "rb"))
    pred, ref = get_data(data, args.num_sample, args.num_part_sample)
    metrics = evaluate(pred, ref, args.batch_size)
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H:%M:%S")
    with open(os.path.join(args.save_dir, f"gen_score{dt_string}_{args.tag}.json"), "w") as outfile:
        json.dump({k: v.item() for k, v in metrics.items()}, outfile)
    


if __name__ == "__main__":
    main()