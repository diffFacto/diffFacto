import torch 
from torch import nn
import os
import numpy as np 
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.autograd import grad
import argparse
from tqdm import tqdm
from anchor_diff.config.config import init_cfg
from anchor_diff.config.config import get_cfg, save_cfg, save_args
from anchor_diff.utils.registry import build_from_cfg,MODELS,HOOKS
from anchor_diff.utils.misc import check_file, build_file, search_ckpt, sync, parse_losses
import pickle
import time
import datetime
from anchor_diff.utils import misc
from anchor_diff.datasets.shapenet_seg import _ShapeNetSegParts
16
point_dist_dict = {
    'Chair': {'1111': [5, 5, 5, 1], '1101':[6, 6, 0, 4], '1110':[6,5,5,0]},
    'Airplane': {'1111':[8, 4, 3, 1], '1110':[8, 5, 3, 0]},
    'Lamp': {'1101':[4, 8, 0, 4], '0111':[0, 8, 4, 4]},
    'Car': {'0111':[0, 3, 3, 10], '1111':[3, 3, 3, 7]},
}

class SampleOnePart:
    def __init__(self, device, args):
        cfg = get_cfg()
        cfg.work_dir = os.path.join('/orion/u/mikacuy/anchored_diffusion', cfg.work_dir) 
        self.logger = build_from_cfg(cfg.logger,HOOKS,work_dir=cfg.work_dir)
        self.args=args
        self.local_rank = args.local_rank

        if args.seed is not None:
            if self.logger is not None:
                self.logger.print_log(f'Set random seed to {args.seed}')
            misc.set_random_seed(args.seed + args.local_rank, deterministic=False) # seed + rank, for augmentation

        self.cfg = cfg
        self.work_dir = cfg.work_dir
        self.save_num_batch = cfg.save_num_batch

        self.max_iter = cfg.max_iter
        self.sample_points=args.sample_points
        self.resume_path = cfg.resume_path
        self.max_norm = cfg.max_norm
        self.model = build_from_cfg(cfg.model,MODELS)
        if device=='cuda':
            self.model.to(self.local_rank)
        self.cat = args.cat  
        

        self.batch_size=args.batch_size
        data = pickle.load(open(args.latent_dir, 'rb'))    
        self.codes = data['part_latents']
        self.valid_id = data['valid_id']
        self.gt_mean = data['sample 0 mean']
        self.gt_logvar = data['sample 0 logvar']     
        self.token = data['token']     
        self.length = self.codes.shape[0]  
        # only save file in the 0th process
        if  self.local_rank == 0:
            save_file = build_file(self.work_dir,prefix="config.yaml")
            save_arg = build_file(self.work_dir,prefix="args.yaml")
            save_args(save_arg, args)
            save_cfg(save_file)

        self.total_num=args.total_num if args.total_num >0 else  self.length
        if args.total_num >0:
            choice = np.random.choice(self.length, self.total_num, False)
            self.codes = self.codes[choice]
            self.valid_id = self.valid_id[choice]
            self.gt_mean = self.gt_mean[choice]
            self.gt_logvar = self.gt_logvar[choice]
            self.token = self.token[choice]
        self.length = self.codes.shape[0] 



        if (cfg.pretrained_weights):
            self.load(cfg.pretrained_weights, model_only=cfg.model_only)
        
        if self.resume_path is None:
            self.resume_path = search_ckpt(self.work_dir)
        if check_file(self.resume_path):
            self.resume()


    @torch.no_grad()
    def sample_one_part(self):
        self.model.eval()
        if self.logger is not None:
            self.logger.print_log("Start running")
        out_dump = {'pred':[], 'pred_seg_mask':[], 'valid_id':[], 'codes':[], 'noise':[], 'means':[], 'logvars':[]}
        for it in tqdm(range(self.length // self.batch_size)):
            if (it + 1) * self.batch_size > self.length:
                end = self.length 
            else:
                end = (it + 1) * self.batch_size
            codes = torch.from_numpy(self.codes[it * self.batch_size:end])
            valid_id = torch.from_numpy(self.valid_id[it * self.batch_size:end])
            gt_mean = torch.from_numpy(self.gt_mean[it * self.batch_size:end])
            gt_logvar = torch.from_numpy(self.gt_logvar[it * self.batch_size:end])    
            seg_mask = []
            for i in range(valid_id.shape[0]):
                _valid = ''.join(map(str, valid_id[i].cpu().tolist()))
                mask = point_dist_dict[self.cat].get(_valid, None)
                if mask is None:
                    _seg_mask = torch.arange(4) * valid_id[i] + torch.argmax(valid_id[i]) * (1 - valid_id[i])
                    _seg_mask = _seg_mask.reshape(4, 1).expand(-1, self.sample_points // 4).reshape(self.sample_points)
                else:
                    ratio = self.sample_points // 16
                    _seg_mask = torch.arange(4).unsqueeze(-1).repeat_interleave(torch.tensor(mask) * ratio, dim=1).reshape(self.sample_points)
                seg_mask.append(_seg_mask)
            seg_mask = torch.stack(seg_mask, dim=0).cuda().int()
            print(seg_mask.shape)
            codes = codes.cuda()
            valid_id = valid_id.cuda()
            gt_mean=gt_mean.cuda()
            gt_logvar=gt_logvar.cuda()
            pred, seg_mask, valid_id, codes, noise_latents, means, logvars = self.model.sample_one_part(codes, valid_id, gt_mean, gt_logvar, seg_mask, self.args.sample_part, self.args.how_many_each, self.args.fix_size, self.args.param_sample_num, self.args.selective_param_sample)
            out_dump['pred'].append(pred.cpu().numpy())
            out_dump['pred_seg_mask'].append(seg_mask.cpu().numpy())
            out_dump['valid_id'].append(valid_id.cpu().numpy())
            out_dump['codes'].append(codes.cpu().numpy())
            out_dump['noise'].append(noise_latents.cpu().numpy())
            out_dump['means'].append(means.cpu().numpy())
            out_dump['logvars'].append(logvars.cpu().numpy())
        out_dump = {k:np.concatenate(v, axis=0) for k, v in out_dump.items()}
        save_file = build_file(self.work_dir, prefix=f"{self.args.cat}_sample{self.args.sample_part}_{self.args.how_many_each}_{self.args.param_sample_num}.pkl")
        self.logger.print_log(f"Saving to {save_file}....")
        pickle.dump(out_dump, open(save_file, 'wb'))
        # return out_dump
                
    def run(self):

        self.model.eval()
        start_time = time.time()
        results = []
        for batch_idx, pcds in enumerate(self.loader):
            z = self.optimize(pcds, batch_idx, start_time)
            with torch.no_grad():
                _result = self.model.cimle_forward(pcds, noise=z, device=self.local_rank)
                _result['latent_code'] = z.detach().cpu()
                results.append(sync(_result, to_numpy=False))
        save_dict, eval_result = self.dataset.evaluate(results, self.save_num_batch, self.local_rank)
        save_file = build_file(self.work_dir, prefix=f"val/sample_{self.prefix}_cimle_optimized.pkl")
        self.logger.print_log(f"Saving to {save_file}....")
        pickle.dump(save_dict, open(save_file, 'wb'))
        eval_result = {k:v for k, v in eval_result.items()}
        self.logger.log(eval_result,iter=self.iter)
            
        
    
    def load(self, load_path, model_only=True):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
        resume_data = torch.load(load_path, map_location=map_location)

        base_ckpt = {k.replace("module.", ""): v for k, v in resume_data['model'].items()}
        self.model.load_state_dict(base_ckpt,strict=True)

        self.logger.print_log(f"Loading model parameters from {load_path}")

    def resume(self):
        self.load(self.resume_path, model_only=self.cfg.model_only)


def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")

    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--seed', type=int, default=0, help='random seed')


    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument(
        "--total_num",
        default=-1,
        type=int
    )

    parser.add_argument(
        "--save_dir",
        default=".",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--sample_points",
        default=8192,
        type=int,
    )
    parser.add_argument(
        "--cat",
        default='Chair',
        type=str,
    )
    parser.add_argument(
        "--latent_dir", 
        type=str
    )
    parser.add_argument(
        "--sample_part", 
        type=int
    )
    parser.add_argument(
        "--how_many_each",
        default=50, 
        type=int
    )
    parser.add_argument(
        "--param_sample_num",
        default=1, 
        type=int
    )
    parser.add_argument(
        "--selective_param_sample",
        action='store_true', 
    )
    parser.add_argument(
        "--fix_size",
        action='store_true', 
    )
    
    args = parser.parse_args()

    device='cuda'

    if args.config_file:
        init_cfg(args.config_file)

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    args.world_size = 1
    

    runner = SampleOnePart(device, args)

    runner.sample_one_part()


if __name__ == "__main__":
    main()