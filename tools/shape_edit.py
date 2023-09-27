import torch 
from torch import nn
import torch.nn.functional as F
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
import math
import datetime
from anchor_diff.utils import misc
from anchor_diff.datasets.shapenet_seg import _ShapeNetSegParts

class ShapeEditor:
    def __init__(self, device, args):
        cfg = get_cfg()

        self.logger = build_from_cfg(cfg.logger,HOOKS,work_dir=cfg.work_dir)

        self.local_rank = args.local_rank
        self.prefix=args.prefix

        if args.seed is not None:
            if self.logger is not None:
                self.logger.print_log(f'Set random seed to {args.seed}, 'f'deterministic: {args.deterministic}')
            misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation

        self.cfg = cfg
        self.work_dir = cfg.work_dir
        self.save_num_batch = cfg.save_num_batch

        self.max_iter = cfg.max_iter
        self.edit_id=args.edit_id

        self.resume_path = cfg.resume_path
        self.max_norm = cfg.max_norm
        self.model = build_from_cfg(cfg.model,MODELS)
        if device=='cuda':
            self.model.to(self.local_rank)

        
        


        self.data = pickle.load(open(args.datadir, 'rb'))
        self.idx = args.idx
        self.n_class = args.n_class
            
        # only save file in the 0th process
        if  self.local_rank == 0:
            save_file = build_file(self.work_dir,prefix="config.yaml")
            save_arg = build_file(self.work_dir,prefix="args.yaml")
            save_args(save_arg, args)
            save_cfg(save_file)


        self.iter = 0

        self.total_iter = self.max_iter

        if (cfg.pretrained_weights):
            self.load(cfg.pretrained_weights, model_only=cfg.model_only)
        
        if self.resume_path is None:
            self.resume_path = search_ckpt(self.work_dir)
        if check_file(self.resume_path):
            self.resume()


    
    def optimize(self, start_time):
        if self.logger is not None:
            self.logger.print_log("Start running")
        noise = torch.randn(1, self.model.encoder.part_aligner.noise_dim).cuda()
        z = nn.Parameter(noise)
        prev = torch.zeros(1).cuda()
        lr = 1
        optimizer = Adam([z], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True, min_lr=5e-2)
        for it in range(self.max_iter):
            optimizer.zero_grad()
            input = self.data['input'][self.idx].cuda().unsqueeze(0)
            seg_mask = self.data['seg_mask'][self.idx].cuda().unsqueeze(0)
            ref_means = self.data['part_shift'][self.idx].cuda().unsqueeze(0)
            ref_vars = self.data['part_scale'][self.idx].cuda().unsqueeze(0) ** 2
            valid_id = self.data['present'][self.idx].cuda().unsqueeze(0)
            # print(valid_id.shape)
            edit_part_mean = None
            edit_part_var = ref_vars[..., self.edit_id]
            print(edit_part_var)
            edit_part_var[:, [2]] *= 1.2
            print(edit_part_var)
            # print(edit_part_mean.shape, edit_part_var.shape)
            fix_ids = torch.tensor([0, 1, 1, 1]).cuda()
            seg_flag = F.one_hot(seg_mask, num_classes=self.n_class)
            losses = self.model.edit_latent(z, input, seg_flag,valid_id, ref_means, ref_vars, fix_ids, self.edit_id, edit_part_mean, edit_part_var, fit_weight=0.05)
            all_loss, losses = parse_losses(losses)
            # all_loss =  (z ** 2).sum(1)
            # print(grad(outputs=all_loss, inputs=z))
            all_loss.backward()
            # print(z.grad, z.requires_grad)
            optimizer.step()
            scheduler.step(all_loss)
            with torch.no_grad():
                ptime = time.time()-start_time
                eta_time = (self.max_iter-it)*ptime/ (it + 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_time)))
                data = dict(
                    name = self.cfg.name,
                    lr = optimizer.param_groups[0]['lr'],
                    iter = it,
                    total_loss = all_loss,
                    eta=eta_str,
                    z_mean = z.mean(),
                )
                data.update(losses)
                data = sync(data)
                if self.logger is not None:
                    self.logger.log(data)
            if torch.allclose(all_loss, prev):
                break 
            prev = all_loss
        return z
                
    def run(self):

        self.model.eval()
        start_time = time.time()
        results = []
        z = self.optimize(start_time)
        with torch.no_grad():
            pcds = {k:v[self.idx].unsqueeze(0) for k, v in self.data.items()}
            for k, v in pcds.items():
                print(k, v.shape)
            _result = self.model.cimle_forward(pcds, noise=z.unsqueeze(1), device=self.local_rank)
            _result['latent_code'] = z.detach().cpu()
        save_dict=dict(pred=_result['pred'].detach().cpu().numpy(), ref=_result['input_ref'].detach().cpu().numpy(), seg_mask_ref=pcds['ref_seg_mask'].detach().cpu().numpy())
        save_file = build_file(self.work_dir, prefix=f"val/sample_{self.prefix}_edited_{self.idx}.pkl")
        self.logger.print_log(f"Saving to {save_file}....")
        pickle.dump(save_dict, open(save_file, 'wb'))
            
        
    
    def load(self, load_path, model_only=False):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
        resume_data = torch.load(load_path, map_location=map_location)

        if (not model_only):
            meta = resume_data.get("meta",dict())
            self.epoch = meta.get("epoch",self.epoch)
            self.iter = meta.get("iter",self.iter)
            self.max_iter = meta.get("max_iter",self.max_iter)
            self.max_epoch = meta.get("max_epoch",self.max_epoch)
            self.scheduler.load_state_dict(resume_data.get("scheduler",dict()))
            # self.optimizer.load_state_dict(resume_data.get("optimizer",dict()))
        base_ckpt = {k.replace("module.", ""): v for k, v in resume_data['model'].items()}
        self.model.load_state_dict(base_ckpt,strict=True)

        self.logger.print_log(f"Loading model parameters from {load_path}")

    def resume(self):
        self.load(self.resume_path, model_only=self.cfg.model_only)


def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher') 
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--edit_id', type=int, default=0)
    parser.add_argument('--idx', type=int, default=21)
    parser.add_argument('--n_class', type=int, default=4)
    parser.add_argument('--datadir', type=str)

    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')

    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument(
        "--prefix",
        default="exp",
        type=str
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
    
    args = parser.parse_args()

    device='cuda'

    if args.config_file:
        init_cfg(args.config_file)

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    args.world_size = 1
    

    runner = ShapeEditor(device, args)

    runner.run()


if __name__ == "__main__":
    main()