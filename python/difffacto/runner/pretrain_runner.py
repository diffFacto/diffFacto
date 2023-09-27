import torch 
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from difffacto.utils.misc import fps
from  difffacto.datasets.evaluation_utils import compute_all_metrics
from tqdm import tqdm
from difffacto.config.config import get_cfg, save_cfg, save_args
from difffacto.utils.registry import build_from_cfg,MODELS,SAMPLERS,DATASETS,OPTIMS,SCHEDULERS,HOOKS
from difffacto.utils.misc import check_file, build_file, search_ckpt, check_interval, sync, parse_losses
import pickle
import time
import datetime
from difffacto.utils import misc
from difffacto.utils import dist_utils
from difffacto.datasets.shapenet_parts import ShapeNetParts
from einops import rearrange


def normalize(pcs):
    pcs_max = pcs.max(1)[0].reshape(-1, 1, 3) # (1, 3)
    pcs_min = pcs.min(1)[0].reshape(-1, 1, 3) # (1, 3)
    pcs_shift = ((pcs_min + pcs_max) / 2).reshape(-1, 1, 3)
    pcs_scale = (pcs_max - pcs_min).reshape(-1, 1, 3) / 2
    return (pcs - pcs_shift) / pcs_scale

def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()
class PretrainRunner:
    def __init__(self, device, args):
        cfg = get_cfg()

        if args.local_rank == 0:
            self.logger = build_from_cfg(cfg.logger,HOOKS,work_dir=cfg.work_dir)
        else:
            self.logger = None
        self.distributed = args.distributed
        self.local_rank = args.local_rank
        self.world_size = args.world_size
        self.eval_both = cfg.eval_both
        self.eval_whole= cfg.eval_whole

        if args.seed is not None:
            if self.logger is not None:
                self.logger.print_log(f'Set random seed to {args.seed}, 'f'deterministic: {args.deterministic}')
            misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
        if args.distributed:
            assert args.local_rank == torch.distributed.get_rank() 
        self.cfg = cfg
        self.work_dir = cfg.work_dir
        self.save_num_batch = cfg.save_num_batch

        self.max_iter = cfg.max_iter
        assert self.max_iter is not None
        self.checkpoint_interval = cfg.checkpoint_interval
        self.eval_interval = cfg.eval_interval
        self.log_interval = cfg.log_interval
        self.resume_path = cfg.resume_path
        self.ret_traj=self.cfg.ret_traj
        self.ret_interval=self.cfg.ret_interval
        self.max_norm = cfg.max_norm
        self.model = build_from_cfg(cfg.model,MODELS)
        self.n_class = cfg.n_class
        assert self.n_class
        if device=='cuda':
            self.model.to(self.local_rank)

        if args.distributed:
            # Sync BN
            if args.sync_bn:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                if self.logger is not None:
                    self.logger.print_log('Using Synchronized BatchNorm ...')
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
            if self.logger is not None:
                self.logger.print_log('Using Distributed Data parallel ...')
        else:
            if self.logger is not None:
                self.logger.print_log('Using Data parallel ...')
            self.model = nn.DataParallel(self.model).cuda()
        
        # optimizer & scheduler
        parameters = self.model.parameters()
            
        self.optimizers = [build_from_cfg(cfg.optimizer,OPTIMS,params=list(self.model.module.encoder.encoder[i].parameters()) + list(self.model.module.diffusion[i].parameters())) for i in range(self.n_class)]
        self.schedulers = [build_from_cfg(cfg.scheduler, SCHEDULERS, optimizer=o) for o in self.optimizers]

        if self.distributed:
            assert cfg.dataset.train.batch_size % self.world_size == 0
            cfg.dataset.train.batch_size = cfg.dataset.train.batch_size // self.world_size
        
        self.train_dsets = []
        self.val_dsets = []
        self.train_bs = cfg.train_bs
        self.val_bs = cfg.val_bs
        self.npoints = cfg.npoints
        for i in range(self.n_class):
            train_dataset = ShapeNetParts(cfg.data_path_train[i], npoints=self.npoints, scale_mode=cfg.scale_mode, eval_mode='gen')
            val_dataset = ShapeNetParts(cfg.data_path_val[i], npoints=self.npoints, scale_mode=cfg.scale_mode, eval_mode='gen')
            train_sampler = DataLoader(train_dataset, cfg.train_bs, shuffle=True, num_workers=4, drop_last=True)
            val_sampler = DataLoader(val_dataset, cfg.val_bs, shuffle=False, num_workers=0)
            train_sampler = get_data_iterator(train_sampler)
            self.train_dsets.append(train_sampler)
            self.val_dsets.append(val_sampler)
        
        # only save file in the 0th process
        if  self.local_rank == 0:
            save_file = build_file(self.work_dir,prefix="config.yaml")
            save_arg = build_file(self.work_dir,prefix="args.yaml")
            save_args(save_arg, args)
            save_cfg(save_file)


        self.iter = 0
        self.epoch = 0

        self.total_iter = self.max_iter

        if (cfg.pretrained_weights):
            self.load(cfg.pretrained_weights, model_only=cfg.model_only)
        
        if self.resume_path is None:
            self.resume_path = search_ckpt(self.work_dir)
        if check_file(self.resume_path):
            self.resume()


    @property
    def finish(self):
        return self.iter>=self.max_iter
    
    def run(self):
        if self.logger is not None:
            self.logger.print_log("Start running")
        
        while not self.finish:
            self.train()
            if check_interval(self.iter,self.eval_interval) and self.local_rank == 0:
                self.val()
            if check_interval(self.iter,self.checkpoint_interval) and self.local_rank == 0:
                self.save()
    
    def train(self):
        # torch.autograd.set_detect_anomaly(True)
        self.model.train()
        l = 0
        res = {}
        for i, iterator in enumerate(self.train_dsets):
            for j in range(self.n_class):
                self.optimizers[j].zero_grad()
                
            nt = next(iterator)
            segmask = torch.eye(self.n_class).cuda()[i].reshape(1, 1, self.n_class).expand(self.train_bs, self.npoints, -1)
            losses = self.model.module.pretrain_part(nt['input'].cuda(), segmask, part_id=i, device='cuda')
            all_loss, losses = parse_losses(losses)
            all_loss.backward()
            self.optimizers[i].step()
            self.schedulers[i].step()
            l += all_loss.detach().cpu().item()
            res.update({f'part_{i}/{k}':v for k, v in losses.items()})
        if check_interval(self.iter,self.log_interval) and self.iter>0:
            data = dict(
                name = self.cfg.name,
                lr = self.optimizers[0].param_groups[0]['lr'] ,
                iter = self.iter,
                batch_size = self.train_bs,
                total_loss = l / self.n_class,
            )
            data.update(res)
            data = sync(data)
            if self.logger is not None:
                self.logger.log(data)
        self.iter+=1
        

    @torch.no_grad()
    def val(self):
        if self.logger is not None:
            self.logger.print_log("Validating....")
        self.model.eval()
        results = [[] for _ in range(self.n_class)]
        l = max([len(d) for d in self.val_dsets])
        for i in range(l):
            r = self.model.module.pretrain_validate(self.val_bs, self.npoints, 'cuda')
            for j, rr in enumerate(r):
                results[j].append(rr.detach().cpu())
        results = [torch.cat(rrr, dim=0) for rrr in results]
        results = [normalize(result) for result in results]
        for j in range(self.n_class):
            save_file = build_file(self.work_dir, prefix=f"val/pretrain_{self.iter}_part_{j}.pt")
            torch.save(results[j], save_file)
        for j in range(self.n_class):
            val_dset = self.val_dsets[j]
            ref = []
            for batch_idx, pcds in tqdm(enumerate(val_dset),total=len(val_dset)):
                ref.append(pcds['input'])
            ref = torch.cat(ref, dim=0)
            ref = normalize(ref)
            metrics = compute_all_metrics(results[j][:ref.shape[0]].cuda(), ref.cuda(), self.val_bs, accelerated_cd=True)
            eval_result = {f"part_{j}_{k}":v for k, v in metrics.items()}
            self.logger.log(eval_result,iter=self.iter)
                    
    def save(self):
        save_data = {
            "meta":{
                "iter": self.iter,
                "max_iter": self.max_iter,
                "config": self.cfg.dump()
            },
            "model":self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            "encoder": self.model.module.encoder.encoder.state_dict(),
            "scheduler": [self.schedulers[i].state_dict() for i in range(self.n_class)],
            "optimizer": [self.optimizers[i].state_dict() for i in range(self.n_class)],
        }

        save_file = build_file(self.work_dir,prefix=f"checkpoints/ckpt_{self.epoch}.pth")
        torch.save(save_data,save_file)
        
    
    def load(self, load_path, model_only=False):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
        resume_data = torch.load(load_path, map_location=map_location)

        if (not model_only):
            meta = resume_data.get("meta",dict())
            self.epoch = meta.get("epoch",self.epoch)
            self.iter = meta.get("iter",self.iter)
            self.max_iter = meta.get("max_iter",self.max_iter)
            self.max_epoch = meta.get("max_epoch",self.max_epoch)
            for i in range(self.n_class):
                self.schedulers[i].load_state_dict(resume_data.get("scheduler",[dict() for _ in range(self.n_class)])[i])
                self.optimizers[i].load_state_dict(resume_data.get("optimizer",[dict() for _ in range(self.n_class)])[i])
        #base_ckpt = {k.replace("module.", ""): v for k, v in resume_data['model'].items()}
        state_dict = resume_data['model']
        model_state_dict = self.model.state_dict()
        # print(model_state_dict.keys())
        for k in state_dict.keys():
            if k in model_state_dict.keys():
                if state_dict[k].shape != model_state_dict[k].shape:
                    self.logger.print_log(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
            else:
                self.logger.print_log(f"Dropping parameter {k}")
        for k in model_state_dict.keys():
            if k not in state_dict.keys():
                print(f"Layer {k} not loaded!")
        self.model.load_state_dict(state_dict,strict=False)

        self.logger.print_log(f"Loading model parameters from {load_path}")

    def resume(self):
        self.load(self.resume_path, model_only=self.cfg.model_only)
