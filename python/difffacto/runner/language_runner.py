import torch 
from torch import nn
import numpy as np
from difffacto.utils.misc import fps
from  difffacto.datasets.evaluation_utils import compute_all_metrics
from tqdm import tqdm
from difffacto.config.config import get_cfg, save_cfg, save_args
from difffacto.utils.registry import build_from_cfg,MODELS,SAMPLERS,DATASETS,OPTIMS,SCHEDULERS,HOOKS
from difffacto.utils.misc import check_file, build_file, search_ckpt, check_interval, sync, parse_losses
import pickle
import os
import time
import datetime
from difffacto.utils import misc
from difffacto.utils import dist_utils
from einops import rearrange
class LanguageRunner:
    def __init__(self, device, args):
        cfg = get_cfg()

        if args.local_rank == 0:
            self.logger = build_from_cfg(cfg.logger,HOOKS,work_dir=cfg.work_dir)
        else:
            self.logger = None
            
        print(cfg.work_dir)
        self.distributed = args.distributed
        self.no_eval=args.no_eval
        self.local_rank = args.local_rank
        self.world_size = args.world_size
        self.short_val = args.short_val
        self.eval_both = cfg.eval_both
        self.eval_whole= cfg.eval_whole
        self.prefix=args.prefix

        if args.seed is not None:
            if self.logger is not None:
                self.logger.print_log(f'Set random seed to {args.seed}, 'f'deterministic: {args.deterministic}')
            misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
        if args.distributed:
            assert args.local_rank == torch.distributed.get_rank() 
        self.cfg = cfg
        self.work_dir = cfg.work_dir
        self.save_num_batch = cfg.save_num_batch

        self.max_epoch = cfg.max_epoch 
        self.max_iter = cfg.max_iter
        assert (self.max_iter is None)^(self.max_epoch is None),"You must set max_iter or max_epoch"

        self.checkpoint_interval = cfg.checkpoint_interval
        self.eval_interval = cfg.eval_interval
        self.log_interval = cfg.log_interval
        self.resume_path = cfg.resume_path
        self.ret_traj=self.cfg.ret_traj
        self.ret_interval=self.cfg.ret_interval
        self.max_norm = cfg.max_norm
        self.model = build_from_cfg(cfg.model,MODELS)
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
        parameters = list(self.model.module.language_encoder.parameters()) + list(self.model.module.update_mlp.parameters()) + [self.model.module.temp]
        self.optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=parameters)
        self.scheduler = build_from_cfg(cfg.scheduler, SCHEDULERS, optimizer=self.optimizer)

        if self.distributed:
            assert cfg.dataset.train.batch_size % self.world_size == 0
            cfg.dataset.train.batch_size = cfg.dataset.train.batch_size // self.world_size
        self.train_dataset, self.train_sampler = build_from_cfg(cfg.dataset.train,DATASETS, distributed=args.distributed)
        self.val_dataset, _ = build_from_cfg(cfg.dataset.val,DATASETS, distributed=False)
        self.cimle = cfg.cimle
        self.cimle_start_epoch = cfg.cimle_start_epoch if cfg.cimle_start_epoch is not None else 0
        self.cache_interval = cfg.cimle_cache_interval
        
        # only save file in the 0th process
        if  self.local_rank == 0:
            save_file = build_file(self.work_dir,prefix="config.yaml")
            save_arg = build_file(self.work_dir,prefix="args.yaml")
            save_args(save_arg, args)
            save_cfg(save_file)


        self.iter = 0
        self.epoch = 0

        if self.max_epoch:
            if (self.train_dataset):
                self.total_iter = self.max_epoch * len(self.train_dataset)
            else:
                self.total_iter = 0
        else:
            self.total_iter = self.max_iter

        if (cfg.pretrained_weights):
            self.load(cfg.pretrained_weights, model_only=cfg.model_only)
        
        if self.resume_path is None:
            self.resume_path = search_ckpt(self.work_dir)
        if check_file(self.resume_path):
            self.resume()


    @property
    def finish(self):
        if self.max_epoch:
            return self.epoch>=self.max_epoch
        else:
            return self.iter>=self.max_iter
    
    def run(self):
        if self.logger is not None:
            self.logger.print_log("Start running")
        
        while not self.finish:
            if self.distributed:
                self.train_sampler.set_epoch(self.epoch)
            # if self.cimle and self.epoch >= self.cimle_start_epoch and check_interval(self.epoch - self.cimle_start_epoch, self.cache_interval) and self.local_rank == 0:
            #     self.cache_noise()
            self.train()
            if check_interval(self.epoch,self.eval_interval) and self.local_rank == 0:
                self.val()
            if check_interval(self.epoch,self.checkpoint_interval) and self.local_rank == 0:
                self.save()
    
    @torch.no_grad()
    def cache_noise(self):
        print("########## CACHING NOISE FOR CIMLE ##########")
        self.model.eval()
        all_noises = []
        all_ids = []
        for batch_idx, pcds in tqdm(enumerate(self.train_dataset), total=len(self.train_dataset)):
            noises = self.model.module.cache_noise(pcds, self.local_rank)
            all_noises.append(noises)
            all_ids.append(pcds['id'])
        self.train_dataset.dataset.store_noise(all_noises, all_ids)
        
    @torch.no_grad()
    def cache_noise(self):
        print("########## CACHING NOISE FOR CIMLE ##########")
        self.model.eval()
        all_noises = []
        all_ids = []
        for batch_idx, pcds in tqdm(enumerate(self.train_dataset), total=len(self.train_dataset)):
            noises = self.model.module.cache_noise(pcds, self.local_rank, eval_whole=self.eval_whole)
            all_noises.append(noises)
            all_ids.append(pcds['id'])
        self.train_dataset.dataset.store_noise(all_noises, all_ids)
        
        print("########## NOISE FOR CIMLE CACHED ##########")
    @torch.no_grad()       
    def cache_part_params(self):
        print("########## CACHING PARAMS ##########")
        self.model.eval()
        all_shifts = []
        all_scales = []
        for batch_idx, pcds in tqdm(enumerate(self.train_dataset), total=len(self.train_dataset)):
            shift, scale = self.model.module.get_params(pcds, self.local_rank)
            all_shifts.append(shift)
            all_scales.append(scale)
        self.all_shifts = all_shifts
        self.all_scales = all_scales
        print("########## PARAMS CACHED ##########")
        
        
    def train(self):
        # torch.autograd.set_detect_anomaly(True)
        self.model.eval()
        self.model.module.train_language_module()
        start_time = time.time()

        for batch_idx, pcds in enumerate(self.train_dataset):
            # pickle.dump({k:v.detach().cpu() for k, v in pcds.items()}, open("/orion/u/w4756677/diffusion/anchorDIff/weights/train_data.pkl", "wb"))
            # exit()
            
            self.optimizer.zero_grad()
            batch_size = self.train_dataset.batch_size
            
            losses = self.model.module.language_train_step(pcds, self.local_rank)
            all_loss, losses = parse_losses(losses)
            all_loss.backward()
            if self.max_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()
            if self.max_epoch is None and self.scheduler is not None:
                self.scheduler.step()
            for m in self.model.module.language_encoder.parameters():
                print(m.mean())
            
            if check_interval(self.iter,self.log_interval) and self.iter>0:
                batch_size = self.train_dataset.batch_size
                ptime = time.time()-start_time
                eta_time = (self.total_iter-self.iter)*ptime/(batch_idx+1)
                eta_str = str(datetime.timedelta(seconds=int(eta_time)))
                data = dict(
                    name = self.cfg.name,
                    lr = self.optimizer.param_groups[0]['lr'] ,
                    iter = self.iter,
                    epoch = self.epoch,
                    batch_idx = batch_idx,
                    batch_size = batch_size,
                    total_loss = dist_utils.reduce_tensor(all_loss, self.world_size) if self.distributed else all_loss,
                    eta=eta_str
                )
                data.update(dist_utils.reduce_tensor(losses, self.world_size) if self.distributed else losses)
                data = sync(data)
                if self.logger is not None:
                    self.logger.log(data)
            
            self.iter+=1
            if self.finish:
                break

        self.epoch +=1
        if self.max_epoch and self.scheduler is not None:
            self.scheduler.step()


    @torch.no_grad()
    def val(self):
        if self.val_dataset is None:
            if self.logger is not None:
                self.logger.print_log("Please set Val dataset")
        else:
            if self.logger is not None:
                self.logger.print_log("Validating....")
            # TODO: need move eval into this function
            self.model.eval()
            self.model.module.eval_language_module()
            results = dict()
            for batch_idx, pcds in tqdm(enumerate(self.val_dataset),total=len(self.val_dataset)):
                if self.short_val and batch_idx > 0: break
                _result = self.model.module.language_edit_step(pcds, device=self.local_rank)
                for result, name in _result:
                    if name not in results.keys():
                        results[name] = []
                    results[name].append(sync(result, to_numpy=False))
            if self.no_eval:
                return
            for k, v in results.items():
                save_dict, eval_result = self.val_dataset.evaluate(v, self.save_num_batch, self.local_rank)
                save_file = build_file(os.path.join("/orion/u/mikacuy/anchored_diffusion/", self.work_dir), prefix=f"val/{k}_{self.prefix}_{self.epoch}.npz")
                self.logger.print_log(f"Saving to {save_file}....")
                print(save_dict.keys())
                np.savez_compressed(save_file,  **save_dict)
                eval_result = {k:v for k, v in eval_result.items()}
                self.logger.log(eval_result,iter=self.iter)
                    
    def generate_samples(self, num_gen, param_sample_num):
        if self.val_dataset is None:
            if self.logger is not None:
                self.logger.print_log("Please set Val dataset")
            return
        if self.logger is not None:
            self.logger.print_log("Generating....")
        # TODO: need move eval into this function
        self.model.eval()
        results = dict(pred=[], ref=[],  seg_mask_ref=[])
        fixed_id = [0,0,0,0]
        B = self.val_dataset.batch_size
        from difffacto.datasets.dataset_utils import shapenet_chair_part_distribution
        valid_ids = []
        probs = []
        for k, v in shapenet_chair_part_distribution.items():
            valid_ids.append(torch.tensor(list(map(float, list(k)))))
            probs.append(v)
        probs = torch.tensor(probs)
        valid_ids = torch.stack(valid_ids, dim=0).cuda()
        K=param_sample_num
        for batch_idx in tqdm(range(num_gen // self.val_dataset.batch_size)):
            idx = torch.multinomial(probs, self.val_dataset.batch_size, replacement=True)
            valid_id = valid_ids[idx]
            ctx, mean_per_point, logvar_per_point, _seg_mask, _valid_id = self.model.module.sample(B, fixed_id, valid_id, 'cuda', self.epoch, K=K)
            variance_per_point = torch.exp(logvar_per_point)
            C, N = variance_per_point.shape[1:]
            pred =self.model.module.decode(mean_per_point, ctx=ctx, device=self.local_rank, variance=variance_per_point, anchor_assignments=_seg_mask.to(torch.int32), valid_id=_valid_id) 
            results['pred'].append(pred['pred'].cpu().detach().numpy())
            results['seg_mask_ref'].append(_seg_mask.cpu().detach().numpy())
        for batch_idx, pcds in tqdm(enumerate(self.val_dataset),total=len(self.val_dataset)):
            results['ref'].append(pcds['ref'].cpu().detach().numpy())
        results = {k:np.concatenate(v, axis=0) for k, v in results.items()}
        save_file = build_file(self.work_dir, prefix=f"val/gen_fixed{''.join(map(str,fixed_id))}_{self.prefix}_{self.epoch}.pkl")
        self.logger.print_log(f"Saving to {save_file}....")
        pickle.dump(results, open(save_file, 'wb'))
        self.evaluate_gen(results['pred'], results['ref'], self.val_dataset.batch_size)
    
    def evaluate_gen(self, pred, ref, batch_size):
        preds = []
        refs = []
        print(pred.shape, ref.shape)
        for i in tqdm(range(pred.shape[0])):
            _pred = pred[i]
            if _pred.shape[0] > 2048:
                _pred = fps(torch.from_numpy(_pred).cuda().unsqueeze(0).contiguous(), 2048).detach().cpu().numpy()[0]
            pred_max = _pred.max(0).reshape(1, 3) # (1, 3)
            pred_min = _pred.min(0).reshape(1, 3) # (1, 3)
            pred_shift = ((pred_min + pred_max) / 2).reshape(1, 3)
            pred_scale = (pred_max - pred_min).max(-1)[0].reshape(1, 1) / 2
            _pred = (_pred - pred_shift) / pred_scale
            preds.append(_pred)
        for i in tqdm(range(ref.shape[0])):
            _ref = ref[i]
            if _ref.shape[0] > 2048:
                _ref = fps(torch.from_numpy(_ref).cuda().unsqueeze(0).contiguous(), 2048).detach().cpu().numpy()[0]
            ref_max = _ref.max(0).reshape(1, 3) # (1, 3)
            ref_min = _ref.min(0).reshape(1, 3) # (1, 3)
            ref_shift = ((ref_min + ref_max) / 2).reshape(1, 3)
            ref_scale = (ref_max - ref_min).max(-1)[0].reshape(1, 1) / 2
            _ref = (_ref - ref_shift) / ref_scale
            refs.append(_ref)

                        
        preds = torch.from_numpy(np.stack(preds, axis=0)).cuda()
        refs = torch.from_numpy(np.stack(refs, axis=0)).cuda()
        print(preds.shape, refs.shape)
        metrics = compute_all_metrics(preds, refs, batch_size, one_way=False)
        print(metrics)
        self.logger.log(metrics,iter=self.iter)
        
    def save(self):
        save_data = {
            "meta":{
                "epoch": self.epoch,
                "iter": self.iter,
                "max_iter": self.max_iter,
                "max_epoch": self.max_epoch,
                "config": self.cfg.dump()
            },
            "model":self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            "encoder": self.model.module.encoder.encoder.state_dict(),
            "decoder": self.model.module.diffusion.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        if self.model.module.encoder.part_aligner is not None:
            save_data['param_sampler'] = self.model.module.encoder.part_aligner.state_dict()

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
            self.scheduler.load_state_dict(resume_data.get("scheduler",dict()))
            # self.optimizer.load_state_dict(resume_data.get("optimizer",dict()))
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
