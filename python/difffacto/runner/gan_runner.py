import torch 
from torch import nn
import numpy as np
from difffacto.utils.misc import fps
from difffacto.datasets.evaluation_utils import compute_all_metrics
from difffacto.utils.gan_losses import gen_loss, dis_loss, gradient_penalty
from tqdm import tqdm
from difffacto.config.config import get_cfg, save_cfg, save_args
from difffacto.utils.registry import build_from_cfg,MODELS,SAMPLERS,DATASETS,OPTIMS,SCHEDULERS,HOOKS, GENERATORS,DISCRIMINATORS 
from difffacto.utils.misc import check_file, build_file, search_ckpt, check_interval, sync, parse_losses
import pickle
import time
import datetime
from difffacto.utils import misc
from difffacto.utils import dist_utils
from einops import rearrange
class GanRunner:
    def __init__(self, device, args):
        cfg = get_cfg()

        if args.local_rank == 0:
            self.logger = build_from_cfg(cfg.logger,HOOKS,work_dir=cfg.work_dir)
        else:
            self.logger = None
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
        self.disc=build_from_cfg(cfg.disc, DISCRIMINATORS)
        self.model = build_from_cfg(cfg.model,MODELS)
        if device=='cuda':
            self.model.to(self.local_rank)
            self.disc.to(self.local_rank)

        if args.distributed:
            # Sync BN
            if args.sync_bn:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.disc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.disc)
                if self.logger is not None:
                    self.logger.print_log('Using Synchronized BatchNorm ...')
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
            self.disc = nn.parallel.DistributedDataParallel(self.disc, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
            if self.logger is not None:
                self.logger.print_log('Using Distributed Data parallel ...')
        else:
            if self.logger is not None:
                self.logger.print_log('Using Data parallel ...')
            self.model = nn.DataParallel(self.model).cuda()
            self.disc = nn.DataParallel(self.disc).cuda()
        
        # optimizer & scheduler
        self.opt_gen = build_from_cfg(cfg.gen_optimizer,OPTIMS,params=self.model.module.encoder.part_aligner.parameters())
        self.gen_scheduler = build_from_cfg(cfg.gen_scheduler, SCHEDULERS, optimizer=self.opt_gen)
        
        self.opt_dis = build_from_cfg(cfg.disc_optimizer,OPTIMS,params=self.disc.parameters())
        self.disc_scheduler = build_from_cfg(cfg.disc_scheduler, SCHEDULERS, optimizer=self.opt_dis)

        if self.distributed:
            assert cfg.dataset.train.batch_size % self.world_size == 0
            cfg.dataset.train.batch_size = cfg.dataset.train.batch_size // self.world_size
        self.train_dataset, self.train_sampler = build_from_cfg(cfg.dataset.train,DATASETS, distributed=args.distributed)
        self.val_dataset, _ = build_from_cfg(cfg.dataset.val,DATASETS, distributed=False)
        self.n_class = cfg.n_class
        # only save file in the 0th process
        if  self.local_rank == 0:
            save_file = build_file(self.work_dir,prefix="config.yaml")
            save_arg = build_file(self.work_dir,prefix="args.yaml")
            save_args(save_arg, args)
            save_cfg(save_file)


        self.iter = 0
        self.total_gan_iters = 0
        self.n_critics = cfg.n_critics
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
            self.train()
            if check_interval(self.epoch,self.eval_interval) and self.local_rank == 0:
                self.val()
            if check_interval(self.epoch,self.checkpoint_interval) and self.local_rank == 0:
                self.save()
    
    
    def _update_gan_(self, data, gen=False, device='cuda'):
        self.model.eval()
        self.model.module.encoder.part_aligner.train()
        self.disc.train()
        self.opt_gen.zero_grad()
        self.opt_dis.zero_grad()


        input = data['input'].to(device)
        valid_id = data['present'].to(device)
        ref = data['ref'].to(device).transpose(1,2)
        seg_flag = data['ref_attn_map'].to(device)
        gt_shift = data.get('part_shift', torch.zeros(ref.shape[0], 3, self.n_class)).to(device)
        gt_var = data.get('part_scale', torch.ones(ref.shape[0], 3, self.n_class)).to(device) ** 2
        batch_size, C, N = ref.shape
        part_code, _ = self.model.module.encoder.get_part_code(input, seg_flag)
        part_code = part_code.transpose(1,2)
        n_class=part_code.shape[-1]
        noise = torch.randn(batch_size, self.model.module.encoder.part_aligner.noise_dim).cuda()
        fake_mean, fake_logvar = self.model.module.encoder.get_params_from_part_code(part_code, valid_id, gt_mean=gt_shift, gt_var=gt_var, ref=ref, noise=noise)
        x_fake = torch.cat([fake_mean, fake_logvar], dim=1)
        x_real = torch.cat([gt_shift, torch.log(gt_var)], dim=1)
        x_real.requires_grad=True
        x_inp = torch.cat([x_real, x_fake], dim=0)
        part_code = part_code.unsqueeze(0).repeat_interleave(2, dim=0).reshape(batch_size*2, -1, n_class)
        valid_id = valid_id.unsqueeze(0).repeat_interleave(2, dim=0).reshape(batch_size*2,  n_class)
        d_out = self.disc(x_inp, part_code, mask=valid_id)
        d_real = d_out[:batch_size, ...]
        d_fake = d_out[batch_size:, ...]
        # print('dfake mean', d_fake.mean())

        loss, loss_res = None, {}
        loss_type = "wgan"
        if gen:
            gen_loss_weight = self.cfg.gen_loss_weight
            loss_gen, gen_loss_res = gen_loss(
                d_real, d_fake, weight=gen_loss_weight, loss_type=loss_type)
            loss = loss_gen + (0. if loss is None else loss)
            loss.backward()
            # print("gen loss", loss)
            self.opt_gen.step()
            loss_res.update({
                ("train/gan_pass/gen/%s" % k): v
                for k, v in gen_loss_res.items()
            })
            loss_res['loss'] = loss.detach().cpu().item()
        else:
            # Get gradient penalty
            gp_weight = self.cfg.gp_weight
            if gp_weight > 0:
                gp_type = self.cfg.gp_type
                gp, gp_res = gradient_penalty(
                    x_real, x_fake, d_real, d_fake,
                    weight=gp_weight, gp_type=gp_type)
                loss = gp + (0. if loss is None else loss)
                # print("gp loss", gp)
                loss_res.update({
                    ("train/gan_pass/gp_loss/%s" % k): v
                    for k, v in gp_res.items()
                })

            dis_loss_weight = self.cfg.dis_loss_weight
            loss_dis, dis_loss_res = dis_loss(
                d_real, d_fake, weight=dis_loss_weight, loss_type=loss_type)
            loss = loss_dis + (0. if loss is None else loss)
            # print("dis loss", loss_dis)
            loss.backward()
            self.opt_dis.step()
            loss_res.update({
                ("train/gan_pass/dis/%s" % k): v for k, v in dis_loss_res.items()
            })
            loss_res['loss'] = loss.detach().cpu().item()
        # print("xfake mean and var", x_fake.mean(), x_fake.var())


        return loss_res
    
    def train(self):
        # torch.autograd.set_detect_anomaly(True)
        start_time = time.time()

        for batch_idx, pcds in enumerate(self.train_dataset):
            self.total_gan_iters += 1
            res = {}
            if self.total_gan_iters % self.n_critics == 0:
                gen_res = self._update_gan_(pcds, gen=True)
                res.update(gen_res)
            dis_res = self._update_gan_(pcds, gen=False)
            res.update(dis_res)
            batch_size = self.train_dataset.batch_size
            
            if check_interval(self.iter,self.log_interval) and self.iter>0:
                batch_size = self.train_dataset.batch_size
                ptime = time.time()-start_time
                eta_time = (self.total_iter-self.iter)*ptime/(batch_idx+1)
                eta_str = str(datetime.timedelta(seconds=int(eta_time)))
                data = dict(
                    name = self.cfg.name,
                    gen_lr = self.opt_gen.param_groups[0]['lr'] ,
                    dis_lr = self.opt_dis.param_groups[0]['lr'] ,
                    iter = self.iter,
                    epoch = self.epoch,
                    batch_idx = batch_idx,
                    batch_size = batch_size,
                    total_loss = dist_utils.reduce_tensor(res['loss'], self.world_size) if self.distributed else res['loss'],
                    eta=eta_str
                )
                data.update(dist_utils.reduce_tensor(res, self.world_size) if self.distributed else res)
                data = sync(data)
                if self.logger is not None:
                    self.logger.log(data)
            
            self.iter+=1
            if self.finish:
                break

        self.epoch +=1
        if self.max_epoch:
            if self.gen_scheduler is not None:
                self.gen_scheduler.step(epoch=self.epoch)
            if self.disc_scheduler is not None:
                self.disc_scheduler.step(epoch=self.epoch)


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
            results = dict()
            for batch_idx, pcds in tqdm(enumerate(self.val_dataset),total=len(self.val_dataset)):
                if self.short_val and batch_idx > 0: break
                _result = self.model(pcds, device=self.local_rank)
                for result, name in _result:
                    if name not in results.keys():
                        results[name] = []
                    results[name].append(sync(result, to_numpy=False))
            if self.no_eval:
                return
            for k, v in results.items():
                save_dict, eval_result = self.val_dataset.evaluate(v, self.save_num_batch, self.local_rank)
                save_file = build_file(self.work_dir, prefix=f"val/{k}_{self.prefix}_{self.epoch}.pkl")
                self.logger.print_log(f"Saving to {save_file}....")
                pickle.dump(save_dict, open(save_file, 'wb'))
                eval_result = {k:v for k, v in eval_result.items()}
                self.logger.log(eval_result,iter=self.iter)
            if self.eval_both and not self.short_val:
                results = dict()
                for batch_idx, pcds in tqdm(enumerate(self.train_dataset),total=len(self.train_dataset)):
                    if self.short_val and batch_idx > 0: break
                    _result = self.model(pcds, device=self.local_rank)
                    for result, name in _result:
                        if name not in results.keys():
                            results[name] = []
                        results[name].append(sync(result, to_numpy=False))
                if self.no_eval:
                    return
                for k, v in results.items():
                    save_dict, eval_result = self.val_dataset.evaluate(v, self.save_num_batch, self.local_rank)
                    save_file = build_file(self.work_dir, prefix=f"val/train_set_{k}_{self.prefix}_{self.epoch}.pkl")
                    self.logger.print_log(f"Saving to {save_file}....")
                    pickle.dump(save_dict, open(save_file, 'wb'))
                    eval_result = {f"train_{k}":v for k, v in eval_result.items()}
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
            "discrimator":self.disc.module.state_dict() if self.distributed else self.disc.state_dict(),
        }
        if self.model.module.encoder.part_aligner is not None:
            save_data['param_sampler'] = self.model.module.encoder.part_aligner.state_dict()
        if self.opt_gen is not None:
            save_data['opt_gen'] = self.opt_gen.state_dict()
        if self.gen_scheduler is not None:
            save_data['gen_scheduler'] = self.gen_scheduler.state_dict()
        if self.opt_dis is not None:
            save_data['opt_dis'] = self.opt_dis.state_dict()
        if self.disc_scheduler is not None:
            save_data['disc_scheduler'] = self.disc_scheduler.state_dict()
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
