import torch 
from torch import nn
from difffacto.datasets.custom import CustomDataset
from difffacto.config.config import get_cfg, save_cfg, save_args
from difffacto.utils.registry import build_from_cfg,MODELS,HOOKS
from difffacto.utils.misc import check_file, build_file, search_ckpt, check_interval, sync, parse_losses
import pickle
from difffacto.utils import misc
import numpy as np
class MixingRunner:
    def __init__(self, device, args):
        cfg = get_cfg()

        if args.local_rank == 0:
            self.logger = build_from_cfg(cfg.logger,HOOKS,work_dir=cfg.work_dir)
        else:
            self.logger = None
        self.distributed = args.distributed
        self.local_rank = args.local_rank
        self.world_size = args.world_size

        if args.seed is not None:
            if self.logger is not None:
                self.logger.print_log(f'Set random seed to {args.seed}, 'f'deterministic: {args.deterministic}')
            misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
        if args.distributed:
            assert args.local_rank == torch.distributed.get_rank() 
        self.cfg = cfg
        self.work_dir = cfg.work_dir
        self.ids = cfg.ids 

        self.resume_path = cfg.resume_path
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
        self.val_dataset = CustomDataset(cfg.data_dir, npoints=cfg.npoints, scale_mode=cfg.scale_mode, part_scale_mode=cfg.part_scale_mode, clip=cfg.clip, n_class=cfg.n_class)
        self.n_class=cfg.n_class
        assert len(self.ids) == cfg.n_class
        
        # only save file in the 0th process
        if  self.local_rank == 0:
            save_file = build_file(self.work_dir,prefix="config.yaml")
            save_arg = build_file(self.work_dir,prefix="args.yaml")
            save_args(save_arg, args)
            save_cfg(save_file)



        if (cfg.pretrained_weights):
            self.load(cfg.pretrained_weights, model_only=cfg.model_only)
        
        if self.resume_path is None:
            self.resume_path = search_ckpt(self.work_dir)
        if check_file(self.resume_path):
            self.resume()


    @torch.no_grad()
    def mixing(self):
        if self.val_dataset is None:
            if self.logger is not None:
                self.logger.print_log("Please set Val dataset")
        else:
            if self.logger is not None:
                self.logger.print_log("Validating....")
            # TODO: need move eval into this function
            self.model.eval()
            datas = [self.val_dataset[i] for i in self.ids if i > 0]
            mask = [data['seg_mask'] for data in datas]
            inps = [data['input'] for data in datas]
            inps = [inps[i][mask[i] == i] if self.ids[i] > 0 else np.zeros(1) for i in range(len(self.ids))]
            inps = [torch.from_numpy(i).cuda().float() for i in inps]
            out = self.model.module.combine_latent_specific(inps, 'cuda')
            out = {k:v.cpu().numpy() for k, v in out.items()}
            save_file = build_file(self.work_dir, prefix=f"val/mixing_.pkl")
            self.logger.print_log(f"Saving to {save_file}....")
            pickle.dump(out, open(save_file, 'wb'))
                    
    def load(self, load_path, model_only=False):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
        resume_data = torch.load(load_path, map_location=map_location)

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
