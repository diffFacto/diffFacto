import argparse
from anchor_diff.runner.mixing_runner import MixingRunner 
from anchor_diff.config.config import init_cfg
from anchor_diff.utils.misc import str_list
from anchor_diff.utils import dist_utils
import torch

def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher') 
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
    '--sync_bn', 
    action='store_true', 
    default=False, 
    help='whether to use sync bn')
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
        "--no_cuda",
        action='store_true'
    )

    parser.add_argument(
        "--save_dir",
        default=".",
        type=str,
    )
    
    args = parser.parse_args()

    if args.no_cuda:
        device='cpu'
    else:
        device='cuda'

    
    if args.config_file:
        init_cfg(args.config_file)

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
        args.world_size = 1
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size

    runner = MixingRunner(device, args)

    runner.mixing()


if __name__ == "__main__":
    main()