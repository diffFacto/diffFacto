# model settings
import torch
model = dict(
    type='AnchorDiffAE',
    encoder=dict(
        type='PartEncoder',
        encoder=dict(
            type='PointNetV2',
            zdim=256,
            point_dim=3,
        ),
        mixer=dict(
            type="PartCodeTransformer",
            device='cuda' if torch.cuda.is_available() else 'cpu',
            input_channels=256,
            n_ctx=4,
            output_channels=256,
            width=256,
            layers=5,
            heads=8,
            init_scale=0.25,
            class_cond=True,
            param_cond=True,
        ),
        part_aligner=dict(
            type="PartAligner",
            n_class=4,
            width=256
        ),
        include_var=False,
        n_class=4
    ),
    diffusion=dict(
        type='AnchoredDiffusion',
        net = dict(
            type='PointwiseNet',
            in_channels=3,
            out_channels=3,
            res=True,
            context_dim=256,
            include_anchors=False,
            include_variances=False
        ),
        beta_1=1e-4,
        beta_T=.05,
        k=1.0,
        res=False,
        mode='linear',
        use_beta=True,
        rescale_timesteps=False,
        loss_type='mse',
        include_anchors=False,
        include_global_latent=True,
        include_anchor_latent=False,
        include_both_latent=False,
        classifier_weight=1.,
        cond_on_global_latent=True,
        cond_on_anchor_latent=True,
        ddim_sampling=False,
        ddim_nsteps=25,
        ddim_discretize='quad',
        ddim_eta=1.
    ),
    sampler = dict(type='Uniform'),
    num_anchors=4,
    num_timesteps=200,
    npoints = 2048,
    anchor_loss_weight=1.,
    
    ret_traj = False,
    ret_interval = 10,
    forward_sample=False,
    interpolate=True,
    save_weights=False,
    save_dir="/mnt/disk3/wang/diffusion/anchorDIff/work_dirs/pn_aware_attn_both_feat_guidance_200T_no_augment/checkpoints"
)

dataset = dict(
    train=dict(
        type="ShapeNetSegPart",
        batch_size = 128,
        split='trainval',
        root='/orion/group/shapenetcore_partanno_segmentation_benchmark_v0_normal',
        npoints=2048,
        scale_mode='shape_unit',
        num_workers=4,
        class_choice='Chair',
    ),
    val=dict(
        type="ShapeNetSegPart",
        batch_size=128,
        split='test',
        root='/orion/group/shapenetcore_partanno_segmentation_benchmark_v0_normal',
        npoints=2048,
        shuffle=False,
        scale_mode='shape_unit',
        num_workers=0,
        class_choice='Chair',
        save_only=True
    ),
)

optimizer = dict(type='Adam', lr=0.001, weight_decay=0.)

scheduler = dict(
    type='StepLR',
    step_size=2666,
    gamma=0.5,
)

logger = dict(
    type="RunLogger")

# when we the trained model from cshuan, image is rgb
resume_path="/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp187/checkpoints/ckpt_4000.pth"
save_num_batch = 1
max_epoch = 8000
eval_interval = 500
checkpoint_interval = 2000
log_interval = 50
max_norm=10
eval_both=True