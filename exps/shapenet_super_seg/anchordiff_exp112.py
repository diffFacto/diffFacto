# model settings
model = dict(
    type='PDM',
    encoder=dict(
        type='PointNet',
        zdim=256,
        point_dim=3 + 4,
        num_anchors=1
    ),
    diffusion=dict(
        type='AnchoredDiffusion',
        net = dict(
            type='PointwiseNet',
            in_channels=3,
            out_channels=3,
            res=True,
            context_dim=256,
            include_anchors=False
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
    num_timesteps=200,
    npoints = 2048,
    include_attn_weight_in_encoder=True,
    

    ret_traj = False,
    ret_interval = 10,
    interpolate=False,
    combine=False,
    save_weights=False,
    save_dir="/mnt/disk3/wang/diffusion/anchorDIff/work_dirs/pn_aware_attn_both_feat_guidance_200T_no_augment/checkpoints"
)

dataset = dict(
    train=dict(
        type="ShapeNetSeg",
        batch_size=128,
        split='trainval',
        root='/orion/u/w4756677/datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal',
        npoints=2048,
        scale_mode='shape_unit',
        num_workers=4,
        class_choice='Chair',
        crop=1.
    ),
    val=dict(
        type="ShapeNetSeg",
        batch_size=128,
        split='test',
        root='/orion/u/w4756677/datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal',
        npoints=2048,
        mode='complete',
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
save_num_batch = 1
max_epoch = 8000
eval_interval = 500
checkpoint_interval = 2000
log_interval = 50
max_norm=10
