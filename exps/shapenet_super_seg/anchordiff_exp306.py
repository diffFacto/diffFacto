# model settings
model = dict(
    type='AnchorDiffAE',
    encoder=dict(
        type='EncoderWithKLLoss',
        encoder=dict(
            type='PointNetVAEBase',
            zdim=256,
            point_dim=3,
        ),
        n_class=4,
        include_z=True,
        include_part_code=False,
        latent_flow_depth=14,
        latent_flow_hidden_dim=256,
        include_params=False,
        kl_weight=0.001,
        use_flow=True,
        use_gt_params=False
    ),
    diffusion=dict(
        type='AnchoredDiffusion',
        net = dict(
            type='PointwiseNet',
            in_channels=3,
            out_channels=3,
            res=True,
            context_dim=256,
        ),
        beta_1=1e-4,
        beta_T=.02,
        k=1.0,
        res=False,
        mode='linear',
        use_beta=True,
        learn_variance=False,
        rescale_timesteps=False,
        loss_type='mse',
        include_anchors=False,
        classifier_weight=1.,
        guidance=False,
        ddim_sampling=False,
        ddim_nsteps=25,
        ddim_discretize='quad',
        ddim_eta=1.
    ),
    sampler = dict(type='Uniform'),
    num_anchors=4,
    num_timesteps=100,
    npoints = 2048,
    
    zero_anchors=True,
    gen=True,
    ret_traj = False,
    ret_interval = 10,
    forward_sample=False,
    interpolate=False,
    save_weights=False,
    save_dir="/mnt/disk3/wang/diffusion/anchorDIff/work_dirs/pn_aware_attn_both_feat_guidance_200T_no_augment/checkpoints"
)

dataset = dict(
    train=dict(
        type="ShapeNetSeg",
        batch_size = 128,
        split='trainval',
        root='/orion/group/shapenetcore_partanno_segmentation_benchmark_v0_normal',
        npoints=2048,
        eval_mode='gen',
        scale_mode='shape_unit',
        num_workers=4,
        class_choice='Chair',
    ),
    val=dict(
        type="ShapeNetSeg",
        batch_size=32,
        split='test',
        root='/orion/group/shapenetcore_partanno_segmentation_benchmark_v0_normal',
        npoints=2048,
        shuffle=False,
        eval_mode='gen',
        scale_mode='shape_unit',
        num_workers=0,
        class_choice='Chair',
        save_only=True
    ),
)

optimizer = dict(type='Adam', lr=0.002, weight_decay=0.)

scheduler = dict(
    type='LinearLR',
    start_lr=2e-3,
    end_lr = 1e-4,
    start_epoch=4000,
    end_epoch=8000,
)

logger = dict(
    type="RunLogger")

# when we the trained model from cshuan, image is rgb
save_num_batch = 1000
max_epoch = 8000
eval_interval = 500
checkpoint_interval = 500
log_interval = 50
max_norm=10
# eval_both=True