# model settings
model = dict(
    type='AnchorDiffAE',
    encoder=dict(
        type='PartEncoder',
        encoder=dict(
            type='PointNetV2',
            zdim=256,
            point_dim=3,
            per_part_mlp=True,
        ),
        n_class=4,
        kl_weight=5e-4,
        fit_loss_type=4,
        fit_loss_weight=1.0,
        use_flow=True,
        latent_flow_depth=14, 
        latent_flow_hidden_dim=256,
        include_z=False,
        include_part_code=True,
        include_params=False,
        use_gt_params=False,
        kl_weight_annealing=False,
        gen=True,
        prior_var=1.0
    ),
    diffusion=dict(
        type='AnchoredDiffusion',
        net = dict(
            type='PointwiseNet',
            in_channels=3,
            out_channels=3,
            context_dim=256,
            res=True, 
        ),
        beta_1=1e-4,
        beta_T=.02,
        k=1.0,
        res=False,
        mode='linear',
        use_beta=True,
        rescale_timesteps=False,
        model_mean_type="epsilon",
        learn_anchor=False,
        learn_variance=False,
        loss_type='mse',
        include_anchors=False,
        include_cov=False,
        
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
    
    gen=True,
    ret_traj = False,
    ret_interval = 10,
    forward_sample=False,
    drift_anchors=False,
    interpolate=False,
    save_weights=False,
    save_dir="/mnt/disk3/wang/diffusion/anchorDIff/work_dirs/pn_aware_attn_both_feat_guidance_200T_no_augment/checkpoints"
)

dataset = dict(
    train=dict(
        type="ShapeNetSeg",
        batch_size = 128,
        split='trainval',
        root='/orion/u/w4756677/datasets/colasce_data_txt',
        npoints=2048,
        scale_mode='shape_unit',
        eval_mode='gen',
        num_workers=4,
        class_choice='Airplane',
    ),
    val=dict(
        type="ShapeNetSeg",
        batch_size= 64,
        split='trainval',
        root='/orion/u/w4756677/datasets/colasce_data_txt',
        npoints=2048,
        shuffle=False,
        scale_mode='shape_unit',
        eval_mode='gen',
        num_workers=0,
        class_choice='Airplane',
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
resume_path="/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp530_flow_uncanonicalized_retrain1_airplane/checkpoints/ckpt_8000.pth"
save_num_batch = 10000
max_epoch = 8000
eval_interval = 500
checkpoint_interval = 500
log_interval = 50
max_norm=10
# eval_both=True