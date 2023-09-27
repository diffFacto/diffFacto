# model settings
cimle=True
cimle_cache_interval=50
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
        part_aligner=dict(
            type="PartAlignerTransformer",
            in_channels = 256,
            out_channels=6,
            n_class=4,
            d_head=32,
            depth=5,
            n_heads=8,
            dropout=0.,
            use_checkpoint=False,
            use_linear=True,
            class_cond=True,
            single_attn=True,
            add_class_cond=True,
            cimle=True,
            noise_scale=50,
            cond_noise_type=0
        ),
        n_class=4,
        kl_weight=0,
        fit_loss_type=4,
        fit_loss_weight=1.0,
        use_flow=True,
        latent_flow_depth=14, 
        latent_flow_hidden_dim=256,
        include_z=False,
        include_part_code=True,
        include_params=False,
        include_class_label=True,
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
            context_dim=256 + 4,
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
    use_input=True,
    
    gen=True,
    cimle=True,
    cimle_sample_num=1,
    ret_traj = False,
    ret_interval = 1,
    forward_sample=False,
    drift_anchors=False,
    interpolate=False,
    save_weights=False,
    save_dir="/mnt/disk3/wang/diffusion/anchorDIff/work_dirs/pn_aware_attn_both_feat_guidance_200T_no_augment/checkpoints"
)

dataset = dict(
    train=dict(
        type="ShapeNetSegPart",
        batch_size = 128,
        split='trainval',
        root='/orion/u/w4756677/datasets/colasce_data_txt',
        npoints=2048,
        scale_mode='shape_unit',
        part_scale_mode='shape_canonical',
        eval_mode='gen',
        drop_last=False,
        clip=False,
        num_workers=4,
        class_choice='Airplane',
    ),
    val=dict(
        type="ShapeNetSegPart",
        batch_size= 32,
        split='test',
        root='/orion/u/w4756677/datasets/colasce_data_txt',
        npoints=2048,
        shuffle=False,
        scale_mode='shape_unit',
        part_scale_mode='shape_canonical',
        eval_mode='gen_part',
        drop_last=False,
        clip=False,
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
resume_path="work_dirs/anchordiff_exp530_flow_normalized_retrain1_airplane_2ndstage/checkpoints/ckpt_1500.pth"
save_num_batch = 1000
max_epoch = 4000
eval_interval = 250
checkpoint_interval = 250
log_interval = 50
max_norm=10
# model_only=True
train_aligner=True
# eval_both=True