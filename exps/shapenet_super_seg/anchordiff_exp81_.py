# model settings
model = dict(
    type='AnchorDiffGenSuperSegments',
    encoder=dict(
        type='PointNet',
        zdim=256,
        point_dim=4+3
    ),
    decomposer=dict(
        type="ComponentMixer",
        part_latent_dim=256,
        include_attention=False,
        nheads=8,
        use_graph_attention=True, 
        use_abs_pe=False,
        include_global_feature=True,
        global_mlp_type=1,),
    diffusion=dict(
        type='AnchoredDiffusion',
        net = dict(
            type='PointwiseNet',
            in_channels=3,
            out_channels=3,
            res=True,
            context_dim=512,
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
        include_global_latent=False,
        include_anchor_latent=False,
        include_both_latent=True,
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
    loss=dict(type='L2Loss'),
    contrastive_loss=None,
    contrastive_weight=1.,
    detach_anchor=False,
    guidance=False,
    part_latent_dropout_prob=0.2,
    global_latent_dropout_prob=0.2,
    use_primary=True,
    use_zero_anchor=True,
    use_global_anchor=False,

    sample_by_seg_mask=True,
    ret_traj = False,
    ret_interval = 10,
    forward_sample=False,
    interpolate=True,
    combine=False,
    save_weights=False,
    save_dir="/mnt/disk3/wang/diffusion/anchorDIff/work_dirs/pn_aware_attn_both_feat_guidance_200T_no_augment/checkpoints"
)

dataset = dict(
    train=dict(
        type="ShapeNetSegSuperSegment",
        data_root = "/orion/u/w4756677/datasets/partglot_data",
        part='pn_aware',
        batch_size=128,
        split='train',
        scale_mode='global_unit',
        num_workers=4,
        augment=False,
        vertical_only=True
    ),
    val=dict(
        type="ShapeNetSegSuperSegment",
        data_root = "/orion/u/w4756677/datasets/partglot_data",
        part='pn_aware',
        batch_size=128,
        split='test',
        scale_mode='global_unit',
        num_workers=4,
        contrastive_learning=False,
        eval_mode="ae"
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
resume_path="/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp81/checkpoints/ckpt_8000.pth"
save_num_batch = 1
max_epoch = 8000
eval_interval = 500
checkpoint_interval = 2000
log_interval = 50
max_norm=10
