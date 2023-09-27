# model settings
model = dict(
    type='AnchorDiffGenSuperSegments',
    encoder=dict(
        type='PCN',
        part_latent_dim=512,
        point_dim=4+3
    ),
    decomposer=dict(
        type="ComponentMixer",
        part_latent_dim=512,
        include_attention=True,
        nheads=8,
        use_graph_attention=True, 
        use_abs_pe=True,
        include_global_feature=True
    ),
    diffusion=dict(
        type='AnchoredDiffusion',
        net = dict(
            type='PointwiseNet',
            in_channels=6,
            out_channels=3,
            res=False,
            context_dim=1024,
        ),
        beta_1=1e-4,
        beta_T=.05,
        k=1.0,
        res=True,
        mode='linear',
        use_beta=True,
        rescale_timesteps=False,
        loss_type='mse',
        include_anchors=True,
        include_global_latent=False,
        include_anchor_latent=False,
        include_both_latent=True,
        classifier_weight=2.,
        cond_on_global_latent=True,
        cond_on_anchor_latent=True,
        ddim_sampling=True,
        ddim_nsteps=25,
        ddim_discretize='quad',
        ddim_eta=1.
    ),
    sampler = dict(type='Uniform'),
    num_anchors=4,
    num_timesteps=100,
    npoints = 2048,
    anchor_loss_weight=1.,
    loss=dict(type='L2Loss'),
    contrastive_loss=None,
    contrastive_weight=1.,
    detach_anchor=False,
    guidance=True,
    part_latent_dropout_prob=0.2,
    global_latent_dropout_prob=0.2,
    sample_by_seg_mask=True,
    language_edit=False,

    ret_traj = False,
    ret_interval = 10,
    forward_sample=False,
    interpolate=False,
    combine=False,
)

dataset = dict(
    train=dict(
        type="ShapeNetSegSuperSegment",
        data_root = "/mnt/disk3/wang/diffusion/datasets/partglot_data/",
        part='pn_aware',
        batch_size=128,
        split='train',
        scale_mode='global_unit',
        num_workers=4,
        augment=True,
        vertical_only=True
    ),
    val=dict(
        type="ShapeNetSegSuperSegment",
        data_root = "/mnt/disk3/wang/diffusion/datasets/partglot_data/",
        part='pn_aware',
        batch_size=128,
        split='test',
        scale_mode='global_unit',
        num_workers=4,
        contrastive_learning=False,
        eval_mode='ae'
    ),
)


optimizer = dict(type='Adam', lr=0.001, weight_decay=0.)

scheduler = dict(
    type='StepLR',
    step_size=300,
    gamma=0.5
)

logger = dict(
    type="RunLogger")

# when we the trained model from cshuan, image is rgb
resume_path = "/mnt/disk3/wang/diffusion/anchorDIff/work_dirs/pn_aware_attn_both_feat_guidance/checkpoints/ckpt_1200.pth"
save_num_batch = 10
max_epoch = 1200
eval_interval = 100
checkpoint_interval = 100
log_interval = 50