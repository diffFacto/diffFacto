# model settings
model = dict(
    type='AnchorDiffGenSuperSegments',
    encoder=dict(
        type='PCN',
        part_latent_dim=512,
        point_dim=3
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
        mode='linear',
        use_beta=True,
        rescale_timesteps=False,
        loss_type='mse',
        include_anchors=True,
        res=False,
        include_global_latent=False,
        include_anchor_latent=False,
        include_both_latent=True,
        
    ),
    sampler = dict(type='Uniform'),
    num_anchors=4,
    num_timesteps=100,
    npoints = 8192,
    anchor_loss_weight=1.,
    loss=dict(type='L2Loss'),
    include_attn_weight_in_encoder=False,
    detach_anchor=False,


    ret_traj = False,
    ret_interval = 10,
    forward_sample=False,
    interpolate=True,
    combine=False,
)

dataset = dict(
    train=dict(
        type="ShapeNetSegSuperSegment",
        data_root = "/mnt/disk/wang/diffusion/datasets/partglot_data/",
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
        data_root = "/mnt/disk/wang/diffusion/datasets/partglot_data/",
        part='pn_aware',
        batch_size=128,
        split='test',
        scale_mode='global_unit',
        num_workers=4,
        augment=False
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
save_num_batch = 1
max_epoch = 1200
eval_interval = 100
checkpoint_interval = 100
log_interval = 50
