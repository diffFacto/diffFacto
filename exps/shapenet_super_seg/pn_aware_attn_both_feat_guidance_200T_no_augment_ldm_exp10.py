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
    latent_diffusion=dict(
        type='LatentDiffusion',
        net = dict(
            type='UNet',
            in_channels=512,
            n_heads=8,
            d_head=64,
            depth=2,
            embed_dim=256,
            use_scale_shift_norm=True,
            dropout=0.1,
            prior_dim=512,
            language_dim=64,
            gated_ff=True,
            rel_pe=True
        ),
        num_timesteps=200,
        beta_1=1e-4,
        beta_T=.02,
        mode='linear',
        use_beta=False,
        rescale_timesteps=False,
        loss_type='mse',
        classifier_weight=2.,
        rescale_latent=True,
        reg_weight=1.,
        model_mean_type='epsilon', 
    ),
    language_encoder=dict(
        type="LSTM",
        text_dim=64,
        embedding_dim=100,
        num_part=4
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

    ret_traj = False,
    ret_interval = 20,
    forward_sample=False,
    interpolate=False,
    combine=False,
    language_edit=True,
    language_encoder_ckpt="/orion/u/w4756677/diffusion/anchorDIff/work_dirs/exp10/checkpoints/language_encoder_ckpt.pth",
    ldm_ckpt="/orion/u/w4756677/diffusion/anchorDIff/work_dirs/exp10/checkpoints/ldm_ckpt.pth",
    save_dir="/orion/u/w4756677/diffusion/anchorDIff/work_dirs/pn_aware_attn_both_feat_guidance_200T_no_augment/checkpoints",
    save_weights=False
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
        batch_size=32,
        split='test',
        scale_mode='global_unit',
        num_workers=4,
        contrastive_learning=False,
        eval_mode="ae"
    ),
)


optimizer = dict(type='Adam', lr=0.001, weight_decay=0.)

scheduler = dict(
    type='CustomStepLR',
    milestone=[300, 2000, 5000],
    gamma=0.5,
)

logger = dict(
    type="RunLogger")

# when we the trained model from cshuan, image is rgb
resume_path="/orion/u/w4756677/diffusion/anchorDIff/weights/ckpt_8000.pth"
model_only=True
save_num_batch = 1
max_epoch = 8000
eval_interval = 500
checkpoint_interval = 500
log_interval = 50
max_norm=10
