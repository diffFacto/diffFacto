# model settings
import torch
model = dict(
    type='AnchorDiffAE',
    encoder=dict(
        type='PartEncoderForTransformerDecoderKLLoss',
        encoder=dict(
            type='PointNetV2VAE',
            zdim=256,
            point_dim=3,
        ),
        part_aligner=dict(
            type="PartAligner",
            n_class=4,
            width=256
        ),
        n_class=4,
        include_var=True,
        kl_weight=0.001,
        fit_loss_weight=1.0,
        include_z=False,
        include_part_code=True,
        include_params=False,
        part_code_dropout_prob=0.0,
        param_dropout_prob=0.0
    ),
    diffusion=dict(
        type='AnchoredDiffusion',
        net = dict(
            type='TransformerNet',
            in_channels=3,
            out_channels=3,
            n_heads=8,
            d_head=16,
            depth=5,
            dropout=0,
            context_dim=256,
            disable_self_attn=True,
            use_linear=True,
            use_checkpoint=False,
            cond_t_as_token=False,
            single_attn=False,
            use_pe=False,
            n_class=4,
            cat_params_to_x=True,
            cond_params_as_second_attn=False
        ),
        beta_1=1e-4,
        beta_T=.02,
        k=1.0,
        res=False,
        mode='linear',
        use_beta=True,
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
    anchor_loss_weight=1.,
    
    gen=True,
    fixed_id=[1, 0, 0, 0],
    ret_traj = False,
    ret_interval = 10,
    forward_sample=False,
    interpolate=False,
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
        eval_mode='gen',
        num_workers=4,
        class_choice='Chair',
    ),
    val=dict(
        type="ShapeNetSegPart",
        batch_size=32,
        split='test',
        root='/orion/group/shapenetcore_partanno_segmentation_benchmark_v0_normal',
        npoints=2048,
        shuffle=False,
        scale_mode='shape_unit',
        eval_mode='gen',
        num_workers=0,
        class_choice='Chair',
        save_only=True
    ),
)

optimizer = dict(type='Adam', lr=0.002, weight_decay=0.)

scheduler = dict(
    type='StepLR',
    step_size=2666,
    gamma=0.5,
)

logger = dict(
    type="RunLogger")

# when we the trained model from cshuan, image is rgb
resume_path="/orion/u/w4756677/diffusion/anchorDIff/work_dirs/anchordiff_exp251/checkpoints/ckpt_8000.pth"
save_num_batch = 1
max_epoch = 8000
eval_interval = 500
checkpoint_interval = 500
log_interval = 50
max_norm=10
# eval_both=True