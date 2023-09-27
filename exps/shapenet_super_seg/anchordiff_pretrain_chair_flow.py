# model settings
model = dict(
    type='AnchorDiffAE',
    pretrain_prior=True,
    encoder=dict(
        type='PartEncoderForTransformerDecoder',
        encoder=dict(
            type='PointNetVAEBase',
            zdim=256,
            point_dim=3,
        ),
        gen=True,
        per_part_encoder=True,
        n_class=4,
        kl_weight=1e-3,
        use_flow=True,
        latent_flow_depth=14, 
        latent_flow_hidden_dim=256,
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

data_path_train = ["/orion/u/w4756677/diffusion/anchorDIff/weights/pruned_back_s.pt", 
                   "/orion/u/w4756677/diffusion/anchorDIff/weights/pruned_seat_s.pt",
                   "/orion/u/w4756677/diffusion/anchorDIff/weights/pruned_leg_s.pt",
                   "/orion/u/w4756677/diffusion/anchorDIff/weights/arm.pt"]
train_bs = 128
val_bs = 32
scale_mode = 'shape_canonical_bbox'
data_path_val = ["/orion/u/w4756677/diffusion/anchorDIff/weights/test_back.pt", 
                   "/orion/u/w4756677/diffusion/anchorDIff/weights/test_seat.pt",
                   "/orion/u/w4756677/diffusion/anchorDIff/weights/test_leg.pt",
                   "/orion/u/w4756677/diffusion/anchorDIff/weights/test_arm.pt"]

optimizer = dict(type='Adam', lr=0.002, weight_decay=0.)

scheduler = dict(
    type='LinearLR',
    start_lr=2e-3,
    end_lr = 1e-4,
    start_epoch=80000,
    end_epoch=160000,
)

logger = dict(
    type="RunLogger")

# when we the trained model from cshuan, image is rgb
max_iter = 160000
eval_interval = 10000
checkpoint_interval = 10000
log_interval = 50
max_norm=10
n_class=4
npoints=2048
# eval_both=True