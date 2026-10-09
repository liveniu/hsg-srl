"""Frozen official Rope3D R50_140 hyperparameters.

Source: experiments/bevheight/BEVHeight-main/exps/rope3d/
        bev_height_lss_r50_864_1536_128x128_140.py
Do not change bounds or final_dim. Batch size may drop on 24 GB cards;
lr still follows the official formula (2e-4/64)*batch*gpus.
"""

FINAL_DIM = (864, 1536)
DEFAULT_HW = (1080, 1920)  # H, W
IMG_CONF = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)

X_BOUND = [0, 140.8, 0.8]
Y_BOUND = [-70.4, 70.4, 0.8]
Z_BOUND = [-5, 3, 8]
D_BOUND = [-1.5, 3.0, 180]

BASIC_LR_PER_IMG = 2e-4 / 64
WEIGHT_DECAY = 1e-7
MILESTONES = [19, 23]
MAX_EPOCHS = 20
GRAD_CLIP = 5
SCORE_THRESHOLD = 0.1
NMS_THR = 0.2
MAX_EPOCHS_OFFICIAL = 20

WEIGHT_URL = "https://cloud.tsinghua.edu.cn/f/343be049d5e74d14a5af/?dl=1"
CONFIG_NAME = "bev_height_lss_r50_864_1536_128x128_140"


def backbone_conf(final_dim=FINAL_DIM):
    return {
        "x_bound": list(X_BOUND),
        "y_bound": list(Y_BOUND),
        "z_bound": list(Z_BOUND),
        "d_bound": list(D_BOUND),
        "final_dim": final_dim,
        "output_channels": 80,
        "downsample_factor": 16,
        "img_backbone_conf": dict(
            type="ResNet",
            depth=50,
            frozen_stages=0,
            out_indices=[0, 1, 2, 3],
            norm_eval=False,
            init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
        ),
        "img_neck_conf": dict(
            type="SECONDFPN",
            in_channels=[256, 512, 1024, 2048],
            upsample_strides=[0.25, 0.5, 1, 2],
            out_channels=[128, 128, 128, 128],
        ),
        "height_net_conf": dict(in_channels=512, mid_channels=512),
    }


def ida_aug_conf(H, W, final_dim=FINAL_DIM):
    return {
        "final_dim": final_dim,
        "H": int(H),
        "W": int(W),
        "bot_pct_lim": (0.0, 0.0),
        "cams": ["CAM_FRONT"],
        "Ncams": 1,
    }
