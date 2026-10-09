#!/usr/bin/env python3
"""Fine-tune official R50_140 on one packed site. 4090 / sm_89 only."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BEV_ROOT = Path(os.environ.get("BEVHEIGHT_ROOT", HERE / "vendor" / "BEVHeight-main"))
sys.path.insert(0, str(BEV_ROOT))
sys.path.insert(0, str(HERE / "tools"))

import torch  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
from pytorch_lightning.callbacks import ModelCheckpoint  # noqa: E402
from torch.optim.lr_scheduler import MultiStepLR  # noqa: E402

from hw_guard import apply_4090_runtime  # noqa: E402

from official_hparams import (  # noqa: E402
    BASIC_LR_PER_IMG,
    CONFIG_NAME,
    FINAL_DIM,
    GRAD_CLIP,
    IMG_CONF,
    MAX_EPOCHS,
    MILESTONES,
    WEIGHT_DECAY,
    backbone_conf,
    ida_aug_conf,
)

# Official class / head tables (car-only training still keeps the official head).
CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
TASKS = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]


def _head_conf():
    bev_backbone = dict(
        type="ResNet", in_channels=80, depth=18, num_stages=3,
        strides=(1, 2, 2), dilations=(1, 1, 1), out_indices=[0, 1, 2],
        norm_eval=False, base_channels=160,
    )
    bev_neck = dict(
        type="SECONDFPN",
        in_channels=[80, 160, 320, 640],
        upsample_strides=[1, 2, 4, 8],
        out_channels=[64, 64, 64, 64],
    )
    bbox_coder = dict(
        type="CenterPointBBoxCoder",
        post_center_range=[0.0, -70.4, -10.0, 140.8, 70.4, 10.0],
        max_num=500, score_threshold=0.1, out_size_factor=4,
        voxel_size=[0.2, 0.2, 8],
        pc_range=[0, -70.4, -5, 140.8, 70.4, 3],
        code_size=9,
    )
    train_cfg = dict(
        point_cloud_range=[0, -70.4, -5, 140.8, 70.4, 3],
        grid_size=[704, 704, 1], voxel_size=[0.2, 0.2, 8],
        out_size_factor=4, dense_reg=1, gaussian_overlap=0.1,
        max_objs=500, min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    )
    test_cfg = dict(
        post_center_limit_range=[0.0, -70.4, -10.0, 140.8, 70.4, 10.0],
        max_per_img=500, max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        score_threshold=0.1, out_size_factor=4,
        voxel_size=[0.2, 0.2, 8], nms_type="circle",
        pre_max_size=1000, post_max_size=83, nms_thr=0.2,
    )
    return {
        "bev_backbone_conf": bev_backbone,
        "bev_neck_conf": bev_neck,
        "tasks": TASKS,
        "common_heads": dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        "bbox_coder": bbox_coder,
        "train_cfg": train_cfg,
        "test_cfg": test_cfg,
        "in_channels": 256,
        "loss_cls": dict(type="GaussianFocalLoss", reduction="mean"),
        "loss_bbox": dict(type="L1Loss", reduction="mean", loss_weight=0.25),
        "gaussian_overlap": 0.1,
        "min_radius": 2,
    }


def _load_hw(site_root: Path) -> tuple[int, int]:
    fmap = json.loads((site_root / "frame_map.json").read_text(encoding="utf-8"))
    if not fmap:
        return 1080, 1920
    wh = next(iter(fmap.values())).get("image_wh") or [1920, 1080]
    return int(wh[1]), int(wh[0])


def _make_labeled_no_aug(base_cls):
    """Official val split drops GT when is_train=False; train mode adds pose jitter.

    Val loss for early-stop needs GT and no 50% intrinsic/extrinsic jitter.
    """

    class LabeledNoAugDataset(base_cls):
        def get_image(self, cam_infos, cams):
            was = self.is_train
            self.is_train = False
            try:
                return super().get_image(cam_infos, cams)
            finally:
                self.is_train = was

    return LabeledNoAugDataset


def resolve_ckpt(out_dir: Path, explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(f"missing ckpt {p}")
        return p
    best = out_dir / "checkpoints" / "best.ckpt"
    last = out_dir / "checkpoints" / "last.ckpt"
    if best.is_file():
        return best
    if last.is_file():
        return last
    raise SystemExit(f"no best.ckpt or last.ckpt under {out_dir / 'checkpoints'}")


class ConvergeStop(pl.Callback):
    """Stop when val loss has no relative drop for ``patience`` epochs, cap at max_epochs."""

    def __init__(self, monitor="val_detection_loss", patience=3, rel_delta=0.01,
                 min_epochs=3, out_path=None):
        self.monitor = monitor
        self.patience = int(patience)
        self.rel_delta = float(rel_delta)
        self.min_epochs = int(min_epochs)
        self.out_path = Path(out_path) if out_path else None
        self.best = None
        self.best_epoch = None
        self.wait = 0
        self.history = []
        self.reason = "max_epochs"

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        raw = trainer.callback_metrics.get(self.monitor)
        if raw is None:
            return
        val = float(raw.detach().cpu())
        epoch_1based = int(trainer.current_epoch) + 1
        train_raw = trainer.callback_metrics.get("detection_loss")
        train_loss = float(train_raw.detach().cpu()) if train_raw is not None else None
        self.history.append({
            "epoch": epoch_1based,
            self.monitor: val,
            "detection_loss": train_loss,
        })
        improved = self.best is None or val < self.best * (1.0 - self.rel_delta)
        if improved:
            self.best = val
            self.best_epoch = epoch_1based
            self.wait = 0
            print(f"  val improved: {self.monitor}={val:.6f} @ epoch {epoch_1based}",
                  flush=True)
        else:
            self.wait += 1
            print(
                f"  val plateau {self.wait}/{self.patience}: {self.monitor}={val:.6f} "
                f"(best={self.best:.6f} @ {self.best_epoch})",
                flush=True,
            )
        if epoch_1based >= self.min_epochs and self.wait >= self.patience:
            self.reason = (
                f"plateau: no {self.rel_delta:.2%} drop in {self.patience} val epochs "
                f"(best={self.best:.6f} @ epoch {self.best_epoch})"
            )
            trainer.should_stop = True

    def on_fit_end(self, trainer, pl_module):
        payload = {
            "reason": self.reason,
            "stopped_epoch": int(trainer.current_epoch) + 1,
            "max_epochs": int(trainer.max_epochs),
            "best_epoch": self.best_epoch,
            "best_val_detection_loss": self.best,
            "patience": self.patience,
            "rel_delta": self.rel_delta,
            "min_epochs": self.min_epochs,
            "history": self.history,
        }
        if self.out_path:
            self.out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            csv_path = self.out_path.with_name("epoch_metrics.csv")
            lines = ["epoch,detection_loss,val_detection_loss"]
            for row in self.history:
                lines.append(
                    f"{row['epoch']},{row.get('detection_loss')},{row.get(self.monitor)}"
                )
            csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_pretrained(model, ckpt_path: Path) -> None:
    raw = torch.load(str(ckpt_path), map_location="cpu")
    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    cleaned = {}
    for k, v in state.items():
        if k.startswith("model."):
            cleaned[k[len("model."):]] = v
        else:
            cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"loaded {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")


class SiteModel(pl.LightningModule):
    def __init__(self, site_root: str, gpus: int, batch_size: int, H: int, W: int):
        super().__init__()
        from models.bev_height import BEVHeight
        from dataset.nusc_mv_det_dataset import NuscMVDetDataset, collate_fn
        self.site_root = site_root
        self.gpus = gpus
        self.batch_size = batch_size
        self.H, self.W = H, W
        self.collate_fn = collate_fn
        self.NuscMVDetDataset = NuscMVDetDataset
        self.model = BEVHeight(backbone_conf(FINAL_DIM), _head_conf())
        self.save_hyperparameters(ignore=["collate_fn", "NuscMVDetDataset"])

    def forward(self, sweep_imgs, mats):
        return self.model(sweep_imgs, mats)

    def training_step(self, batch, batch_idx):
        sweep_imgs, mats, _, _, gt_boxes, gt_labels = batch[:6]
        if torch.cuda.is_available():
            sweep_imgs = sweep_imgs.cuda()
            mats = {k: v.cuda() for k, v in mats.items()}
            gt_boxes = [g.cuda() for g in gt_boxes]
            gt_labels = [g.cuda() for g in gt_labels]
        preds = self(sweep_imgs, mats)
        targets = self.model.get_targets(gt_boxes, gt_labels)
        loss = self.model.loss(targets, preds)
        self.log("detection_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sweep_imgs, mats, _, _, gt_boxes, gt_labels = batch[:6]
        if torch.cuda.is_available():
            sweep_imgs = sweep_imgs.cuda()
            mats = {k: v.cuda() for k, v in mats.items()}
            gt_boxes = [g.cuda() for g in gt_boxes]
            gt_labels = [g.cuda() for g in gt_labels]
        preds = self(sweep_imgs, mats)
        targets = self.model.get_targets(gt_boxes, gt_labels)
        loss = self.model.loss(targets, preds)
        self.log("val_detection_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = BASIC_LR_PER_IMG * self.batch_size * max(self.gpus, 1)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        sch = MultiStepLR(opt, MILESTONES)
        return [[opt], [sch]]

    def _loader(self, split: str, shuffle: bool, labeled_no_aug: bool = False):
        from functools import partial
        info = os.path.join(self.site_root, f"infos_{split}.pkl")
        cls = self.NuscMVDetDataset
        is_train = shuffle
        if labeled_no_aug:
            if not hasattr(self, "_ValDS"):
                self._ValDS = _make_labeled_no_aug(self.NuscMVDetDataset)
            cls = self._ValDS
            is_train = True
        ds = cls(
            ida_aug_conf=ida_aug_conf(self.H, self.W),
            classes=CLASSES,
            data_root=self.site_root,
            info_path=info,
            is_train=is_train,
            img_conf=IMG_CONF,
            num_sweeps=1,
            return_depth=False,
        )
        return torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, num_workers=4,
            drop_last=shuffle, shuffle=shuffle,
            collate_fn=partial(self.collate_fn, is_return_depth=False)
            if shuffle else self.collate_fn,
        )

    def train_dataloader(self):
        return self._loader("train", True)

    def val_dataloader(self):
        return self._loader("val", False, labeled_no_aug=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--data-root", default=str(HERE / "data"))
    ap.add_argument("--out-root", default=str(HERE / "outputs"))
    ap.add_argument("--ckpt-in", default=str(HERE / "ckpts" / "R50_140.pth"))
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS,
                    help="Hard cap (official Rope3D is 20). Early-stop may finish sooner.")
    ap.add_argument("--patience", type=int, default=3,
                    help="Stop after this many val epochs without a relative drop.")
    ap.add_argument("--rel-delta", type=float, default=0.01,
                    help="Relative val-loss drop that counts as improvement (0.01 = 1%).")
    ap.add_argument("--min-epochs", type=int, default=3,
                    help="Do not early-stop before this many epochs.")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="Run the full --max-epochs (official 20-epoch schedule).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    site_root = str((Path(args.data_root) / args.site).resolve())
    out_dir = Path(args.out_root) / args.site
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = _load_hw(Path(site_root))
    hw = apply_4090_runtime(strict=True)
    pl.seed_everything(args.seed)
    model = SiteModel(site_root, args.gpus, args.batch_size, H, W)
    if Path(args.ckpt_in).is_file():
        load_pretrained(model.model, Path(args.ckpt_in))
    else:
        raise SystemExit(f"missing official weight {args.ckpt_in}")
    use_early = (not args.no_early_stop) and args.max_epochs > 1
    hparams = {
        "site": args.site,
        "config": CONFIG_NAME,
        "batch_size": args.batch_size,
        "gpus": args.gpus,
        "max_epochs": args.max_epochs,
        "early_stop": use_early,
        "patience": args.patience,
        "rel_delta": args.rel_delta,
        "min_epochs": args.min_epochs,
        "lr": BASIC_LR_PER_IMG * args.batch_size * max(args.gpus, 1),
        "H": H, "W": W,
        "ckpt_in": args.ckpt_in,
        "flavor": "4090_sm89",
        "hw": hw,
        "stack": "torch1.13.1+cu117 / mmcv-full1.7.1 / mmdet2.28.2 / mmdet3d1.0.0rc5",
        "note": "4090 fine-tune R50_140; TF32 off; early-stop on val detection loss",
    }
    (out_dir / "hparams.json").write_text(json.dumps(hparams, indent=2), encoding="utf-8")
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best",
        monitor="val_detection_loss" if use_early else None,
        mode="min",
        save_top_k=1 if use_early else 0,
        save_last=True,
    )
    callbacks = [ckpt_cb]
    stopper = None
    if use_early:
        stopper = ConvergeStop(
            patience=args.patience,
            rel_delta=args.rel_delta,
            min_epochs=min(args.min_epochs, args.max_epochs),
            out_path=out_dir / "early_stop.json",
        )
        callbacks.append(stopper)
    trainer = pl.Trainer(
        gpus=args.gpus, max_epochs=args.max_epochs,
        gradient_clip_val=GRAD_CLIP, callbacks=callbacks,
        default_root_dir=str(out_dir), precision=32,
        num_sanity_val_steps=0,
        limit_val_batches=1.0 if use_early else 0,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model)
    best_src = ckpt_cb.best_model_path
    best_dst = out_dir / "checkpoints" / "best.ckpt"
    if best_src and Path(best_src).is_file():
        if Path(best_src).resolve() != best_dst.resolve():
            shutil.copy2(best_src, best_dst)
        print(f"best ckpt -> {best_dst} (from {best_src})")
    last_ckpt = out_dir / "checkpoints" / "last.ckpt"
    print(f"last ckpt  -> {last_ckpt}")
    if stopper is not None:
        print(f"early stop -> {out_dir / 'early_stop.json'} ({stopper.reason})")
        hparams["stopped_epoch"] = int(trainer.current_epoch) + 1
        hparams["best_epoch"] = stopper.best_epoch
        hparams["best_val_detection_loss"] = stopper.best
        hparams["stop_reason"] = stopper.reason
        (out_dir / "hparams.json").write_text(json.dumps(hparams, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
