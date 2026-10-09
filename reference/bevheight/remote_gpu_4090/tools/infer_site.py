#!/usr/bin/env python3
"""Dump BEVHeight boxes on the packed test split. No official Rope3D evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BEV_ROOT = Path(os.environ.get("BEVHEIGHT_ROOT", HERE / "vendor" / "BEVHeight-main"))
sys.path.insert(0, str(BEV_ROOT))
sys.path.insert(0, str(HERE / "tools"))

import torch  # noqa: E402

from hw_guard import apply_4090_runtime  # noqa: E402
from official_hparams import SCORE_THRESHOLD  # noqa: E402
from train_site import SiteModel, _load_hw, load_pretrained, resolve_ckpt  # noqa: E402


def _boxes_to_list(result) -> list[dict]:
    """Official get_bboxes item: [boxes_tensor(N,9), scores, labels]."""
    boxes, scores, labels = result[0], result[1], result[2]
    if hasattr(boxes, "tensor"):
        boxes = boxes.tensor
    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    out = []
    for i in range(len(scores)):
        if float(scores[i]) < SCORE_THRESHOLD:
            continue
        b = boxes[i]
        out.append({
            "x": float(b[0]), "y": float(b[1]), "z": float(b[2]),
            "w": float(b[3]), "l": float(b[4]), "h": float(b[5]),
            "yaw": float(b[6]) if b.shape[0] > 6 else 0.0,
            "score": float(scores[i]),
            "label": int(labels[i]),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="Default: outputs/<site>/checkpoints/best.ckpt, else last.ckpt")
    ap.add_argument("--data-root", default=str(HERE / "data"))
    ap.add_argument("--out-root", default=str(HERE / "outputs"))
    ap.add_argument("--split", default="test", choices=("test", "val"))
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()
    apply_4090_runtime(strict=True)
    site_root = Path(args.data_root) / args.site
    H, W = _load_hw(site_root)
    ckpt = resolve_ckpt(Path(args.out_root) / args.site, args.ckpt)
    print(f"infer ckpt -> {ckpt}")
    model = SiteModel(str(site_root.resolve()), args.gpus, args.batch_size, H, W)
    load_pretrained(model.model, ckpt)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        model.model.cuda()
    loader = model._loader(args.split, False)
    infos = loader.dataset.infos
    out_path = Path(args.out_root) / args.site / "detections.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    cursor = 0
    with out_path.open("w", encoding="utf-8") as f:
        with torch.no_grad():
            for batch in loader:
                sweep_imgs, mats, _, img_metas = batch[:4]
                if torch.cuda.is_available():
                    sweep_imgs = sweep_imgs.cuda()
                    mats = {k: v.cuda() for k, v in mats.items()}
                preds = model.model(sweep_imgs, mats)
                results = model.model.get_bboxes(preds, img_metas)
                for i, res in enumerate(results):
                    token = infos[cursor]["sample_token"] if cursor < len(infos) else None
                    cursor += 1
                    rec = {
                        "site": args.site,
                        "sample_token": token,
                        "boxes_lidar": _boxes_to_list(res),
                    }
                    f.write(json.dumps(rec) + "\n")
                    n += 1
    print(f"wrote {n} samples -> {out_path}")


if __name__ == "__main__":
    main()
