#!/usr/bin/env python3
"""Train the 7-64-64-2 residual and deploy it by retained validation support.

The published camera set is every camera that has at least one retained
validation pair after a model is trained. Pair-distance improvement is not
this trainer.

Do not launch src/pipeline/geometry_guided_residual_mlp.py. Its argparse
defaults are an earlier setup and are not this trainer.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from bootstrap import SRC_ROOT
from protocol import (
    BATCH,
    EPOCHS,
    LAMBDA_MAG,
    LAMBDA_MEAN,
    LAMBDA_PAIR,
    LAMBDA_TEMP,
    LR,
    MIN_TRAIN_PAIRS,
    R_MAX_M,
    SITES,
    TEMPORAL_REUSE,
    site_dir,
)

import sys
sys.path.insert(0, str(SRC_ROOT))


def mvc_loss(r_a, r_b, pred_a, pred_b, ka, kb):
    import torch

    eps = 1e-6
    floor_s = 0.05
    floor_r = 0.25
    confidence_a = torch.clamp(ka, 0.0, 1.0)
    confidence_b = torch.clamp(kb, 0.0, 1.0)
    teacher_conf = torch.maximum(confidence_a, confidence_b)
    teacher = (confidence_a.unsqueeze(1) * pred_a + confidence_b.unsqueeze(1) * pred_b) / (
        confidence_a + confidence_b + eps
    ).unsqueeze(1)
    teacher = teacher.detach()
    weight_a = teacher_conf * ((1.0 - confidence_a) + floor_s)
    weight_b = teacher_conf * ((1.0 - confidence_b) + floor_s)
    loss_a = torch.sum((pred_a - teacher) ** 2, dim=1)
    loss_b = torch.sum((pred_b - teacher) ** 2, dim=1)
    loss_mvc = torch.sum(weight_a * loss_a + weight_b * loss_b) / torch.clamp(
        torch.sum(weight_a + weight_b), min=eps,
    )
    reg_a = torch.sum(r_a ** 2, dim=1)
    reg_b = torch.sum(r_b ** 2, dim=1)
    loss_reg = 0.5 * (
        torch.sum(reg_a * (confidence_a + floor_r)) / torch.clamp(torch.sum(confidence_a + floor_r), min=eps)
        + torch.sum(reg_b * (confidence_b + floor_r)) / torch.clamp(torch.sum(confidence_b + floor_r), min=eps)
    )
    mean_vec = torch.mean(torch.cat([r_a, r_b], dim=0), dim=0)
    loss_mean = torch.sum(mean_vec ** 2)
    center = 0.5 * (r_a + r_b)
    center_w = torch.minimum(confidence_a, confidence_b)
    loss_center = torch.sum(torch.sum(center ** 2, dim=1) * center_w) / torch.clamp(
        torch.sum(center_w), min=eps,
    )
    total = loss_mvc + LAMBDA_MAG * loss_reg + LAMBDA_MEAN * loss_mean + LAMBDA_PAIR * loss_center
    return total


def support_cameras(cameras: list[str], val_ca, val_cb) -> list[str]:
    counts: Counter[str] = Counter()
    for index in val_ca.tolist():
        counts[cameras[int(index)]] += 1
    for index in val_cb.tolist():
        counts[cameras[int(index)]] += 1
    return [cam for cam in cameras if counts[cam] > 0]


def deploy_path(site: str, seed: int):
    name = f"{site}_F0.json" if seed == 0 else f"{site}_F0_seed{seed}.json"
    return site_dir(site) / "deploy" / name


def train_site(site: str, seed: int, anchor_mode: str) -> None:
    import torch
    from pipeline.geometry_guided_residual_mlp import GeometryGuidedResidualMLP

    meta = json.loads((site_dir(site) / "pairs" / f"{site}.json").read_text(encoding="utf-8"))
    if meta.get("anchor_mode") not in (None, anchor_mode):
        raise SystemExit(f"{site} pairs were frozen with {meta.get('anchor_mode')}, not {anchor_mode}")
    arrays = np.load(site_dir(site) / "pairs" / f"{site}.npz")
    cameras = meta["cameras"]
    count = int(arrays["train_fa"].shape[0])
    ckpt = site_dir(site) / "checkpoints" / f"{site}_F0_seed{seed}.pt"
    dest = deploy_path(site, seed)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if count < MIN_TRAIN_PAIRS:
        dest.write_text(json.dumps({
            "site": site, "seed": seed, "trained": False, "deploy": "geo", "S": [],
            "reason": "too_few_train_pairs", "rule": "retained_val_support",
            "uses_test_gt": False, "cameras": cameras,
        }, indent=2), encoding="utf-8")
        print(f"[train] {site} seed {seed}: no residual, {count} pairs")
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = GeometryGuidedResidualMLP(in_dim=7, hidden=64, max_residual_m=R_MAX_M)
    fa = torch.from_numpy(arrays["train_fa"])
    fb = torch.from_numpy(arrays["train_fb"])
    ka = torch.from_numpy(arrays["train_ka"].astype(np.float32))
    kb = torch.from_numpy(arrays["train_kb"].astype(np.float32))
    xa = torch.from_numpy(arrays["train_xa"].astype(np.float32))
    xb = torch.from_numpy(arrays["train_xb"].astype(np.float32))
    t0 = torch.from_numpy(arrays["temp_f0"])
    t1 = torch.from_numpy(arrays["temp_f1"])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in range(EPOCHS):
        perm = np.random.permutation(count)
        for start in range(0, count, BATCH):
            idx = torch.from_numpy(perm[start:start + BATCH].astype(np.int64))
            r_a = model(fa[idx])
            r_b = model(fb[idx])
            loss = mvc_loss(r_a, r_b, xa[idx] + r_a, xb[idx] + r_b, ka[idx], kb[idx])
            if t0.shape[0] and LAMBDA_TEMP > 0:
                take = min(TEMPORAL_REUSE, int(t0.shape[0]))
                loss = loss + LAMBDA_TEMP * torch.mean((model(t1[:take]) - model(t0[:take])) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    opened = support_cameras(cameras, arrays["val_ca"], arrays["val_cb"])
    torch.save({
        "state_dict": model.state_dict(),
        "cameras": cameras,
        "in_dim": 7,
        "cell": "F0",
        "use_b": False,
        "use_g": False,
        "max_residual_m": R_MAX_M,
        "anchor_mode": anchor_mode,
    }, ckpt)
    dest.write_text(json.dumps({
        "site": site,
        "seed": seed,
        "trained": True,
        "deploy": "model" if opened else "geo",
        "S": opened,
        "rule": "retained_val_support",
        "uses_test_gt": False,
        "cameras": cameras,
        "n_train_pairs": count,
        "n_val_pairs": int(arrays["val_fa"].shape[0]),
    }, indent=2), encoding="utf-8")
    print(f"[train] {site} seed {seed} S={opened}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, choices=sorted(SITES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--anchor-mode",
        default="multiplicative_gating_offsetted",
        choices=("multiplicative_gating_offsetted", "fixed_beta_050", "fixed_beta_070", "center", "bottom_center"),
    )
    args = parser.parse_args()
    train_site(args.site, args.seed, args.anchor_mode)


if __name__ == "__main__":
    main()
