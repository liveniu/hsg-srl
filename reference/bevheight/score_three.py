#!/usr/bin/env python3
"""Score the BEVHeight reference on M6, M3, and Town05.

Attachment is detection order: the first unused fusion box that contains the
projected ground point. Origin and ground height come from the geometry-database
import metadata. Localization is on those attached test observations. Tracking
uses the same fusion identity and the paper TrackEval 2.5 m readout.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ARTIFACT = Path(__file__).resolve().parents[2]
BH = Path(__file__).resolve().parent
sys.path.insert(0, str(ARTIFACT / "scripts"))
sys.path.insert(0, str(ARTIFACT / "src"))

from db_access import connect_ro, load_adapted, load_bias, load_mlp, load_test_keys  # noqa: E402
from protocol import DET_BATCH, GT_BATCH, MAX_T, SITES, geometry_db, match_report, origin_db  # noqa: E402
from evaluation.bev_hota import load_gt_bins, load_time_maps_from_match_report  # noqa: E402
from evaluation.hota_trackeval import hota_trackeval  # noqa: E402
from pipeline.geometry_guided_residual_mlp import compute_geometry_base, xywh_to_xyxy  # noqa: E402

def site_match_report(site: str):
    return match_report(site)

SITE_CAMERAS = {name: spec["cameras"] for name, spec in SITES.items()}
FINAL = ARTIFACT


def lidar_to_world(box: dict, T_l2w: np.ndarray) -> tuple[float, float]:
    xyz = np.array([box["x"], box["y"], box["z"] - 0.5 * box.get("h", 0.0), 1.0])
    w = T_l2w @ xyz
    return float(w[0]), float(w[1])


def project_kitti(T_w2c: np.ndarray, K: np.ndarray, xyz_w: np.ndarray):
    p = T_w2c @ np.append(xyz_w, 1.0)
    if p[2] <= 0.5:
        return None
    u = float(K[0, 0] * p[0] / p[2] + K[0, 2])
    v = float(K[1, 1] * p[1] / p[2] + K[1, 2])
    return u, v
SITES = ("lumpi_M6", "lumpi_M3", "synth_Town05_O_day")


def connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def import_frame(site: str) -> tuple[np.ndarray, float]:
    """Import origin and ground height stored with the ground-truth batch."""
    conn = connect_ro(origin_db(site))
    row = conn.execute(
        "SELECT metadata_json FROM trajectory_batches WHERE scene_id=? AND batch_id=?",
        (site, GT_BATCH),
    ).fetchone()
    conn.close()
    if row is None or not row[0]:
        raise RuntimeError(f"{site} has no {GT_BATCH} import metadata")
    meta = json.loads(row[0])
    raw = meta.get("origin_world", meta.get("origin_carla_world"))
    if raw is None or "z_ground" not in meta:
        raise RuntimeError(f"{site} metadata missing origin or z_ground")
    return np.asarray(raw, dtype=np.float64), float(meta["z_ground"])


def observations(site: str) -> list[dict]:
    test = load_test_keys(site)
    bias = load_bias(site)
    cameras = set(SITE_CAMERAS[site])
    max_t = MAX_T.get(site)
    conn = connect_ro(geometry_db(site))
    rows = []
    for cam, tid, fr, t, gid, bx, by_, bw, bh, wx, wy, gx, gy in conn.execute(
        """SELECT camera_name, local_track_id, frame_index, timestamp_sec, global_id,
                  bbox_x, bbox_y, bbox_w, bbox_h, world_x, world_y,
                  outside_reference_x, outside_reference_y
           FROM trajectory_observations
           WHERE scene_id=? AND batch_id=? AND world_x IS NOT NULL
             AND outside_reference_x IS NOT NULL""",
        (site, DET_BATCH),
    ):
        if cam not in cameras or (cam, int(tid), int(fr)) not in test:
            continue
        if max_t is not None and float(t) > max_t:
            continue
        if None in (bx, by_, bw, bh, gid, gx, gy):
            continue
        rows.append({
            "cam": cam, "fr": int(fr), "tid": int(tid), "gid": int(gid), "t": float(t),
            "xyxy": xywh_to_xyxy((float(bx), float(by_), float(bw), float(bh))),
            "geo": np.array([float(wx), float(wy)]),
            "gt": np.array([float(gx) + bias[0], float(gy) + bias[1]]),
        })
    conn.close()
    return rows


def attach(site: str, obs: list[dict]) -> tuple[list[dict], dict]:
    fmap = json.loads((BH / "step2_pack" / "work" / site / "frame_map.json").read_text(encoding="utf-8"))
    origin, z_ground = import_frame(site)
    by_frame: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for o in obs:
        by_frame[(o["cam"], o["fr"])].append(o)
    used: dict[tuple[str, int], set[int]] = defaultdict(set)
    hung = []
    n_box = n_token_miss = n_no_uv = n_no_pool = 0
    det_path = BH / "remote_gpu_4090" / "outputs" / site / "detections.jsonl"
    with det_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta = fmap.get(rec.get("sample_token"))
            if not meta:
                n_token_miss += 1
                continue
            cam, fr = meta["camera_id"], int(meta["frame_index"])
            pool = by_frame.get((cam, fr))
            if not pool:
                n_no_pool += 1
                continue
            T = np.asarray(meta["T_lidar_to_world"], dtype=np.float64)
            Tw = np.asarray(meta["T_world_to_kitti_cam"], dtype=np.float64)
            K = np.asarray(meta["K"], dtype=np.float64)
            z = z_ground
            for box in rec.get("boxes_lidar") or []:
                n_box += 1
                wx, wy = lidar_to_world(box, T)
                uv = project_kitti(Tw, K, np.array([wx, wy, z]))
                if uv is None:
                    n_no_uv += 1
                    continue
                hit = None
                for j, o in enumerate(pool):
                    if j in used[(cam, fr)]:
                        continue
                    x1, y1, x2, y2 = o["xyxy"]
                    if x1 <= uv[0] <= x2 and y1 <= uv[1] <= y2:
                        hit = j
                        break
                if hit is None:
                    continue
                used[(cam, fr)].add(hit)
                paper = np.array([wx - float(origin[0]), wy - float(origin[1])])
                hung.append({**pool[hit], "bh": paper, "score": float(box["score"])})
    info = {
        "origin_xy": [float(origin[0]), float(origin[1])],
        "z_ground": z_ground,
        "n_boxes": n_box,
        "n_token_miss": n_token_miss,
        "n_frames_without_test_obs": n_no_pool,
        "n_no_uv": n_no_uv,
    }
    return hung, info


def f0_errors(site: str, hung: list[dict]) -> np.ndarray:
    adapted, sizes = load_adapted(site)
    dep = json.loads((ARTIFACT / "data" / "work" / "sites" / site / "deploy" / f"{site}_F0.json").read_text(encoding="utf-8"))
    opened = set(dep.get("S") or [])
    mlp = load_mlp(site, 0) if dep.get("deploy") == "model" else None
    feats = []
    for o in hung:
        if mlp is None or o["cam"] not in opened or o["cam"] not in adapted:
            feats.append(None)
            continue
        H, scale = adapted[o["cam"]]
        _uv, _w, feat, _r = compute_geometry_base(o["xyxy"], H, scale, sizes[o["cam"]])
        feats.append(feat)
    residual = {}
    usable = [f for f in feats if f is not None]
    if usable and mlp is not None:
        import torch
        with torch.no_grad():
            pred = mlp(torch.from_numpy(np.stack(usable))).cpu().numpy()
        j = 0
        for i, f in enumerate(feats):
            if f is not None:
                residual[i] = pred[j]
                j += 1
    err = []
    for i, o in enumerate(hung):
        pos = o["geo"] if i not in residual else o["geo"] + residual[i]
        err.append(float(np.linalg.norm(pos - o["gt"])))
    return np.asarray(err)


def loc_line(name: str, err: np.ndarray) -> str:
    if len(err) == 0:
        return f"- {name}: n=0"
    return (
        f"- {name}: n={len(err)} med={float(np.median(err)):.3f} "
        f"rmse={float(np.sqrt(np.mean(err * err))):.3f} "
        f"p90={float(np.percentile(err, 90)):.3f} "
        f"frac>5m={float(np.mean(err > 5)):.3f} frac>10m={float(np.mean(err > 10)):.3f}"
    )


def hota_block(site: str, hung: list[dict]) -> list[str]:
    tm = load_time_maps_from_match_report(site_match_report(site))
    acc = defaultdict(list)
    t_min = float("inf")
    for o in hung:
        a, b = tm.get(o["cam"], (0.0, 0.0))
        lt = o["t"] + a + b * o["t"]
        t_min = min(t_min, lt)
        acc[(int(round(lt * 10)), o["gid"])].append(o["bh"])
    pred = {}
    for (bn, gid), pts in acc.items():
        arr = np.asarray(pts)
        pred.setdefault(bn, {})[gid] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    if t_min == float("inf"):
        t_min = 0.0
    gt = load_gt_bins(
        origin_db(site), site, load_bias(site), t_min, gt_batch=GT_BATCH,
        cameras=set(SITE_CAMERAS[site]), max_t=MAX_T.get(site),
    )
    rows = hota_trackeval(gt, pred)
    hotas = [r["hota"] for r in rows]
    bev = float(sum(hotas) / len(hotas))
    lines = [f"- BEVHeight TrackEval HOTA (hung tracks only), BEV={100*bev:.1f}"]
    for r in rows:
        lines.append(
            f"  - gate {r['gate']:.1f} m: HOTA={100*r['hota']:.1f} "
            f"DetA={100*r['deta']:.1f} AssA={100*r['assa']:.1f} LocA={100*r['loca']:.1f}"
        )
    return lines


def paper_hota() -> dict[tuple[str, str], str]:
    out = {}
    geo_path = ARTIFACT / "results" / "recorded" / "identity_hota25.csv"
    support_path = ARTIFACT / "results" / "recorded" / "identity_hota25_support.csv"
    grouped = defaultdict(list)
    with geo_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["site"] in SITES and row["method"] == "Geo":
                grouped[(row["site"], "Geo")].append(row)
    with support_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["site"] in SITES and row["method"] == "HSG-SRL" and row["seed"] == "0":
                grouped[(row["site"], "HSG-SRL")].append(row)
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda r: float(r["gate"]))
        bev = 100 * float(np.mean([float(r["hota"]) for r in rows]))
        bits = [f"{float(r['gate']):.1f}:{100*float(r['hota']):.1f}" for r in rows]
        out[key] = f"BEV={bev:.1f} ({', '.join(bits)})"
    return out


def main() -> None:
    paper = paper_hota()
    lines = [
        "# BEVHeight reference on M6, M3, and Town05",
        "",
        "Hang: detection order, first unused test box containing the projected ground point.",
        "Origin and z_ground are the import metadata stored with ground-truth batch l0hn_gt_001, subtracted once.",
        "Localization errors are on attached test observations only. HSG and HSG-SRL seed 0 on that subset",
        "are the comparison in supplementary Table S12.",
        "HSG, HSG-SRL, test keys, bias, and GT are read from data/work/sites.",
        "The residual camera set is the retained-validation-pair deploy file written by training.",
        "BEVHeight tracking HOTA is written by score_hota_remerge.py after the paper fusion.",
        "",
    ]
    for site in SITES:
        obs = observations(site)
        hung, info = attach(site, obs)
        bh = np.array([float(np.linalg.norm(o["bh"] - o["gt"])) for o in hung])
        geo = np.array([float(np.linalg.norm(o["geo"] - o["gt"])) for o in hung])
        f0 = f0_errors(site, hung)
        delta = np.stack([o["bh"] - o["gt"] for o in hung]) if hung else np.zeros((0, 2))
        lines.append(f"## {site}")
        lines.append(f"- origin_xy: {info['origin_xy']}, z_ground: {info['z_ground']}")
        lines.append(
            f"- boxes={info['n_boxes']} token_miss={info['n_token_miss']} "
            f"frames_without_test_obs={info['n_frames_without_test_obs']} no_uv={info['n_no_uv']}"
        )
        lines.append(f"- test observations: {len(obs)}")
        lines.append(
            f"- hung: {len(hung)} ({100*len(hung)/max(len(obs),1):.1f}%)"
        )
        lines.extend([
            loc_line("HSG on attached subset", geo),
            loc_line("HSG-SRL seed 0 on attached subset", f0),
            loc_line("BEVHeight on attached subset", bh),
        ])
        if len(delta):
            lines.append(
                f"- BEVHeight minus GT, median dx={float(np.median(delta[:,0])):.3f} "
                f"dy={float(np.median(delta[:,1])):.3f}"
            )
        lines.append(f"- full-set HOTA HSG: {paper.get((site,'Geo'), 'missing')}")
        lines.append(f"- full-set HOTA HSG-SRL seed 0: {paper.get((site,'HSG-SRL'), 'missing')}")
        lines.append("")
        print("\n".join(lines[-12:]), flush=True)
    dest = BH / "outputs" / "score_three.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    print(f"[score] -> {dest}", flush=True)


if __name__ == "__main__":
    main()
