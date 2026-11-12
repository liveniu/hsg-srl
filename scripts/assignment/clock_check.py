#!/usr/bin/env python3
"""Clock-source check for M2, M6, and Ko-PER.

Two estimation coordinates (fixed midpoint, HSG) and two clock models
(constant dt0, that same dt0 plus the existing two-round affine fit).
Clocks are fit only before the test interval. Frozen seed-0 HSG-SRL is
scored and is not used to estimate a clock. Databases are opened read-only.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np

from common import (
    ARTIFACT_ROOT,
    BOOT_DRAWS,
    BOOT_SEED,
    DET_BATCH,
    DT_GATE_M,
    DT_MAX,
    DT_MIN,
    DT_SAMPLE_TRACKS,
    DT_STEP,
    GATE_M,
    GT_BATCH,
    GT_CLASSES,
    MARGIN_M,
    MAX_GAP_S,
    MAX_T,
    MIN_AFFINE_SAMPLES,
    MIN_OVERLAP,
    SANITY_M,
    SITES,
    SITES_CLOCK,
    UPSTREAM,
    connect_ro,
    ensure,
    geometry_db,
    label_of,
    load_test_keys,
    match_report,
    median,
    origin_db,
    rmse,
    test_start,
    write_json,
)
from db_access import camera_polygons, load_adapted, load_mlp
from geometry.overlap_gate import inside_mask
from pipeline.geometry_guided_residual_mlp import compute_geometry_base, xywh_to_xyxy

ARMS = ("midpoint_const", "midpoint_affine", "hsg_const", "hsg_affine")


def load_matcher():
    spec = importlib.util.spec_from_file_location(
        "fab5c_matcher", UPSTREAM / "eval_fab2_det_gt_match.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATCHER = load_matcher()
TimeMap = MATCHER.TimeMap
GtTable = MATCHER.GtTable


def load_gt(conn, site: str, camera: str, t_max: float | None):
    sql = (
        "SELECT timestamp_sec, global_id, outside_reference_x, outside_reference_y,"
        "       outside_reference_image_x, outside_reference_image_y, class_id"
        " FROM trajectory_observations"
        " WHERE scene_id=? AND batch_id=? AND camera_name=?"
        "   AND outside_reference_x IS NOT NULL AND outside_reference_y IS NOT NULL"
    )
    rows = []
    for record in conn.execute(sql, (site, GT_BATCH, camera)):
        stamp = float(record["timestamp_sec"])
        if t_max is not None and stamp >= t_max:
            continue
        class_id = record["class_id"]
        if class_id is not None and int(class_id) not in GT_CLASSES:
            continue
        if record["global_id"] is None:
            continue
        rows.append((
            stamp, int(record["global_id"]),
            float(record["outside_reference_x"]), float(record["outside_reference_y"]),
            None if record["outside_reference_image_x"] is None else float(record["outside_reference_image_x"]),
            None if record["outside_reference_image_y"] is None else float(record["outside_reference_image_y"]),
        ))
    return GtTable(rows)


def slice_tracks(tracks: dict, t_max: float) -> dict:
    out = {}
    for tid, track in tracks.items():
        keep = track["t"] < t_max
        if np.any(keep):
            out[tid] = {key: value[keep] for key, value in track.items()}
    return out


def estimate_dt0(tracks: dict, gt) -> dict:
    returned, curve = MATCHER._estimate_dt(
        tracks, gt,
        dt_min=DT_MIN, dt_max=DT_MAX, dt_step=DT_STEP, max_gap=MAX_GAP_S,
        gate=DT_GATE_M, min_overlap=MIN_OVERLAP, sample_tracks=DT_SAMPLE_TRACKS,
    )
    maximum = max((item["matched_tracks"] for item in curve), default=0)
    eligible = [
        item for item in curve
        if maximum and item["matched_tracks"] >= 0.8 * maximum and item["median_residual_m"] is not None
    ]
    if not eligible:
        return {
            "valid": False, "status": "no_eligible_offset", "dt0": None,
            "max_matched_tracks": int(maximum), "ignored_default_return": float(returned),
        }
    chosen = min(eligible, key=lambda item: item["median_residual_m"])
    return {
        "valid": True, "status": "estimated", "dt0": float(chosen["dt_sec"]),
        "matched_tracks_at_dt0": int(chosen["matched_tracks"]),
        "median_residual_at_dt0": chosen["median_residual_m"],
        "max_matched_tracks": int(maximum),
    }


def estimate_affine(dt0: float, tracks: dict, gt) -> dict:
    clock = TimeMap(dt0)
    assigned = MATCHER._assign_tracks(
        tracks, gt, clock, max_gap=MAX_GAP_S, gate=GATE_M,
        min_overlap=MIN_OVERLAP, margin_m=MARGIN_M,
    )
    if not assigned:
        return {"valid": False, "status": "no_pretest_assignment", "a": None, "b": None, "n_fit": 0}
    n_fit = 0
    for round_index in range(2):
        step_a, step_b, n_fit = MATCHER._fit_time_drift(tracks, gt, assigned, clock, max_gap=MAX_GAP_S)
        if n_fit < MIN_AFFINE_SAMPLES:
            return {
                "valid": False, "status": "insufficient_samples", "round": round_index + 1,
                "n_fit": int(n_fit), "ignored_a": float(step_a), "ignored_b": float(step_b),
                "a": None, "b": None,
            }
        clock = TimeMap(dt0, clock.a + step_a, clock.b + step_b)
        assigned = MATCHER._assign_tracks(
            tracks, gt, clock, max_gap=MAX_GAP_S, gate=GATE_M,
            min_overlap=MIN_OVERLAP, margin_m=MARGIN_M,
        )
        if not assigned:
            return {
                "valid": False, "status": "assignment_lost_after_affine",
                "round": round_index + 1, "n_fit": int(n_fit), "a": None, "b": None,
            }
    return {
        "valid": True, "status": "estimated", "a": float(clock.a), "b": float(clock.b),
        "n_fit": int(n_fit), "a_ms": round(1000.0 * clock.a, 1),
        "b_ms_per_min": round(60_000.0 * clock.b, 2),
    }


def kept_frames(tracks: dict, gt, assigned: dict, clock) -> dict:
    out = {}
    for tid, (gid, _median, _count) in assigned.items():
        track = tracks[tid]
        times = track["t"] + clock.offsets(track["t"])
        gx, gy, ok, _iu, _iv = gt.at(gid, times, MAX_GAP_S)
        residual = np.hypot(track["x"] - gx, track["y"] - gy)
        for index in np.flatnonzero(ok & (residual <= SANITY_M)):
            out[(int(tid), int(track["frame"][index]))] = (int(gid), float(track["t"][index]))
    return out


def assign_test(tracks: dict, gt, clock, camera: str, test: set) -> dict:
    assigned = MATCHER._assign_tracks(
        tracks, gt, clock, max_gap=MAX_GAP_S, gate=GATE_M,
        min_overlap=MIN_OVERLAP, margin_m=MARGIN_M,
    )
    frames = kept_frames(tracks, gt, assigned, clock)
    return {
        key: value for key, value in frames.items()
        if (camera, key[0], key[1]) in test
    }


def project_midpoint(box_xywh: np.ndarray, homography: np.ndarray, scale: float) -> np.ndarray:
    box = np.asarray(box_xywh, dtype=np.float64)
    uv = np.column_stack((
        box[:, 0] + box[:, 2] / 2.0, box[:, 1] + 0.75 * box[:, 3], np.ones(len(box)),
    ))
    metric = np.diag([1.0 / scale, 1.0 / scale, 1.0]) @ homography
    projected = uv @ metric.T
    return projected[:, :2] / projected[:, 2:3]


def load_camera(conn, site: str, camera: str, adapted):
    limit = MAX_T.get(site)
    rows = conn.execute(
        """SELECT local_track_id, frame_index, timestamp_sec, bbox_x, bbox_y, bbox_w, bbox_h,
                  world_x, world_y, outside_reference_x
           FROM trajectory_observations
           WHERE scene_id=? AND batch_id=? AND camera_name=?
             AND world_x IS NOT NULL AND world_y IS NOT NULL
           ORDER BY local_track_id, timestamp_sec""",
        (site, DET_BATCH, camera),
    ).fetchall()
    homography, scale = adapted[camera]
    grouped: dict[int, list] = {}
    published = set()
    boxes = {}
    for row in rows:
        if limit is not None and float(row["timestamp_sec"]) > limit:
            continue
        tid = int(row["local_track_id"])
        frame = int(row["frame_index"])
        grouped.setdefault(tid, []).append(row)
        boxes[(tid, frame)] = (float(row["bbox_x"]), float(row["bbox_y"]), float(row["bbox_w"]), float(row["bbox_h"]))
        if row["outside_reference_x"] is not None:
            published.add((tid, frame))
    hsg, mid = {}, {}
    for tid, items in grouped.items():
        stamp = np.asarray([float(item["timestamp_sec"]) for item in items])
        frame = np.asarray([int(item["frame_index"]) for item in items])
        hsg_xy = np.asarray([[float(item["world_x"]), float(item["world_y"])] for item in items])
        box = np.asarray([boxes[(tid, int(item["frame_index"]))] for item in items])
        mid_xy = project_midpoint(box, homography, scale)
        hsg[tid] = {"t": stamp, "frame": frame, "x": hsg_xy[:, 0], "y": hsg_xy[:, 1]}
        mid[tid] = {"t": stamp, "frame": frame, "x": mid_xy[:, 0], "y": mid_xy[:, 1]}
    return hsg, mid, boxes, published


def xy_at(tracks: dict, tid: int, frame: int) -> tuple[float, float]:
    track = tracks[tid]
    index = int(np.flatnonzero(track["frame"] == frame)[0])
    return float(track["x"][index]), float(track["y"][index])


def stored_clock(info: dict):
    return TimeMap(
        float(info["dt_applied_sec"]),
        float(info.get("drift_const_ms", 0.0)) / 1000.0,
        float(info.get("drift_slope_ms_per_min", 0.0)) / 60_000.0,
    )


def jaccard(left: set, right: set) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def arm_from_source(source: str, tracks: dict, gt_fit, gt_full, camera: str, test: set, cut: float) -> dict:
    pre = slice_tracks(tracks, cut)
    dt0 = estimate_dt0(pre, gt_fit)
    out = {"source": source, "dt0": dt0, "const": None, "affine_fit": None, "affine": None}
    if not dt0["valid"]:
        return out
    const = TimeMap(dt0["dt0"])
    const_frames = assign_test(tracks, gt_full, const, camera, test)
    out["const"] = {"valid": True, "clock": const, "frames": const_frames}
    affine = estimate_affine(dt0["dt0"], pre, gt_fit)
    out["affine_fit"] = {key: value for key, value in affine.items() if key != "clock"}
    if not affine["valid"]:
        out["affine"] = {"valid": False, "status": affine["status"]}
        return out
    clock = TimeMap(dt0["dt0"], affine["a"], affine["b"])
    out["affine"] = {"valid": True, "clock": clock, "frames": assign_test(tracks, gt_full, clock, camera, test)}
    return out


def srl_map(site, camera, keys, boxes, hsg, adapted, sizes, polygons, deploy) -> dict:
    if not keys:
        return {}
    homography, scale = adapted[camera]
    features = np.stack([
        compute_geometry_base(xywh_to_xyxy(boxes[key]), homography, scale, sizes[camera])[2]
        for key in keys
    ]).astype(np.float32)
    base = np.asarray([xy_at(hsg, tid, frame) for tid, frame in keys])
    correction = np.zeros_like(base)
    if deploy.get("deploy") == "model" and camera in set(deploy.get("S") or []):
        model = load_mlp(site, 0, ARTIFACT_ROOT / "results" / "recorded" / "checkpoints")
        if model is None:
            raise RuntimeError(f"{site} seed-0 checkpoint is missing")
        import torch
        with torch.no_grad():
            correction = model(torch.from_numpy(features)).cpu().numpy().astype(np.float64)
        correction[~inside_mask(base, polygons.get(camera, []))] = 0.0
    return {key: (float(p[0] + d[0]), float(p[1] + d[1])) for key, p, d in zip(keys, base, correction)}


def score(clock, rows, hsg, mid, srl, gt):
    e_mid, e_hsg, e_srl, tids, row_ids = [], [], [], [], []
    skipped = 0
    for camera, tid, frame, gid, stamp in rows:
        query = np.asarray([stamp]) + clock.offsets(np.asarray([stamp]))
        gx, gy, ok, _iu, _iv = gt[camera].at(gid, query, MAX_GAP_S)
        if not bool(ok[0]):
            skipped += 1
            continue
        mx, my = xy_at(mid[camera], tid, frame)
        hx, hy = xy_at(hsg[camera], tid, frame)
        sx, sy = srl[camera][(tid, frame)]
        e_mid.append(float(np.hypot(mx - gx[0], my - gy[0])))
        e_hsg.append(float(np.hypot(hx - gx[0], hy - gy[0])))
        e_srl.append(float(np.hypot(sx - gx[0], sy - gy[0])))
        tids.append((camera, tid))
        row_ids.append((camera, tid, frame))
    return {
        "mid": np.asarray(e_mid), "hsg": np.asarray(e_hsg), "srl": np.asarray(e_srl),
        "cluster": tids, "ids": row_ids, "skipped_no_gt": skipped,
    }


def tracks_of(errors: dict) -> list[np.ndarray]:
    groups: dict = {}
    for index, key in enumerate(errors["cluster"]):
        groups.setdefault(key, []).append(index)
    blocks = []
    for indexes in groups.values():
        chosen = np.asarray(indexes)
        block = np.column_stack([errors["mid"][chosen], errors["hsg"][chosen], errors["srl"][chosen]])
        if np.isfinite(block).all():
            blocks.append(block)
    return blocks


def delta(sample: np.ndarray, left: int, right: int) -> np.ndarray:
    return np.asarray([
        median(sample[:, left]) - median(sample[:, right]),
        rmse(sample[:, left]) - rmse(sample[:, right]),
    ])


def interval_for(draws: np.ndarray, point: np.ndarray) -> dict:
    med_lo, med_hi = np.percentile(draws[:, 0], [2.5, 97.5])
    rmse_lo, rmse_hi = np.percentile(draws[:, 1], [2.5, 97.5])
    return {
        "median": {"d": float(point[0]), "lo": float(med_lo), "hi": float(med_hi)},
        "rmse": {"d": float(point[1]), "lo": float(rmse_lo), "hi": float(rmse_hi)},
    }


def summarize_errors(errors: dict) -> dict:
    blocks = tracks_of(errors)
    if not blocks:
        return {"n": 0}
    full = np.vstack(blocks)
    pairs = {"hsg_minus_mid": (1, 0), "srl_minus_hsg": (2, 1)}
    point = {name: delta(full, *cols) for name, cols in pairs.items()}
    rng = np.random.default_rng(BOOT_SEED)
    draws = {name: np.empty((BOOT_DRAWS, 2)) for name in pairs}
    for draw in range(BOOT_DRAWS):
        take = rng.integers(0, len(blocks), size=len(blocks))
        sample = np.vstack([blocks[index] for index in take])
        for name, cols in pairs.items():
            draws[name][draw] = delta(sample, *cols)
    return {
        "n": int(full.shape[0]),
        "n_tracks": len(blocks),
        "skipped_no_gt": errors["skipped_no_gt"],
        "levels": {
            "mid": {"median": median(full[:, 0]), "rmse": rmse(full[:, 0])},
            "hsg": {"median": median(full[:, 1]), "rmse": rmse(full[:, 1])},
            "srl": {"median": median(full[:, 2]), "rmse": rmse(full[:, 2])},
        },
        "deltas": {name: interval_for(draws[name], point[name]) for name in pairs},
        "_by_id": {
            row_id: (float(errors["mid"][index]), float(errors["hsg"][index]), float(errors["srl"][index]))
            for index, row_id in enumerate(errors["ids"])
        },
    }


def change_between(base: dict, other: dict) -> dict:
    if not base.get("_by_id") or not other.get("_by_id"):
        return {}
    common = sorted(set(base["_by_id"]) & set(other["_by_id"]))
    if not common:
        return {}
    groups: dict = {}
    for camera, tid, frame in common:
        groups.setdefault((camera, tid), []).append((camera, tid, frame))
    blocks = []
    for members in groups.values():
        left = np.asarray([base["_by_id"][key] for key in members])
        right = np.asarray([other["_by_id"][key] for key in members])
        blocks.append((left, right))
    pairs = {"hsg_minus_mid": (1, 0), "srl_minus_hsg": (2, 1)}
    full_left = np.vstack([block[0] for block in blocks])
    full_right = np.vstack([block[1] for block in blocks])
    point = {name: delta(full_right, *cols) - delta(full_left, *cols) for name, cols in pairs.items()}
    rng = np.random.default_rng(BOOT_SEED)
    draws = {name: np.empty((BOOT_DRAWS, 2)) for name in pairs}
    count = len(blocks)
    for draw in range(BOOT_DRAWS):
        take = rng.integers(0, count, size=count)
        sample_left = np.vstack([blocks[index][0] for index in take])
        sample_right = np.vstack([blocks[index][1] for index in take])
        for name, cols in pairs.items():
            draws[name][draw] = delta(sample_right, *cols) - delta(sample_left, *cols)
    return {name: interval_for(draws[name], point[name]) for name in pairs}


def public_summary(stats: dict) -> dict:
    return {key: value for key, value in stats.items() if not key.startswith("_")}


def run_site(site: str) -> dict:
    print(f"[clock] {site}", flush=True)
    cut = test_start(site)
    test = load_test_keys(site)
    adapted, sizes = load_adapted(site)
    polygons = camera_polygons(site)
    deploy = json.loads((ARTIFACT_ROOT / "results" / "recorded" / "deploy_support" / f"{site}_F0.json").read_text(encoding="utf-8"))
    report = json.loads(match_report(site).read_text(encoding="utf-8"))
    det_conn = connect_ro(geometry_db(site))
    gt_conn = connect_ro(origin_db(site))
    cameras = {}
    try:
        for camera in SITES[site]["cameras"]:
            print(f"[clock] {site} {camera}", flush=True)
            hsg, mid, boxes, published = load_camera(det_conn, site, camera, adapted)
            gt_fit = load_gt(gt_conn, site, camera, cut)
            gt_full = load_gt(gt_conn, site, camera, None)
            cam_test = {key for key in test if key[0] == camera}
            published_test = {key for key in published if (camera, key[0], key[1]) in cam_test}
            sources = {}
            for source, tracks in (("midpoint", mid), ("hsg", hsg)):
                sources[source] = arm_from_source(source, tracks, gt_fit, gt_full, camera, cam_test, cut)
            stored = stored_clock(report["cameras"][camera])
            stored_frames = assign_test(hsg, gt_full, stored, camera, cam_test)
            cameras[camera] = {
                "hsg": hsg, "mid": mid, "boxes": boxes, "gt": gt_full,
                "published_test": published_test, "stored_frames": stored_frames,
                "stored_clock": stored, "sources": sources,
            }
    finally:
        det_conn.close()
        gt_conn.close()
    return assemble(site, cut, cameras, adapted, sizes, polygons, deploy)


def arm_from_source(source, tracks, gt_fit, gt_full, camera, cam_test, cut) -> dict:
    pre = slice_tracks(tracks, cut)
    dt0 = estimate_dt0(pre, gt_fit)
    out = {"dt0": dt0}
    if not dt0["valid"]:
        out["const_frames"] = None
        out["affine"] = {"valid": False, "status": "dt0_invalid"}
        out["const_clock"] = None
        out["affine_clock"] = None
        return out
    const = TimeMap(dt0["dt0"])
    out["const_clock"] = const
    out["const_frames"] = assign_test(tracks, gt_full, const, camera, cam_test)
    affine = estimate_affine(dt0["dt0"], pre, gt_fit)
    out["affine"] = {key: value for key, value in affine.items()}
    out["affine_clock"] = TimeMap(dt0["dt0"], affine["a"], affine["b"]) if affine["valid"] else None
    out["affine_frames"] = (
        assign_test(tracks, gt_full, out["affine_clock"], camera, cam_test) if affine["valid"] else None
    )
    return out


def stored_clock(info: dict):
    return TimeMap(
        float(info["dt_applied_sec"]),
        float(info.get("drift_const_ms", 0.0)) / 1000.0,
        float(info.get("drift_slope_ms_per_min", 0.0)) / 60_000.0,
    )


def common_rows(cameras: dict) -> tuple[list, dict]:
    rows = []
    retention = {}
    for camera, item in cameras.items():
        arms = {
            "midpoint_const": item["sources"]["midpoint"]["const_frames"],
            "midpoint_affine": item["sources"]["midpoint"]["affine_frames"],
            "hsg_const": item["sources"]["hsg"]["const_frames"],
            "hsg_affine": item["sources"]["hsg"]["affine_frames"],
        }
        if any(frames is None for frames in arms.values()):
            retention[camera] = {"status": "clock_invalid", "published": len(item["published_test"]), "common": 0}
            continue
        keys = set.intersection(*(set(frames) for frames in arms.values()), item["published_test"])
        kept = []
        for key in keys:
            gids = {frames[key][0] for frames in arms.values()}
            stored = item["stored_frames"].get(key)
            if len(gids) != 1:
                continue
            gid = next(iter(gids))
            if stored is not None and stored[0] != gid:
                continue
            _gid, stamp = arms["hsg_const"][key]
            kept.append((camera, key[0], key[1], gid, stamp))
        rows.extend(kept)
        retention[camera] = {
            "status": "ok", "published": len(item["published_test"]), "common": len(kept),
            "fraction": (len(kept) / len(item["published_test"])) if item["published_test"] else None,
        }
    return rows, retention


def camera_clocks(item: dict) -> dict:
    clocks = {}
    for source in ("midpoint", "hsg"):
        arm = item["sources"][source]
        if arm["const_clock"] is not None:
            clocks[f"{source}_const"] = arm["const_clock"]
        if arm["affine_clock"] is not None:
            clocks[f"{source}_affine"] = arm["affine_clock"]
    clocks["published"] = item["stored_clock"]
    return clocks


def assemble(site, cut, cameras, adapted, sizes, polygons, deploy) -> dict:
    rows, retention = common_rows(cameras)
    by_camera: dict[str, list] = {}
    for row in rows:
        by_camera.setdefault(row[0], []).append(row)
    srl = {}
    hsg = {}
    mid = {}
    gt = {}
    for camera, item in cameras.items():
        local = [(row[1], row[2]) for row in by_camera.get(camera, [])]
        srl[camera] = srl_map(site, camera, local, item["boxes"], item["hsg"], adapted, sizes, polygons, deploy)
        hsg[camera] = item["hsg"]
        mid[camera] = item["mid"]
        gt[camera] = item["gt"]
    names = ["published"]
    for source in ("midpoint", "hsg"):
        if all(item["sources"][source]["const_clock"] is not None for item in cameras.values()):
            names.append(f"{source}_const")
        if all(item["sources"][source]["affine_clock"] is not None for item in cameras.values()):
            names.append(f"{source}_affine")
    scored = {}
    for name in names:
        parts = []
        skipped = 0
        for camera, item in cameras.items():
            clock = camera_clocks(item)[name] if name == "published" or name in camera_clocks(item) else None
            if clock is None:
                parts = None
                break
            local_rows = by_camera.get(camera, [])
            if not local_rows:
                continue
            one = score(clock, local_rows, hsg, mid, srl, gt)
            parts.append(one)
            skipped += one["skipped_no_gt"]
        if not parts:
            scored[name] = {"n": 0}
            continue
        merged = {
            "mid": np.concatenate([part["mid"] for part in parts]),
            "hsg": np.concatenate([part["hsg"] for part in parts]),
            "srl": np.concatenate([part["srl"] for part in parts]),
            "cluster": [key for part in parts for key in part["cluster"]],
            "ids": [key for part in parts for key in part["ids"]],
            "skipped_no_gt": skipped,
        }
        scored[name] = summarize_errors(merged)
    changes = {}
    for left, right in (
        ("hsg_const", "hsg_affine"),
        ("midpoint_const", "midpoint_affine"),
        ("midpoint_const", "hsg_const"),
        ("midpoint_affine", "hsg_affine"),
        ("published", "hsg_const"),
        ("published", "hsg_affine"),
    ):
        if left in scored and right in scored and scored[left].get("n") and scored[right].get("n"):
            changes[f"{right}_minus_{left}"] = change_between(scored[left], scored[right])
    published_n = sum(len(item["published_test"]) for item in cameras.values())
    meta = {}
    for camera, item in cameras.items():
        meta[camera] = {}
        for source in ("midpoint", "hsg"):
            arm = item["sources"][source]
            published_frames = item["published_test"]
            meta[camera][source] = {
                "dt0": arm["dt0"],
                "affine": arm["affine"] if "affine" in arm else arm.get("affine"),
                "const_n": None if arm["const_frames"] is None else len(arm["const_frames"]),
                "affine_n": None if arm.get("affine_frames") is None else len(arm["affine_frames"]),
                "const_jaccard": None if arm["const_frames"] is None else jaccard(set(arm["const_frames"]), published_frames),
                "affine_jaccard": None if arm.get("affine_frames") is None else jaccard(set(arm["affine_frames"]), published_frames),
            }
        # affine details live inside arm_from_source under the key affine for the fit dict
    # The structure from arm_from_source stores affine fit under key "affine" only as the fit dict
    # Rebuild metadata from sources directly below.
    meta = camera_metadata(cameras)
    return {
        "site": site,
        "label": label_of(site),
        "test_start_sec": cut,
        "published_n": published_n,
        "common_n": len(rows),
        "common_fraction": (len(rows) / published_n) if published_n else None,
        "retention": retention_of(cameras, rows),
        "clocks": meta,
        "scores": {name: public_summary(value) for name, value in scored.items()},
        "changes": changes,
    }


def retention_of(cameras, rows) -> dict:
    counts: dict[str, int] = {}
    for camera, _tid, _frame, _gid, _stamp in rows:
        counts[camera] = counts.get(camera, 0) + 1
    out = {}
    for camera, item in cameras.items():
        published = len(item["published_test"])
        common = counts.get(camera, 0)
        invalid = any(item["sources"][source]["const_frames"] is None or item["sources"][source]["affine_frames"] is None for source in ("midpoint", "hsg"))
        out[camera] = {
            "published": published,
            "common": 0 if invalid else common,
            "fraction": None if invalid or not published else common / published,
            "status": "clock_invalid" if invalid else "ok",
        }
    return out


def camera_metadata(cameras: dict) -> dict:
    out = {}
    for camera, item in cameras.items():
        out[camera] = {}
        for source in ("midpoint", "hsg"):
            arm = item["sources"][source]
            published = item["published_test"]
            affine = arm["affine"]
            out[camera][source] = {
                "dt0": arm["dt0"],
                "affine": affine,
                "const_n": None if arm["const_frames"] is None else len(arm["const_frames"]),
                "affine_n": None if arm["affine_frames"] is None else len(arm["affine_frames"]),
                "const_jaccard": None if arm["const_frames"] is None else jaccard(set(arm["const_frames"]), published),
                "affine_jaccard": None if arm["affine_frames"] is None else jaccard(set(arm["affine_frames"]), published),
            }
    return out


def arm_from_source(source, tracks, gt_fit, gt_full, camera, cam_test, cut):
    pre = slice_tracks(tracks, cut)
    dt0 = estimate_dt0(pre, gt_fit)
    out = {
        "dt0": dt0, "const_clock": None, "const_frames": None,
        "affine": {"valid": False, "status": "dt0_invalid"}, "affine_clock": None, "affine_frames": None,
    }
    if not dt0["valid"]:
        return out
    const = TimeMap(dt0["dt0"])
    out["const_clock"] = const
    out["const_frames"] = assign_test(tracks, gt_full, const, camera, cam_test)
    affine = estimate_affine(dt0["dt0"], pre, gt_fit)
    out["affine"] = affine
    if affine["valid"]:
        out["affine_clock"] = TimeMap(dt0["dt0"], affine["a"], affine["b"])
        out["affine_frames"] = assign_test(tracks, gt_full, out["affine_clock"], camera, cam_test)
    return out


def fmt_interval(block: dict, key: str) -> str:
    item = block[key]
    return f"{item['d']:+.4f} [{item['lo']:+.4f}, {item['hi']:+.4f}]"


def write_markdown(payload: dict, path) -> None:
    lines = [
        "# Clock-source check",
        "",
        "Clocks are fit before the test interval from either the fixed midpoint or HSG. HSG-SRL seed 0 is scored only. Deltas are median(e_A)-median(e_B) and RMSE(e_A)-RMSE(e_B) on the common set C.",
        "",
    ]
    for site in payload["sites"]:
        lines += [
            f"## {site['label']}",
            "",
            f"Test start {site['test_start_sec']:.3f} s. Common set {site['common_n']} / published {site['published_n']} ({site['common_fraction']}).",
            "",
            "| Camera | Published | Common | Fraction | Status |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for camera, item in site["retention"].items():
            fraction = "" if item["fraction"] is None else f"{item['fraction']:.3f}"
            lines.append(f"| {camera} | {item['published']} | {item['common']} | {fraction} | {item['status']} |")
        lines += ["", "| Clock | n | HSG−mid median | HSG−mid RMSE | SRL−HSG median | SRL−HSG RMSE |", "| --- | ---: | --- | --- | --- | --- |"]
        for name, score in site["scores"].items():
            if not score.get("n"):
                lines.append(f"| {name} | 0 |  |  |  |  |")
                continue
            deltas = score["deltas"]
            lines.append(
                f"| {name} | {score['n']} | {fmt_interval(deltas['hsg_minus_mid'], 'median')} | "
                f"{fmt_interval(deltas['hsg_minus_mid'], 'rmse')} | {fmt_interval(deltas['srl_minus_hsg'], 'median')} | "
                f"{fmt_interval(deltas['srl_minus_hsg'], 'rmse')} |"
            )
        lines += ["", "| Clock change | HSG−mid median | HSG−mid RMSE | SRL−HSG median | SRL−HSG RMSE |", "| --- | --- | --- | --- | --- |"]
        for name, change in site["changes"].items():
            if "hsg_minus_mid" not in change:
                lines.append(f"| {name} | {change.get('status', '')} |  |  |  |")
                continue
            lines.append(
                f"| {name} | {fmt_interval(change['hsg_minus_mid'], 'median')} | {fmt_interval(change['hsg_minus_mid'], 'rmse')} | "
                f"{fmt_interval(change['srl_minus_hsg'], 'median')} | {fmt_interval(change['srl_minus_hsg'], 'rmse')} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = ensure()
    sites = [run_site(site) for site in SITES_CLOCK]
    payload = {"sites": sites}
    write_json(out / "clock_check.json", payload)
    write_markdown(payload, out / "clock_check.md")
    print(f"[clock] wrote {out / 'clock_check.md'}", flush=True)


if __name__ == "__main__":
    main()
