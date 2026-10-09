"""Read a released geometry database for scoring and timing."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import numpy as np

from bootstrap import SRC_ROOT  # noqa: F401
from protocol import (
    DET_BATCH,
    MAX_T,
    SITES,
    geometry_db,
    match_report,
    split_keys,
    verdict_path,
)


def connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def load_test_keys(site: str, path: Path | None = None) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    path = path or split_keys(site, "test")
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cam = row.get("camera_id") or row.get("camera_name") or ""
            keys.add((cam, int(row["local_track_id"]), int(row["frame_index"])))
    if not keys:
        raise RuntimeError(f"empty test keys for {site}")
    return keys


def load_bias(site: str, path: Path | None = None) -> np.ndarray:
    from evaluation.bev_hota import parse_reference_bias

    path = path or verdict_path(site)
    if not path.exists():
        return np.zeros(2)
    return parse_reference_bias(path.read_text(encoding="utf-8"))


def load_adapted(site: str, db: Path | None = None):
    from pipeline.geometry_guided_residual_mlp import GeometryDataView, _infer_and_calib_wh_from_db
    from pipeline.multi_camera_trajectory_fusion import adapt_homography_to_resolution

    cameras = list(SITES[site]["cameras"])
    conn = connect_ro(db or geometry_db(site))
    view = GeometryDataView()
    calibrations = view.load_camera_calibrations(conn, site, cameras)
    infer_wh, calib_wh = _infer_and_calib_wh_from_db(conn, site, DET_BATCH, cameras)
    adapted, sizes = {}, {}
    for cam, (homography, scale) in calibrations.items():
        infer_size, calib_size = infer_wh.get(cam), calib_wh.get(cam)
        if infer_size and calib_size and infer_size != calib_size:
            homography = adapt_homography_to_resolution(homography, calib_size, infer_size)
        adapted[cam] = (homography, scale)
        sizes[cam] = tuple(infer_size or calib_size or (1920, 1080))
    conn.close()
    return adapted, sizes


def matched_rows(
    site: str,
    db: Path | None = None,
    keys_path: Path | None = None,
    bias_path: Path | None = None,
) -> tuple[list[dict], dict]:
    """Test-split detections that already have a frozen GT attachment."""
    from pipeline.geometry_guided_residual_mlp import compute_geometry_base, xywh_to_xyxy

    test = load_test_keys(site, keys_path)
    bias = load_bias(site, bias_path)
    adapted, sizes = load_adapted(site, db)
    cameras = set(SITES[site]["cameras"])
    max_t = MAX_T.get(site)
    conn = connect_ro(db or geometry_db(site))
    rows: list[dict] = []
    by: dict[tuple[str, int], list[int]] = {}
    query = """SELECT camera_name, local_track_id, frame_index, timestamp_sec,
                      bbox_x, bbox_y, bbox_w, bbox_h,
                      world_x, world_y, outside_reference_x, outside_reference_y
               FROM trajectory_observations
               WHERE scene_id=? AND batch_id=?
                 AND world_x IS NOT NULL AND outside_reference_x IS NOT NULL"""
    for cam, tid, frame, timestamp, bx, by_, bw, bh, wx, wy, gx, gy in conn.execute(
        query, (site, DET_BATCH),
    ):
        if cam not in cameras or cam not in adapted:
            continue
        if (cam, int(tid), int(frame)) not in test:
            continue
        if max_t is not None and float(timestamp) > max_t:
            continue
        if None in (bx, by_, bw, bh, wx, wy, gx, gy):
            continue
        homography, scale = adapted[cam]
        box = xywh_to_xyxy((float(bx), float(by_), float(bw), float(bh)))
        _anchor, _world, feature, _risk = compute_geometry_base(
            box, homography, scale, sizes[cam],
        )
        record = {
            "cam": cam,
            "tid": int(tid),
            "frame": int(frame),
            "t": float(timestamp),
            "box": (float(bx), float(by_), float(bw), float(bh)),
            "feat": feature.astype(np.float32),
            "xy": np.array([float(wx), float(wy)]),
            "gt": np.array([float(gx) + bias[0], float(gy) + bias[1]]),
        }
        by.setdefault((cam, int(tid)), []).append(len(rows))
        rows.append(record)
    conn.close()
    return rows, by


def load_mlp(site: str, seed: int, checkpoint_dir: Path | None = None):
    import torch
    from pipeline.geometry_guided_residual_mlp import GeometryGuidedResidualMLP
    from protocol import site_dir

    directory = checkpoint_dir or (site_dir(site) / "checkpoints")
    path = directory / f"{site}_F0_seed{seed}.pt"
    if not path.exists():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = GeometryGuidedResidualMLP(
        in_dim=int(blob["in_dim"]),
        hidden=64,
        max_residual_m=float(blob["max_residual_m"]),
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def positions(
    site: str,
    seed: int,
    rows: list[dict],
    deploy: dict,
    checkpoint_dir: Path | None = None,
) -> np.ndarray:
    xy = np.stack([row["xy"] for row in rows]) if rows else np.zeros((0, 2))
    if deploy.get("deploy") != "model" or not rows:
        return xy.copy()
    model = load_mlp(site, seed, checkpoint_dir)
    if model is None:
        return xy.copy()
    import torch

    features = torch.from_numpy(np.stack([row["feat"] for row in rows]).astype(np.float32))
    with torch.no_grad():
        residual = model(features).cpu().numpy().astype(np.float64)
    opened = set(deploy.get("S") or [])
    out = xy.copy()
    for index, row in enumerate(rows):
        if row["cam"] in opened:
            out[index] = out[index] + residual[index]
    return out


def reference_report(site: str) -> Path:
    return match_report(site)
