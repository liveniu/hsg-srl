#!/usr/bin/env python3
"""Freeze unlabeled train and validation pairs on uncorrected HSG positions.

Pairs are mined before residual fitting and are not updated by the network.
Test keys and ground-truth coordinates are not targets. A ground-truth
identity check only rejects accidental leakage.
"""

from __future__ import annotations

import json
import sqlite3
import sys

import numpy as np

from bootstrap import ARTIFACT_ROOT, SRC_ROOT
from protocol import (
    DET_BATCH,
    PAIR_DIST_M,
    SITES,
    TEACHER_GAP,
    TEACHER_KAPPA,
    geometry_db,
    overlap_path,
    site_dir,
)

sys.path.insert(0, str(SRC_ROOT))
from geometry.geomfeat import build_frames  # noqa: E402
from pipeline.geometry_guided_residual_mlp import (  # noqa: E402
    GeometryDataView,
    ResidualMLPModel,
    TrainConfig,
    _filter_mvc_pairs_by_world_distance,
    _filter_observations_for_training_protocol,
    _gt_identity_leakage_report,
    _infer_and_calib_wh_from_db,
    _load_training_split_filter,
)
from pipeline.multi_camera_trajectory_fusion import adapt_homography_to_resolution  # noqa: E402


def edges_of(site: str) -> list[tuple[str, str]]:
    path = overlap_path(site)
    if not path.exists():
        path = ARTIFACT_ROOT / "configs" / "overlap" / f"{site}.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    pairs = []
    for edge in blob.get("edges") or []:
        pairs.append(tuple(sorted((str(edge["a"]), str(edge["b"])))))
    return sorted(set(pairs))


def _pack(pairs, frames, cam_index):
    count = len(pairs)
    packed = {
        "fa": np.zeros((count, 7), np.float32),
        "fb": np.zeros((count, 7), np.float32),
        "ea": np.zeros((count, 3), np.float32),
        "eb": np.zeros((count, 3), np.float32),
        "xa": np.zeros((count, 2), np.float64),
        "xb": np.zeros((count, 2), np.float64),
        "ka": np.zeros(count, np.float32),
        "kb": np.zeros(count, np.float32),
        "ca": np.zeros(count, np.int16),
        "cb": np.zeros(count, np.int16),
    }
    for index, pair in enumerate(pairs):
        packed["fa"][index] = pair.obs_a.feature
        packed["fb"][index] = pair.obs_b.feature
        packed["xa"][index] = pair.obs_a.world_geo
        packed["xb"][index] = pair.obs_b.world_geo
        packed["ka"][index] = pair.obs_a.horizon_confidence
        packed["kb"][index] = pair.obs_b.horizon_confidence
        packed["ca"][index] = cam_index[pair.obs_a.camera_name]
        packed["cb"][index] = cam_index[pair.obs_b.camera_name]
    names = [None] * len(cam_index)
    for name, index in cam_index.items():
        names[index] = name
    for side_x, side_c, side_e in (
        (packed["xa"], packed["ca"], packed["ea"]),
        (packed["xb"], packed["cb"], packed["eb"]),
    ):
        for index, name in enumerate(names):
            mask = side_c == index
            if np.any(mask):
                side_e[mask] = frames[name].extras(side_x[mask])
    return packed


def _temporal(pairs, frames):
    count = len(pairs)
    packed = {
        "f0": np.zeros((count, 7), np.float32),
        "f1": np.zeros((count, 7), np.float32),
        "e0": np.zeros((count, 3), np.float32),
        "e1": np.zeros((count, 3), np.float32),
    }
    for index, pair in enumerate(pairs):
        packed["f0"][index] = pair.prev.feature
        packed["f1"][index] = pair.curr.feature
        packed["e0"][index] = frames[pair.prev.camera_name].extras(np.asarray(pair.prev.world_geo)[None, :])
        packed["e1"][index] = frames[pair.curr.camera_name].extras(np.asarray(pair.curr.world_geo)[None, :])
    return packed


def mine_split(conn, view, site, cameras, overlap, split_name, frames, anchor_mode):
    manifest = site_dir(site) / "dataset_split_manifest.csv"
    split_filter = _load_training_split_filter(
        manifest, split_name, scene_ids=[site], batch_id=DET_BATCH,
    )
    calibrations = view.load_camera_calibrations(conn, site, cameras)
    infer_wh, calib_wh = _infer_and_calib_wh_from_db(conn, site, DET_BATCH, cameras)
    frame_wh = {cam: infer_wh.get(cam) or (1920, 1080) for cam in cameras}
    adapted = {}
    for cam, (homography, scale) in calibrations.items():
        infer_size, calib_size = infer_wh.get(cam), calib_wh.get(cam)
        if infer_size and calib_size and infer_size != calib_size:
            homography = adapt_homography_to_resolution(homography, calib_size, infer_size)
        adapted[cam] = (homography, scale)
    observations = list(view.iter_observations_from_db(
        conn, site, DET_BATCH, cameras, adapted,
        frame_wh_by_camera=frame_wh,
        anchor_mode=anchor_mode,
    ))
    _stats, observations = _filter_observations_for_training_protocol(
        observations, scene_id=site, batch_id=DET_BATCH,
        split_filter=split_filter, time_start=None, time_end=None,
    )
    leak = _gt_identity_leakage_report(
        observations, scene_id=site, batch_id=DET_BATCH, split_filter=split_filter,
    )
    if leak["compared"] >= max(1, min(32, len(observations))) and leak["match_fraction"] >= 0.95:
        raise SystemExit(f"{site} {split_name}: global_id matches GT identity")
    pairs = view.mine_mvc_pairs(observations, overlap)
    pairs = _filter_mvc_pairs_by_world_distance(pairs, PAIR_DIST_M)
    model = ResidualMLPModel.create(TrainConfig(
        min_teacher_confidence=TEACHER_KAPPA,
        min_teacher_confidence_gap=TEACHER_GAP,
        max_pair_dist_m=PAIR_DIST_M,
    ))
    selected = model.select_mvc_pairs(pairs)
    temporal = view.mine_temporal_pairs(observations) if split_name == "train" else []
    return selected, temporal, len(observations)


def components_and_weights(pack, cameras):
    parent = {cam: cam for cam in cameras}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    weights = {cam: 0.0 for cam in cameras}
    for index in range(len(pack["ca"])):
        left = cameras[int(pack["ca"][index])]
        right = cameras[int(pack["cb"][index])]
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right
        weights[left] += float(pack["ka"][index])
        weights[right] += float(pack["kb"][index])
    groups: dict[str, list[str]] = {}
    for cam in cameras:
        groups.setdefault(find(cam), []).append(cam)
    comps = [sorted(members) for members in groups.values() if len(members) >= 2]
    return comps, weights


def freeze_site(site: str, anchor_mode: str = "multiplicative_gating_offsetted") -> None:
    dest = site_dir(site) / "pairs" / f"{site}.npz"
    meta_path = site_dir(site) / "pairs" / f"{site}.json"
    if dest.exists() and meta_path.exists():
        print(f"[freeze] reuse {site}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    graph_path = overlap_path(site)
    if not graph_path.exists():
        graph_path = ARTIFACT_ROOT / "configs" / "overlap" / f"{site}.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    cameras = [str(cam) for cam in graph["cameras"]]
    overlap = set(edges_of(site))
    conn = sqlite3.connect(str(geometry_db(site)))
    view = GeometryDataView()
    frames = build_frames(conn, site, cameras, DET_BATCH)
    train_pairs, temporal, n_train_obs = mine_split(
        conn, view, site, cameras, overlap, "train", frames, anchor_mode,
    )
    val_pairs, _tmp, n_val_obs = mine_split(
        conn, view, site, cameras, overlap, "val", frames, anchor_mode,
    )
    conn.close()
    cam_index = {cam: index for index, cam in enumerate(cameras)}
    train = _pack(train_pairs, frames, cam_index)
    val = _pack(val_pairs, frames, cam_index)
    temporal_pack = _temporal(temporal, frames)
    comps, weights = components_and_weights(train, cameras)
    meta = {
        "site": site,
        "anchor_mode": anchor_mode,
        "cameras": cameras,
        "edges": [f"{a}:{b}" for a, b in sorted(overlap)],
        "n_train_obs": n_train_obs,
        "n_val_obs": n_val_obs,
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "n_temporal": len(temporal),
        "components": comps,
        "weights": weights,
        "uses_test_gt": False,
    }
    np.savez_compressed(
        dest,
        train_fa=train["fa"], train_fb=train["fb"],
        train_ea=train["ea"], train_eb=train["eb"],
        train_xa=train["xa"], train_xb=train["xb"],
        train_ka=train["ka"], train_kb=train["kb"],
        train_ca=train["ca"], train_cb=train["cb"],
        val_fa=val["fa"], val_fb=val["fb"],
        val_ea=val["ea"], val_eb=val["eb"],
        val_xa=val["xa"], val_xb=val["xb"],
        val_ka=val["ka"], val_kb=val["kb"],
        val_ca=val["ca"], val_cb=val["cb"],
        temp_f0=temporal_pack["f0"], temp_f1=temporal_pack["f1"],
        temp_e0=temporal_pack["e0"], temp_e1=temporal_pack["e1"],
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[freeze] {site} train={len(train_pairs)} val={len(val_pairs)} temporal={len(temporal)}",
        flush=True,
    )


def main() -> None:
    sites = sys.argv[1:] or list(SITES)
    for site in sites:
        freeze_site(site)


if __name__ == "__main__":
    main()
