#!/usr/bin/env python3
"""Build the chronological split and detector-to-ground-truth manifests.

The filename keeps the internal tag fab2. This is the split used by the
published 60/10/30 protocol, not a separate method.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.fab2_common import (  # noqa: E402
    CANONICAL_PREDICTION_FIELDS,
    FAB2_DEFAULT_RESULTS,
    GT_INDEX_FIELDS,
    checkpoint_meta,
    connect_db,
    ensure_dir,
    event_key,
    finite_float,
    infer_dataset_labels,
    latest_calibration_rows,
    load_import_audit_metadata,
    load_slice_rules,
    read_csv_rows,
    resolve_path,
    rule_slice_for,
    selected_scenes,
    source_detection_id,
    split_csv_arg,
    table_columns,
    utc_now,
    write_csv,
    write_json,
    write_run_metadata,
)
from utils.scene_geometry_db import load_scene_overlap_pairs  # noqa: E402


SPLIT_FIELDS = [
    "dataset",
    "site",
    "recording",
    "split",
    "scene_id",
    "batch_id",
    "timestamp",
    "frame_index",
    "camera_id",
    "source_detection_id",
    "event_key",
    "local_track_id",
    "gt_global_id",
    "gt_world_x",
    "gt_world_y",
    "camera_distance_m",
    "class_id",
    "class_name",
    "spatial_slice",
    "has_gt",
]

TOPOLOGY_FIELDS = [
    "dataset",
    "site",
    "recording",
    "scene_id",
    "record_type",
    "camera_id",
    "camera_a",
    "camera_b",
    "scale_px_per_meter",
    "homography_source",
    "point_count",
    "inlier_count",
    "reproj_rms",
    "geometry_version",
    "bfs_hop",
    "calibrated_at",
]

CHECKPOINT_FIELDS = [
    "dataset",
    "site",
    "recording",
    "scene_id",
    "method",
    "checkpoint_path",
    "exists",
    "sha256",
    "bytes",
    "feature_version",
    "risk_scalar_mode",
    "anchor_mode",
    "max_residual_m",
    "mvc_pair_count",
    "raw_mvc_pair_count",
    "seed",
    "batch_id",
]

ELIGIBILITY_FIELDS = [
    "dataset",
    "site",
    "method",
    "comparison_layer",
    "evaluation_regime",
    "training_supervision",
    "missing_conditions",
    "included_position",
    "status",
]

RUN_MANIFEST_FIELDS = [
    "dataset",
    "site",
    "recording",
    "scene_id",
    "batch_id",
    "method",
    "evaluation_regime",
    "training_supervision",
    "runtime_calibration",
    "learner_calibration_input",
    "calibration_source",
    "detection_source",
    "fusion_location",
    "tracking_backend",
    "world_coord_field",
    "checkpoint",
    "seed",
    "command_hint",
]

ADAPTER_FIELDS = [
    "dataset",
    "site",
    "recording",
    "scene_id",
    "method",
    "coordinate_system",
    "vertical_axis",
    "vertical_axis_sign",
    "box_reference_point",
    "dimensions_order",
    "transform_direction",
    "ground_height_residual_m",
    "visual_audit_index",
    "status",
]


def _split_by_time(
    timestamps: list[float],
    *,
    train_frac: float,
    val_frac: float,
) -> dict[float, str]:
    uniq = sorted(set(float(t) for t in timestamps))
    if not uniq:
        return {}
    n = len(uniq)
    train_end = int(round(n * train_frac))
    val_end = int(round(n * (train_frac + val_frac)))
    train_end = max(0, min(n, train_end))
    val_end = max(train_end, min(n, val_end))
    out: dict[float, str] = {}
    for i, t in enumerate(uniq):
        if i < train_end:
            out[t] = "train"
        elif i < val_end:
            out[t] = "val"
        else:
            out[t] = "test"
    return out


def _scene_split(scene_id: str, args: argparse.Namespace, time_split: dict[float, str], t: float) -> str:
    train_scenes = set(split_csv_arg(args.train_scenes))
    val_scenes = set(split_csv_arg(args.val_scenes))
    test_scenes = set(split_csv_arg(args.test_scenes))
    if args.split_mode == "scene-list":
        if scene_id in train_scenes:
            return "train"
        if scene_id in val_scenes:
            return "val"
        if scene_id in test_scenes:
            return "test"
        return args.default_split
    if args.split_mode == "time":
        return time_split.get(float(t), args.default_split)
    return args.default_split


def _position_from_mapping(row: dict[str, Any]) -> tuple[float, float] | None:
    x = finite_float(
        row.get("camera_world_x"),
        finite_float(row.get("world_x"), finite_float(row.get("x_m"), finite_float(row.get("x")))),
    )
    y = finite_float(
        row.get("camera_world_y"),
        finite_float(row.get("world_y"), finite_float(row.get("y_m"), finite_float(row.get("y")))),
    )
    if x is None or y is None:
        return None
    return float(x), float(y)


def _position_from_value(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        return _position_from_mapping(value)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = finite_float(value[0])
        y = finite_float(value[1])
        if x is not None and y is not None:
            return float(x), float(y)
    return None


def _load_camera_positions(args: argparse.Namespace) -> dict[tuple[str, str], tuple[float, float]]:
    """Load preregistered camera ground positions for distance strata.

    Keys are ``(scene_id, camera_id)``. A blank scene id acts as a camera-name
    fallback shared by all selected scenes.
    """

    positions: dict[tuple[str, str], tuple[float, float]] = {}

    def add(scene: str, camera: str, value: Any) -> None:
        scene = str(scene or "").strip()
        camera = str(camera or "").strip()
        pos = _position_from_value(value)
        if camera and pos is not None:
            positions[(scene, camera)] = pos

    if args.camera_positions_csv:
        for row in read_csv_rows(args.camera_positions_csv):
            scene = row.get("scene_id", row.get("scene", ""))
            camera = row.get("camera_id", row.get("camera_name", row.get("camera", "")))
            add(scene, camera, row)

    if args.camera_positions_json:
        p = resolve_path(args.camera_positions_json)
        payload = json.loads(p.read_text(encoding="utf-8"))
        root: Any = payload
        if isinstance(payload, dict):
            root = (
                payload.get("camera_positions_m_by_scene")
                or payload.get("camera_positions_by_scene")
                or payload.get("camera_positions")
                or payload
            )
        if isinstance(root, list):
            for item in root:
                if isinstance(item, dict):
                    scene = item.get("scene_id", item.get("scene", ""))
                    camera = item.get("camera_id", item.get("camera_name", item.get("camera", "")))
                    add(scene, camera, item)
        elif isinstance(root, dict):
            for scene, scene_value in root.items():
                if isinstance(scene_value, dict):
                    for camera, value in scene_value.items():
                        add(str(scene), str(camera), value)
                else:
                    add("", str(scene), scene_value)
    return positions


def _camera_distance_from_position(
    camera_positions: dict[tuple[str, str], tuple[float, float]],
    *,
    scene_id: str,
    camera_id: str,
    gt_x: Any,
    gt_y: Any,
) -> float | None:
    gx = finite_float(gt_x)
    gy = finite_float(gt_y)
    if gx is None or gy is None:
        return None
    pos = camera_positions.get((scene_id, camera_id)) or camera_positions.get(("", camera_id))
    if pos is None:
        return None
    return float(math.hypot(float(gx) - pos[0], float(gy) - pos[1]))


def _load_gt_id_overrides(path: str | None) -> dict[str, int] | None:
    """Read ``det_gt_pairs.csv``: source_detection_id -> true GT identity."""
    if not path:
        return None
    out: dict[str, int] = {}
    for row in read_csv_rows(path):
        sid = row.get("source_detection_id", "")
        gid = row.get("gt_global_id", "")
        if sid and gid != "":
            out[sid] = int(float(gid))
    return out


def _build_split_rows(
    conn: sqlite3.Connection,
    scenes: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    cols = table_columns(conn, "trajectory_observations")
    have_gt = {"outside_reference_x", "outside_reference_y"}.issubset(cols)
    camera_distance_expr = "camera_distance_m" if "camera_distance_m" in cols else "NULL AS camera_distance_m"
    camera_positions = _load_camera_positions(args)
    rows_out: list[dict[str, Any]] = []
    slice_rules = load_slice_rules(args.slice_csv)
    gt_ids_by_source = _load_gt_id_overrides(args.gt_id_csv)
    for scene_id in scenes:
        labels = infer_dataset_labels(scene_id, args.dataset)
        all_times = [
            float(r[0])
            for r in conn.execute(
                "SELECT timestamp_sec FROM trajectory_observations "
                "WHERE scene_id=? AND batch_id=? ORDER BY timestamp_sec",
                (scene_id, args.batch),
            ).fetchall()
        ]
        time_split = _split_by_time(all_times, train_frac=args.train_frac, val_frac=args.val_frac)
        rows = conn.execute(
            f"""
            SELECT camera_name, local_track_id, global_id, frame_index,
                   timestamp_sec, outside_reference_x, outside_reference_y,
                   class_id, class_name, {camera_distance_expr}
              FROM trajectory_observations
             WHERE scene_id=? AND batch_id=?
             ORDER BY timestamp_sec, frame_index, camera_name, local_track_id
            """,
            (scene_id, args.batch),
        ).fetchall()
        for r in rows:
            sid = source_detection_id(scene_id, args.batch, r["camera_name"], r["local_track_id"], r["frame_index"])
            ekey = event_key(scene_id, r["camera_name"], r["local_track_id"], r["frame_index"])
            split = _scene_split(scene_id, args, time_split, float(r["timestamp_sec"]))
            gx = r["outside_reference_x"] if have_gt else None
            gy = r["outside_reference_y"] if have_gt else None
            # Detector tracks carry their own identities, so the GT identity has
            # to come from the association step rather than from global_id.
            gt_id: int | str = int(r["global_id"]) if r["global_id"] is not None else ""
            if gt_ids_by_source is not None:
                gt_id = gt_ids_by_source.get(sid, "")
                if gt_id == "":
                    gx = gy = None
            db_camera_distance = finite_float(r["camera_distance_m"])
            camera_distance = db_camera_distance
            if camera_distance is None:
                camera_distance = _camera_distance_from_position(
                    camera_positions,
                    scene_id=scene_id,
                    camera_id=str(r["camera_name"]),
                    gt_x=gx,
                    gt_y=gy,
                )
            slice_row = {
                "scene_id": scene_id,
                "camera_id": str(r["camera_name"]),
                "timestamp": float(r["timestamp_sec"]),
                "gt_world_x": gx,
                "gt_world_y": gy,
            }
            rows_out.append(
                {
                    "dataset": labels.dataset,
                    "site": labels.site,
                    "recording": labels.recording,
                    "split": split,
                    "scene_id": scene_id,
                    "batch_id": args.batch,
                    "timestamp": float(r["timestamp_sec"]),
                    "frame_index": int(r["frame_index"]),
                    "camera_id": str(r["camera_name"]),
                    "source_detection_id": sid,
                    "event_key": ekey,
                    "local_track_id": int(r["local_track_id"]),
                    "gt_global_id": gt_id,
                    "gt_world_x": gx,
                    "gt_world_y": gy,
                    "camera_distance_m": camera_distance if camera_distance is not None else "",
                    "class_id": r["class_id"] if r["class_id"] is not None else "",
                    "class_name": r["class_name"] or "",
                    "spatial_slice": rule_slice_for(slice_row, slice_rules) or "",
                    "has_gt": int(gx is not None and gy is not None),
                }
            )
    return rows_out


def _build_topology_rows(conn: sqlite3.Connection, scenes: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for scene_id in scenes:
        labels = infer_dataset_labels(scene_id, args.dataset)
        meta = load_import_audit_metadata(conn, scene_id, args.batch)
        hop_by_camera = {}
        if isinstance(meta.get("bfs_hop_by_camera"), dict):
            hop_by_camera = meta["bfs_hop_by_camera"]
        for r in latest_calibration_rows(conn, scene_id):
            rows_out.append(
                {
                    "dataset": labels.dataset,
                    "site": labels.site,
                    "recording": labels.recording,
                    "scene_id": scene_id,
                    "record_type": "camera_calibration",
                    "camera_id": r["camera_name"],
                    "camera_a": "",
                    "camera_b": "",
                    "scale_px_per_meter": r["scale_px_per_meter"],
                    "homography_source": r["source"],
                    "point_count": r["point_count"],
                    "inlier_count": r["inlier_count"],
                    "reproj_rms": r["reproj_rms"],
                    "geometry_version": r["geometry_version"],
                    "bfs_hop": hop_by_camera.get(str(r["camera_name"]), ""),
                    "calibrated_at": r["calibrated_at"],
                }
            )
        cameras = [str(r["camera_id"]) for r in rows_out if r.get("scene_id") == scene_id and r.get("record_type") == "camera_calibration"]
        for a, b in sorted(load_scene_overlap_pairs(conn, scene_id, cameras or None)):
            rows_out.append(
                {
                    "dataset": labels.dataset,
                    "site": labels.site,
                    "recording": labels.recording,
                    "scene_id": scene_id,
                    "record_type": "overlap_edge",
                    "camera_id": "",
                    "camera_a": a,
                    "camera_b": b,
                    "scale_px_per_meter": "",
                    "homography_source": "",
                    "point_count": "",
                    "inlier_count": "",
                    "reproj_rms": "",
                    "geometry_version": "",
                    "bfs_hop": "",
                    "calibrated_at": "",
                }
            )
    return rows_out


def _build_checkpoint_rows(scenes: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checkpoints = args.checkpoint or []
    for scene_id in scenes:
        labels = infer_dataset_labels(scene_id, args.dataset)
        for ckpt in checkpoints:
            meta = checkpoint_meta(ckpt)
            rows.append(
                {
                    "dataset": labels.dataset,
                    "site": labels.site,
                    "recording": labels.recording,
                    "scene_id": scene_id,
                    "method": args.checkpoint_method,
                    "checkpoint_path": meta.get("path", ckpt),
                    "exists": int(bool(meta.get("exists"))),
                    "sha256": meta.get("sha256", ""),
                    "bytes": meta.get("bytes", ""),
                    "feature_version": meta.get("feature_version", ""),
                    "risk_scalar_mode": meta.get("risk_scalar_mode", ""),
                    "anchor_mode": meta.get("anchor_mode", ""),
                    "max_residual_m": meta.get("max_residual_m", ""),
                    "mvc_pair_count": meta.get("mvc_pair_count", ""),
                    "raw_mvc_pair_count": meta.get("raw_mvc_pair_count", ""),
                    "seed": meta.get("seed", args.seed if args.seed is not None else ""),
                    "batch_id": meta.get("batch_id", args.batch),
                }
            )
    return rows


def _build_comparability_rows(scenes: list[str], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    eligibility: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []
    adapter: list[dict[str, Any]] = []
    methods = split_csv_arg(args.methods) or [
        "Baseline-IPM",
        "Ours-HSG-SRL-Frozen",
        "BEVHeight",
        "MonoGAE",
        "RopeBEV",
    ]
    for scene_id in scenes:
        labels = infer_dataset_labels(scene_id, args.dataset)
        for method in methods:
            low = method.lower()
            if "baseline" in low or "ipm" in low:
                layer = "A"
                regime = "Frozen-Geometry"
                sup = "none"
                missing = ""
                status = "ready"
                position = "frozen-geometry comparison"
                coord = "world_x/world_y"
            elif "hsg" in low or "srl" in low or "ours" in low:
                layer = "A"
                regime = "Frozen-Residual"
                sup = "unlabeled MVC train split"
                missing = "checkpoint/final_world_x_y if not generated"
                status = "ready_after_checkpoint"
                position = "frozen-geometry comparison"
                coord = "final_world_x/final_world_y"
            else:
                layer = "B_or_C"
                regime = "Target-Supervised_or_Official-Native"
                sup = "method-specific 3D supervision"
                missing = "adapter audit, native labels/calibration, frozen predictions"
                status = "N/A_until_adapter_audit"
                position = "separate reference table"
                coord = "predictions CSV"
            eligibility.append(
                {
                    "dataset": labels.dataset,
                    "site": labels.site,
                    "method": method,
                    "comparison_layer": layer,
                    "evaluation_regime": regime,
                    "training_supervision": sup,
                    "missing_conditions": missing,
                    "included_position": position,
                    "status": status,
                }
            )
            run_manifest.append(
                {
                    "dataset": labels.dataset,
                    "site": labels.site,
                    "recording": labels.recording,
                    "scene_id": scene_id,
                    "batch_id": args.batch,
                    "method": method,
                    "evaluation_regime": regime,
                    "training_supervision": sup,
                    "runtime_calibration": "homography",
                    "learner_calibration_input": "local homography features" if layer == "A" else "method-specific",
                    "calibration_source": "camera_calibrations",
                    "detection_source": "shared 2D DB observations" if layer == "A" else "method native detector",
                    "fusion_location": "common MCMTT backend" if layer == "A" else "declared by adapter",
                    "tracking_backend": args.tracking_backend,
                    "world_coord_field": coord,
                    "checkpoint": args.checkpoint[0] if args.checkpoint and "hsg" in low else "",
                    "seed": args.seed if args.seed is not None else "",
                    "command_hint": "",
                }
            )
        adapter.append(
            {
                "dataset": labels.dataset,
                "site": labels.site,
                "recording": labels.recording,
                "scene_id": scene_id,
                "method": "fab2_db_homography",
                "coordinate_system": "dataset_world_minus_import_origin",
                "vertical_axis": "not_used_for_planar_homography",
                "vertical_axis_sign": "",
                "box_reference_point": "bottom_center_or_HSG_anchor",
                "dimensions_order": "",
                "transform_direction": "image_to_world_bev",
                "ground_height_residual_m": "",
                "visual_audit_index": "",
                "status": "ready",
            }
        )
    return eligibility, run_manifest, adapter


def build(args: argparse.Namespace) -> None:
    out_root = ensure_dir(Path(args.out_dir))
    manifest_dir = ensure_dir(out_root / "manifest")
    comparability_dir = ensure_dir(out_root / "comparability")
    conn = connect_db(args.db)
    try:
        scenes = selected_scenes(conn, args.batch, args.scene)
        if not scenes:
            raise SystemExit("No scenes found. Pass --scene or check --batch.")
        split_rows = _build_split_rows(conn, scenes, args)
        topology_rows = _build_topology_rows(conn, scenes, args)
        checkpoint_rows = _build_checkpoint_rows(scenes, args)
        eligibility_rows, run_rows, adapter_rows = _build_comparability_rows(scenes, args)
    finally:
        conn.close()

    write_csv(manifest_dir / "dataset_split_manifest.csv", split_rows, SPLIT_FIELDS)
    write_csv(manifest_dir / "gt_index.csv", split_rows, GT_INDEX_FIELDS)
    write_csv(manifest_dir / "homography_topology_manifest.csv", topology_rows, TOPOLOGY_FIELDS)
    write_csv(manifest_dir / "checkpoint_manifest.csv", checkpoint_rows, CHECKPOINT_FIELDS)
    write_csv(comparability_dir / "method_eligibility.csv", eligibility_rows, ELIGIBILITY_FIELDS)
    write_csv(comparability_dir / "method_run_manifest.csv", run_rows, RUN_MANIFEST_FIELDS)
    write_csv(comparability_dir / "coordinate_adapter_audit.csv", adapter_rows, ADAPTER_FIELDS)
    write_json(
        manifest_dir / "manifest_summary.json",
        {
            "created_at": utc_now(),
            "db": str(Path(args.db)),
            "batch_id": args.batch,
            "scenes": scenes,
            "rows": {
                "dataset_split_manifest": len(split_rows),
                "homography_topology_manifest": len(topology_rows),
                "checkpoint_manifest": len(checkpoint_rows),
            },
        },
    )
    write_run_metadata(out_root, "fab2_manifest", args, {"scenes": scenes})
    print(f"FAB2 manifests written under {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Trajectory SQLite DB")
    parser.add_argument("--batch", required=True, help="Batch id to freeze")
    parser.add_argument("--scene", default=None, help="Comma-separated scenes; default all scenes in batch")
    parser.add_argument("--dataset", default=None, help="Override dataset label")
    parser.add_argument("--out-dir", default=str(FAB2_DEFAULT_RESULTS), help="FAB2 result root")
    parser.add_argument("--split-mode", choices=("all-test", "time", "scene-list"), default="all-test")
    parser.add_argument("--default-split", choices=("train", "val", "test", "ignore"), default="test")
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--train-scenes", default="")
    parser.add_argument("--val-scenes", default="")
    parser.add_argument("--test-scenes", default="")
    parser.add_argument("--slice-csv", default=None, help="Optional spatial slice rule CSV")
    parser.add_argument(
        "--gt-id-csv",
        default=None,
        help=(
            "det_gt_pairs.csv from eval_fab2_det_gt_match.py. Required for detector "
            "tracks, where global_id is a predicted identity rather than a GT one; "
            "observations absent from the file are marked has_gt=0."
        ),
    )
    parser.add_argument(
        "--camera-positions-csv",
        default=None,
        help=(
            "Optional preregistered camera ground positions. Columns: "
            "scene_id,camera_id,camera_world_x,camera_world_y; scene_id may be blank as a camera-name fallback."
        ),
    )
    parser.add_argument(
        "--camera-positions-json",
        default=None,
        help=(
            "Optional camera positions JSON, e.g. "
            "{\"camera_positions_m_by_scene\":{\"scene\":{\"cam\":[x,y]}}}."
        ),
    )
    parser.add_argument("--checkpoint", action="append", default=[], help="Checkpoint to record; repeatable")
    parser.add_argument("--checkpoint-method", default="Ours-HSG-SRL-Frozen")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--methods", default="", help="Comma-separated methods for comparability manifests")
    parser.add_argument("--tracking-backend", default="common_frozen_backend")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
