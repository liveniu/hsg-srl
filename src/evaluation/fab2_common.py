#!/usr/bin/env python3
"""Shared utilities for the detector-to-ground-truth attachment.

The filename keeps the internal tag fab2. These helpers are the published
attachment and split utilities. They keep the evaluation scripts tied to the existing
trajectory SQLite schema while exporting paper-facing CSV files with stable
fields.  The code intentionally treats GT as an external audit source: when a
frozen ``gt_index.csv`` is provided it overrides any mutable DB identity field.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.project_paths import resolve_project_path  # noqa: E402
from utils.scene_geometry_db import (  # noqa: E402
    load_import_audit_metadata,
    load_scene_overlap_pairs,
)


_ARTIFACT = Path(__file__).resolve().parents[2]
FAB2_DEFAULT_RESULTS = _ARTIFACT / "results"

CANONICAL_PREDICTION_FIELDS = [
    "dataset",
    "site",
    "recording",
    "split",
    "scene_id",
    "batch_id",
    "timestamp",
    "frame_index",
    "camera_id",
    "method",
    "evaluation_regime",
    "source_detection_id",
    "event_key",
    "local_track_id",
    "global_track_id",
    "class_id",
    "class_name",
    "confidence",
    "world_x",
    "world_y",
    "gt_world_x",
    "gt_world_y",
    "matched_gt_id",
    "camera_distance_m",
    "spatial_slice",
    "runtime_ms",
]

GT_INDEX_FIELDS = [
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

RUN_METADATA_FIELDS = [
    "tool",
    "created_at",
    "cwd",
    "git_revision",
    "git_dirty",
    "python",
    "argv",
]


@dataclass(frozen=True)
class DatasetLabels:
    dataset: str
    site: str
    recording: str


@dataclass(frozen=True)
class SliceRule:
    scene_id: str
    camera_id: str
    spatial_slice: str
    timestamp_min: float | None = None
    timestamp_max: float | None = None
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_path(path: str | Path) -> Path:
    return resolve_project_path(path)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def connect_db(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(resolve_path(db_path)))
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not table_exists(conn, table):
        return set()
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def require_columns(conn: sqlite3.Connection, table: str, cols: Iterable[str]) -> None:
    existing = table_columns(conn, table)
    missing = [c for c in cols if c not in existing]
    if missing:
        raise RuntimeError(f"{table} missing required columns: {', '.join(missing)}")


def split_csv_arg(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def selected_scenes(conn: sqlite3.Connection, batch_id: str, scene_arg: str | None) -> list[str]:
    explicit = split_csv_arg(scene_arg)
    if explicit:
        return explicit
    rows = conn.execute(
        "SELECT DISTINCT scene_id FROM trajectory_observations "
        "WHERE batch_id=? ORDER BY scene_id",
        (batch_id,),
    ).fetchall()
    return [str(r[0]) for r in rows]


def scene_camera_names(conn: sqlite3.Connection, scene_id: str, batch_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT camera_name FROM trajectory_observations "
        "WHERE scene_id=? AND batch_id=? ORDER BY camera_name",
        (scene_id, batch_id),
    ).fetchall()
    return [str(r[0]) for r in rows]


def infer_dataset_labels(scene_id: str, dataset: str | None = None) -> DatasetLabels:
    sid = str(scene_id)
    low = sid.lower()
    if dataset:
        ds = dataset
    elif low.startswith("synth_") or low.startswith("synthehicle"):
        ds = "Synthehicle Core"
    elif low.startswith("lumpi"):
        ds = "LUMPI"
    elif low.startswith("xizi"):
        ds = "Xizi"
    elif low.startswith("a9"):
        ds = "A9"
    else:
        ds = "unknown"
    return DatasetLabels(dataset=ds, site=sid, recording=sid)


def source_detection_id(
    scene_id: str,
    batch_id: str,
    camera_id: str,
    local_track_id: int | str,
    frame_index: int | str,
) -> str:
    return f"{scene_id}|{batch_id}|{camera_id}|{local_track_id}|{frame_index}"


def event_key(
    scene_id: str,
    camera_id: str,
    local_track_id: int | str,
    frame_index: int | str,
) -> str:
    return f"{scene_id}|{camera_id}|{local_track_id}|{frame_index}"


def time_key(timestamp: float, quantum: float = 0.05) -> int:
    if quantum <= 0:
        quantum = 0.05
    return int(round(float(timestamp) / quantum))


def finite_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def finite_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> Path:
    out = resolve_path(path)
    ensure_dir(out.parent)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k, "")) for k in fields})
    return out


def append_csv(path: str | Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> Path:
    out = resolve_path(path)
    ensure_dir(out.parent)
    exists = out.is_file()
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k, "")) for k in fields})
    return out


def upsert_csv_rows(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fields: Sequence[str],
    *,
    key_fields: Sequence[str],
) -> Path:
    """Merge metric rows into a CSV, replacing rows with the same logical key."""

    out = resolve_path(path)
    ensure_dir(out.parent)
    new_rows = list(rows)
    new_keys = {
        tuple(str(row.get(k, "")) for k in key_fields)
        for row in new_rows
    }
    merged: list[dict[str, Any]] = []
    if out.is_file():
        for row in read_csv_rows(out):
            key = tuple(str(row.get(k, "")) for k in key_fields)
            if key not in new_keys:
                merged.append(row)
    merged.extend(new_rows)
    return write_csv(out, merged, fields)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    p = resolve_path(path)
    with p.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_json(path: str | Path, payload: Any) -> Path:
    out = resolve_path(path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return out


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.10g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def git_revision() -> tuple[str, bool]:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_ROOT),
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(_ROOT),
                text=True,
                capture_output=True,
                check=False,
            ).stdout.strip()
        )
        return rev or "unknown", dirty
    except Exception:
        return "unknown", False


def file_sha256(path: str | Path) -> str:
    p = resolve_path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_metadata(out_dir: str | Path, tool: str, args: argparse.Namespace, extra: dict[str, Any] | None = None) -> Path:
    rev, dirty = git_revision()
    payload: dict[str, Any] = {
        "tool": tool,
        "created_at": utc_now(),
        "cwd": str(Path.cwd()),
        "git_revision": rev,
        "git_dirty": dirty,
        "python": sys.version.replace("\n", " "),
        "argv": sys.argv,
        "args": vars(args),
    }
    if extra:
        payload.update(extra)
    safe_tool = tool.replace(os.sep, "_").replace("/", "_")
    return write_json(Path(out_dir) / f"{safe_tool}_run_metadata.json", payload)


def stats_from_errors(errors: Sequence[float]) -> dict[str, float]:
    arr = np.asarray([e for e in errors if math.isfinite(float(e))], dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "rmse_m": float("nan"),
            "mae_m": float("nan"),
            "median_m": float("nan"),
            "p90_m": float("nan"),
            "p95_m": float("nan"),
            "max_m": float("nan"),
        }
    return {
        "n": int(arr.size),
        "rmse_m": float(np.sqrt(np.mean(arr * arr))),
        "mae_m": float(np.mean(arr)),
        "median_m": float(np.percentile(arr, 50)),
        "p90_m": float(np.percentile(arr, 90)),
        "p95_m": float(np.percentile(arr, 95)),
        "max_m": float(np.max(arr)),
    }


def paired_bootstrap_delta(
    base_errors: dict[str, float],
    test_errors: dict[str, float],
    *,
    seed: int = 20260729,
    samples: int = 1000,
) -> dict[str, float]:
    keys = sorted(set(base_errors) & set(test_errors))
    if not keys:
        return {"paired_n": 0, "delta_rmse_m": float("nan"), "ci95_low_m": float("nan"), "ci95_high_m": float("nan")}
    base = np.asarray([base_errors[k] for k in keys], dtype=np.float64)
    test = np.asarray([test_errors[k] for k in keys], dtype=np.float64)
    delta = float(np.sqrt(np.mean(test * test)) - np.sqrt(np.mean(base * base)))
    rng = np.random.default_rng(seed)
    boots = []
    n = len(keys)
    for _ in range(max(1, samples)):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.sqrt(np.mean(test[idx] * test[idx])) - np.sqrt(np.mean(base[idx] * base[idx]))))
    return {
        "paired_n": n,
        "delta_rmse_m": delta,
        "ci95_low_m": float(np.percentile(boots, 2.5)),
        "ci95_high_m": float(np.percentile(boots, 97.5)),
    }


def load_slice_rules(path: str | Path | None) -> list[SliceRule]:
    if not path:
        return []
    rules: list[SliceRule] = []
    for row in read_csv_rows(path):
        rules.append(
            SliceRule(
                scene_id=row.get("scene_id", ""),
                camera_id=row.get("camera_id", row.get("camera", "")),
                spatial_slice=row.get("spatial_slice", row.get("slice", "overall")) or "overall",
                timestamp_min=finite_float(row.get("timestamp_min")),
                timestamp_max=finite_float(row.get("timestamp_max")),
                x_min=finite_float(row.get("x_min")),
                x_max=finite_float(row.get("x_max")),
                y_min=finite_float(row.get("y_min")),
                y_max=finite_float(row.get("y_max")),
            )
        )
    return rules


def rule_slice_for(row: dict[str, Any], rules: Sequence[SliceRule]) -> str | None:
    if not rules:
        return None
    x = finite_float(row.get("gt_world_x"), finite_float(row.get("world_x")))
    y = finite_float(row.get("gt_world_y"), finite_float(row.get("world_y")))
    t = finite_float(row.get("timestamp"))
    scene_id = str(row.get("scene_id", ""))
    camera_id = str(row.get("camera_id", ""))
    for rule in rules:
        if rule.scene_id and rule.scene_id != scene_id:
            continue
        if rule.camera_id and rule.camera_id != camera_id:
            continue
        if rule.timestamp_min is not None and (t is None or t < rule.timestamp_min):
            continue
        if rule.timestamp_max is not None and (t is None or t > rule.timestamp_max):
            continue
        if rule.x_min is not None and (x is None or x < rule.x_min):
            continue
        if rule.x_max is not None and (x is None or x > rule.x_max):
            continue
        if rule.y_min is not None and (y is None or y < rule.y_min):
            continue
        if rule.y_max is not None and (y is None or y > rule.y_max):
            continue
        return rule.spatial_slice
    return None


def assign_spatial_slices(
    rows: list[dict[str, Any]],
    *,
    slice_rules: Sequence[SliceRule] | None = None,
    handoff_window_sec: float = 1.0,
    time_quantum_sec: float = 0.05,
    infer_from_tracks: bool = False,
) -> list[dict[str, Any]]:
    rules = list(slice_rules or [])
    for row in rows:
        explicit = rule_slice_for(row, rules)
        if explicit:
            row["spatial_slice"] = explicit

    unresolved = [r for r in rows if not str(r.get("spatial_slice", "")).strip()]
    if not infer_from_tracks:
        for row in unresolved:
            row["spatial_slice"] = "unregistered"
        return rows

    overlap_keys: set[tuple[str, str, int]] = set()
    by_gt_time: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in unresolved:
        gt = str(row.get("matched_gt_id", ""))
        if not gt:
            continue
        key = (str(row.get("scene_id", "")), gt, time_key(float(row.get("timestamp", 0.0)), time_quantum_sec))
        by_gt_time[key].add(str(row.get("camera_id", "")))
    for key, cams in by_gt_time.items():
        if len(cams) >= 2:
            overlap_keys.add(key)

    by_gt: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unresolved:
        gt = str(row.get("matched_gt_id", ""))
        if gt:
            by_gt[(str(row.get("scene_id", "")), gt)].append(row)

    for row in unresolved:
        gt = str(row.get("matched_gt_id", ""))
        if not gt:
            row["spatial_slice"] = "overall"
            continue
        scene = str(row.get("scene_id", ""))
        tk = time_key(float(row.get("timestamp", 0.0)), time_quantum_sec)
        if (scene, gt, tk) in overlap_keys:
            row["spatial_slice"] = "Overlap"
            continue
        cam = str(row.get("camera_id", ""))
        t = float(row.get("timestamp", 0.0))
        handoff = any(
            str(other.get("camera_id", "")) != cam
            and abs(float(other.get("timestamp", 0.0)) - t) <= handoff_window_sec
            for other in by_gt.get((scene, gt), [])
        )
        row["spatial_slice"] = "Handoff" if handoff else "Non-overlap"
    return rows


def load_gt_index(path: str | Path | None) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    by_source: dict[str, dict[str, str]] = {}
    by_event: dict[str, dict[str, str]] = {}
    if not path:
        return by_source, by_event
    for row in read_csv_rows(path):
        sid = row.get("source_detection_id", "")
        ekey = row.get("event_key", "")
        if sid:
            by_source[sid] = row
        if ekey:
            by_event[ekey] = row
    return by_source, by_event


def load_db_prediction_rows(
    conn: sqlite3.Connection,
    *,
    scenes: Sequence[str],
    batch_id: str,
    method: str,
    evaluation_regime: str,
    world_coords: str,
    dataset: str | None = None,
    gt_index_by_source: dict[str, dict[str, str]] | None = None,
    gt_index_by_event: dict[str, dict[str, str]] | None = None,
    require_gt: bool = True,
    class_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    require_columns(
        conn,
        "trajectory_observations",
        [
            "scene_id",
            "batch_id",
            "camera_name",
            "local_track_id",
            "global_id",
            "frame_index",
            "timestamp_sec",
            "world_x",
            "world_y",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
        ],
    )
    cols = table_columns(conn, "trajectory_observations")
    have_final = {"final_world_x", "final_world_y"}.issubset(cols)
    have_gt = {"outside_reference_x", "outside_reference_y"}.issubset(cols)
    camera_distance_expr = "camera_distance_m" if "camera_distance_m" in cols else "NULL AS camera_distance_m"
    if world_coords == "final" and not have_final:
        raise RuntimeError("world_coords=final requires final_world_x/final_world_y")
    if require_gt and not have_gt and not gt_index_by_source and not gt_index_by_event:
        raise RuntimeError("GT required but outside_reference_x/y and gt_index are unavailable")

    rows_out: list[dict[str, Any]] = []
    gt_source = gt_index_by_source or {}
    gt_event = gt_index_by_event or {}

    for scene_id in scenes:
        labels = infer_dataset_labels(scene_id, dataset)
        rows = conn.execute(
            f"""
            SELECT camera_name, local_track_id, global_id, frame_index,
                   timestamp_sec, world_x, world_y, final_world_x, final_world_y,
                   outside_reference_x, outside_reference_y,
                   bbox_x, bbox_y, bbox_w, bbox_h,
                   confidence, class_id, class_name, {camera_distance_expr}
              FROM trajectory_observations
             WHERE scene_id=? AND batch_id=?
             ORDER BY timestamp_sec, frame_index, camera_name, local_track_id
            """,
            (scene_id, batch_id),
        ).fetchall()
        for r in rows:
            cls = finite_int(r["class_id"])
            if class_ids is not None and cls not in class_ids:
                continue
            sid = source_detection_id(scene_id, batch_id, r["camera_name"], r["local_track_id"], r["frame_index"])
            ekey = event_key(scene_id, r["camera_name"], r["local_track_id"], r["frame_index"])
            if world_coords == "p_geo":
                px, py = finite_float(r["world_x"]), finite_float(r["world_y"])
            elif world_coords == "final":
                px, py = finite_float(r["final_world_x"]), finite_float(r["final_world_y"])
                if px is None or py is None:
                    continue
            elif world_coords == "auto":
                px = finite_float(r["final_world_x"]) if have_final else None
                py = finite_float(r["final_world_y"]) if have_final else None
                if px is None or py is None:
                    px, py = finite_float(r["world_x"]), finite_float(r["world_y"])
            else:
                raise ValueError(f"unknown world coordinate mode: {world_coords}")
            if px is None or py is None:
                continue

            gt_row = gt_source.get(sid) or gt_event.get(ekey)
            if gt_row:
                gt_id = gt_row.get("gt_global_id") or gt_row.get("matched_gt_id") or ""
                gx = finite_float(gt_row.get("gt_world_x"))
                gy = finite_float(gt_row.get("gt_world_y"))
                split = gt_row.get("split", "")
            else:
                gt_id = str(r["global_id"]) if r["global_id"] is not None else ""
                gx = finite_float(r["outside_reference_x"]) if have_gt else None
                gy = finite_float(r["outside_reference_y"]) if have_gt else None
                split = ""
            if require_gt and (gx is None or gy is None or not gt_id):
                continue
            db_camera_distance = finite_float(r["camera_distance_m"])
            rows_out.append(
                {
                    "dataset": labels.dataset,
                    "site": labels.site,
                    "recording": labels.recording,
                    "scene_id": scene_id,
                    "batch_id": batch_id,
                    "timestamp": float(r["timestamp_sec"]),
                    "frame_index": int(r["frame_index"]),
                    "camera_id": str(r["camera_name"]),
                    "method": method,
                    "evaluation_regime": evaluation_regime,
                    "source_detection_id": sid,
                    "event_key": ekey,
                    "local_track_id": int(r["local_track_id"]),
                    "global_track_id": int(r["global_id"]) if r["global_id"] is not None else "",
                    "class_id": cls if cls is not None else "",
                    "class_name": r["class_name"] or "",
                    "confidence": finite_float(r["confidence"]),
                    "world_x": px,
                    "world_y": py,
                    "gt_world_x": gx,
                    "gt_world_y": gy,
                    "matched_gt_id": gt_id,
                    "camera_distance_m": gt_row.get("camera_distance_m", "") if gt_row else (db_camera_distance if db_camera_distance is not None else ""),
                    "spatial_slice": gt_row.get("spatial_slice", "") if gt_row else "",
                    "runtime_ms": "",
                    "split": split,
                }
            )
    return rows_out


def load_prediction_csv(path: str | Path, *, require_gt: bool = True) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        px, py = finite_float(row.get("world_x")), finite_float(row.get("world_y"))
        gx, gy = finite_float(row.get("gt_world_x")), finite_float(row.get("gt_world_y"))
        if px is None or py is None:
            continue
        if require_gt and (gx is None or gy is None):
            continue
        row = dict(row)
        row["world_x"], row["world_y"] = px, py
        row["gt_world_x"], row["gt_world_y"] = gx, gy
        row["timestamp"] = finite_float(row.get("timestamp"), 0.0) or 0.0
        out.append(row)
    return out


def group_errors(rows: Iterable[dict[str, Any]], group_fields: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in rows:
        px, py = finite_float(row.get("world_x")), finite_float(row.get("world_y"))
        gx, gy = finite_float(row.get("gt_world_x")), finite_float(row.get("gt_world_y"))
        if None in (px, py, gx, gy):
            continue
        key = tuple(row.get(f, "") for f in group_fields)
        groups[key].append(math.hypot(float(px) - float(gx), float(py) - float(gy)))
        counts[key] += 1
    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        result = {field: key[i] for i, field in enumerate(group_fields)}
        result.update(stats_from_errors(groups[key]))
        result["matched_count"] = counts[key]
        out.append(result)
    return out


def distance_bin(row: dict[str, Any]) -> str:
    camera_distance = finite_float(row.get("camera_distance_m"))
    if camera_distance is not None:
        d = camera_distance
    else:
        # Distance strata are defined relative to the target camera, not the
        # arbitrary BEV world origin. Provide camera_distance_m in the frozen
        # manifest/predictions; otherwise keep the bin explicitly unknown.
        return "unknown"
    if d < 15.0:
        return "0-15m"
    if d < 30.0:
        return "15-30m"
    return ">=30m"


def command_to_str(cmd: Sequence[str | Path]) -> str:
    return " ".join(str(c) for c in cmd)


def run_command(cmd: Sequence[str | Path], *, dry_run: bool = False, cwd: Path | None = None) -> int:
    print(command_to_str(cmd))
    if dry_run:
        return 0
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd or _ROOT), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def latest_calibration_rows(conn: sqlite3.Connection, scene_id: str) -> list[sqlite3.Row]:
    if not table_exists(conn, "camera_calibrations"):
        return []
    rows = conn.execute(
        """
        SELECT camera_name, scene_id, scale_px_per_meter, source, point_count,
               inlier_count, reproj_rms, geometry_version, calibrated_at
          FROM camera_calibrations
         WHERE scene_id=?
         ORDER BY camera_name, calibrated_at DESC
        """,
        (scene_id,),
    ).fetchall()
    out: list[sqlite3.Row] = []
    seen: set[str] = set()
    for row in rows:
        cam = str(row["camera_name"])
        if cam in seen:
            continue
        seen.add(cam)
        out.append(row)
    return out


def checkpoint_meta(path: str | Path) -> dict[str, Any]:
    p = resolve_path(path)
    if not p.is_file():
        return {"path": str(p), "exists": False}
    meta: dict[str, Any] = {"path": str(p), "exists": True, "sha256": file_sha256(p), "bytes": p.stat().st_size}
    try:
        import torch

        blob = torch.load(str(p), map_location="cpu")
        if isinstance(blob, dict) and isinstance(blob.get("meta"), dict):
            meta.update(blob["meta"])
    except Exception as exc:
        meta["meta_error"] = str(exc)
    return meta


def scene_observation_count(conn: sqlite3.Connection, scene_id: str, batch_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
        (scene_id, batch_id),
    ).fetchone()
    return int(row[0]) if row else 0


def iter_scene_times(conn: sqlite3.Connection, scene_id: str, batch_id: str) -> Iterator[float]:
    rows = conn.execute(
        "SELECT timestamp_sec FROM trajectory_observations "
        "WHERE scene_id=? AND batch_id=? ORDER BY timestamp_sec",
        (scene_id, batch_id),
    )
    for row in rows:
        yield float(row[0])
