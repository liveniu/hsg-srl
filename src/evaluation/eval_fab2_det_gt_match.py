#!/usr/bin/env python3
"""Attach detector tracks to ground-truth identities.

The filename keeps the internal tag fab2. The attachment gates are documented
in docs/REPRODUCTION.md. This is not a separate method.

Associate detector tracks with ground-truth identities for the detector track.

The GT-import track needs no association: every observation *is* a GT box, so
``outside_reference_x/y`` and the GT identity come for free. Feeding a detector
(tight boxes, own track ids, own clock) into the same evaluation requires the
association to be built first, and three properties of the data make a naive
per-frame nearest-neighbour match unusable:

1. Clock offset. Video PTS and LiDAR label timestamps do not share an origin.
   A constant per-camera offset ``dt`` is estimated by grid search, minimising
   the median baseline residual. Estimating ``dt`` on the *baseline* geometry is
   deliberately conservative: it removes the part of the residual a clock shift
   could explain, so no later improvement can come from time misalignment.

2. Sampling rate mismatch. GT is 10 Hz, detections 12.5 Hz. Nearest-in-time
   matching leaves up to 50 ms of lag, i.e. ~0.75 m at 54 km/h, which is larger
   than the effect under test. GT positions are therefore linearly interpolated
   to the detection timestamp.

3. Track fragmentation. A detector track may cover part of a GT trajectory, and
   one GT identity may be split across several detector tracks. Association is
   therefore many-to-one (detector track -> GT id) and decided at *track* level
   on the median residual, with a margin over the runner-up. Once identity is
   settled, every frame of that track is kept, including its large residuals:
   censoring per-frame outliers would quietly remove exactly the errors the
   method is supposed to correct.

Outputs ``det_gt_pairs.csv`` (the frozen association, consumed by
``fab2_manifest.py --gt-id-csv``) plus a report with the estimated offsets and
match rates. With ``--write-outside-reference`` the matched GT position is also
written into the detection DB so the standard manifest builder picks it up.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.fab2_common import (  # noqa: E402
    FAB2_DEFAULT_RESULTS,
    connect_db,
    ensure_dir,
    event_key,
    source_detection_id,
    split_csv_arg,
    write_csv,
    write_json,
    write_run_metadata,
)

PAIR_FIELDS = [
    "scene_id",
    "batch_id",
    "camera_id",
    "local_track_id",
    "frame_index",
    "timestamp",
    "source_detection_id",
    "event_key",
    "gt_global_id",
    "gt_world_x",
    "gt_world_y",
    "gt_image_x",
    "gt_image_y",
    "dt_applied_sec",
    "residual_m",
]


class GtTable:
    """GT samples of one camera, interpolatable at arbitrary times."""

    def __init__(
        self,
        rows: list[tuple[float, int, float, float, float | None, float | None]],
    ) -> None:
        # (t, x, y, iu, iv); iu/iv may be NaN when the GT-import row lacks pinhole pixels.
        by_id: dict[int, list[tuple[float, float, float, float, float]]] = defaultdict(list)
        for t, gid, x, y, iu, iv in rows:
            by_id[gid].append((
                t, x, y,
                float("nan") if iu is None else float(iu),
                float("nan") if iv is None else float(iv),
            ))
        self.ids: list[int] = []
        self.t: dict[int, np.ndarray] = {}
        self.x: dict[int, np.ndarray] = {}
        self.y: dict[int, np.ndarray] = {}
        self.iu: dict[int, np.ndarray] = {}
        self.iv: dict[int, np.ndarray] = {}
        for gid, samples in by_id.items():
            samples.sort()
            arr = np.asarray(samples, dtype=float)
            # Duplicate timestamps break the monotonic assumption of np.interp.
            keep = np.concatenate(([True], np.diff(arr[:, 0]) > 1e-9))
            arr = arr[keep]
            if len(arr) < 2:
                continue
            self.ids.append(gid)
            self.t[gid] = arr[:, 0]
            self.x[gid] = arr[:, 1]
            self.y[gid] = arr[:, 2]
            self.iu[gid] = arr[:, 3]
            self.iv[gid] = arr[:, 4]
        self.ids.sort()
        # Candidate lookup runs once per (offset, track, id); keep spans as arrays
        # so the temporal pre-filter stays vectorised.
        self._span_ids = np.asarray(self.ids, dtype=np.int64)
        self._span_lo = np.asarray([self.t[g][0] for g in self.ids], dtype=float)
        self._span_hi = np.asarray([self.t[g][-1] for g in self.ids], dtype=float)
        self._box = np.asarray(
            [[self.x[g].min(), self.x[g].max(), self.y[g].min(), self.y[g].max()] for g in self.ids],
            dtype=float,
        ) if self.ids else np.empty((0, 4))

    def at(self, gid: int, times: np.ndarray, max_gap: float
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate GT id ``gid`` at ``times``; invalid where GT is too far away in time.

        Returns ``(x, y, ok, image_u, image_v)``. Image channels are NaN where the
        source GT row had no pinhole projection (``watch`` falls back to H then).
        """
        tt = self.t[gid]
        idx = np.searchsorted(tt, times)
        prev = np.clip(idx - 1, 0, len(tt) - 1)
        nxt = np.clip(idx, 0, len(tt) - 1)
        gap = np.minimum(np.abs(times - tt[prev]), np.abs(tt[nxt] - times))
        ok = (times >= tt[0] - max_gap) & (times <= tt[-1] + max_gap) & (gap <= max_gap)
        return (
            np.interp(times, tt, self.x[gid]),
            np.interp(times, tt, self.y[gid]),
            ok,
            np.interp(times, tt, self.iu[gid]),
            np.interp(times, tt, self.iv[gid]),
        )

    def overlapping(self, lo: float, hi: float,
                    box: tuple[float, float, float, float] | None = None,
                    pad_m: float = 0.0) -> list[int]:
        mask = (self._span_hi >= lo) & (self._span_lo <= hi)
        if box is not None and len(self._box):
            x0, x1, y0, y1 = box
            mask &= (
                (self._box[:, 1] >= x0 - pad_m) & (self._box[:, 0] <= x1 + pad_m)
                & (self._box[:, 3] >= y0 - pad_m) & (self._box[:, 2] <= y1 + pad_m)
            )
        return self._span_ids[mask].tolist()


def _load_gt(conn: sqlite3.Connection, scene: str, batch: str, camera: str,
             class_ids: set[int] | None) -> GtTable:
    sql = (
        "SELECT timestamp_sec, global_id, outside_reference_x, outside_reference_y,"
        "       outside_reference_image_x, outside_reference_image_y, class_id"
        " FROM trajectory_observations"
        " WHERE scene_id=? AND batch_id=? AND camera_name=?"
        "   AND outside_reference_x IS NOT NULL AND outside_reference_y IS NOT NULL"
    )
    rows: list[tuple[float, int, float, float, float | None, float | None]] = []
    for r in conn.execute(sql, (scene, batch, camera)):
        if class_ids is not None and r["class_id"] is not None and int(r["class_id"]) not in class_ids:
            continue
        if r["global_id"] is None:
            continue
        iu = r["outside_reference_image_x"]
        iv = r["outside_reference_image_y"]
        rows.append((
            float(r["timestamp_sec"]), int(r["global_id"]),
            float(r["outside_reference_x"]), float(r["outside_reference_y"]),
            None if iu is None else float(iu),
            None if iv is None else float(iv),
        ))
    return GtTable(rows)


def _load_det(conn: sqlite3.Connection, scene: str, batch: str, camera: str) -> dict[int, dict[str, np.ndarray]]:
    rows = conn.execute(
        "SELECT local_track_id, frame_index, timestamp_sec, world_x, world_y"
        " FROM trajectory_observations"
        " WHERE scene_id=? AND batch_id=? AND camera_name=?"
        "   AND world_x IS NOT NULL AND world_y IS NOT NULL"
        " ORDER BY local_track_id, timestamp_sec",
        (scene, batch, camera),
    ).fetchall()
    grouped: dict[int, list[tuple[float, int, float, float]]] = defaultdict(list)
    for r in rows:
        grouped[int(r["local_track_id"])].append(
            (float(r["timestamp_sec"]), int(r["frame_index"]), float(r["world_x"]), float(r["world_y"]))
        )
    out: dict[int, dict[str, np.ndarray]] = {}
    for tid, samples in grouped.items():
        arr = np.asarray(samples, dtype=float)
        out[tid] = {
            "t": arr[:, 0],
            "frame": arr[:, 1].astype(int),
            "x": arr[:, 2],
            "y": arr[:, 3],
        }
    return out


class TimeMap:
    """Video-to-label clock map: label_time = video_time + dt(video_time).

    The constant part handles a start-time offset; the linear part handles fps
    metadata error (e.g. a nominal 25 fps stream that actually runs at 25.012),
    which accumulates over the session and cannot be absorbed by any constant.
    """

    def __init__(self, dt0: float, a: float = 0.0, b: float = 0.0) -> None:
        self.dt0 = float(dt0)
        self.a = float(a)
        self.b = float(b)

    def offsets(self, times: np.ndarray) -> np.ndarray:
        return self.dt0 + self.a + self.b * np.asarray(times, dtype=float)

    def label_times(self, times: np.ndarray) -> np.ndarray:
        return np.asarray(times, dtype=float) + self.offsets(times)


def _candidate_gt_ids(gt: GtTable, track: dict[str, np.ndarray], tm: TimeMap, pad: float,
                      gate: float) -> list[int]:
    box = (float(track["x"].min()), float(track["x"].max()),
           float(track["y"].min()), float(track["y"].max()))
    lt = tm.label_times(track["t"])
    return gt.overlapping(float(lt.min()) - pad, float(lt.max()) + pad,
                          box=box, pad_m=gate)


def _track_residuals(track: dict[str, np.ndarray], gt: GtTable, gid: int, tm: TimeMap,
                     max_gap: float) -> np.ndarray:
    times = tm.label_times(track["t"])
    gx, gy, ok, _iu, _iv = gt.at(gid, times, max_gap)
    if not np.any(ok):
        return np.empty(0)
    return np.hypot(track["x"][ok] - gx[ok], track["y"][ok] - gy[ok])


def _fit_time_drift(det: dict[int, dict[str, np.ndarray]], gt: GtTable,
                    assigned: dict[int, tuple[int, float, int]], tm: TimeMap, *,
                    max_gap: float, min_speed: float = 2.0,
                    vel_half_step: float = 0.2) -> tuple[float, float, int]:
    """Fit residual drift tau(t) = a + b*t from moving vehicles.

    For a clock error tau, the residual satisfies r ~= v_gt * tau + bias, where
    the (anchor) bias is roughly velocity-independent. Projecting r onto v_gt
    and solving a weighted least squares in t therefore isolates the temporal
    part; parked vehicles carry no time information and are excluded.
    """
    ts: list[np.ndarray] = []
    num: list[np.ndarray] = []
    den: list[np.ndarray] = []
    for tid, (gid, _med, _n) in assigned.items():
        track = det[tid]
        times = tm.label_times(track["t"])
        gx, gy, ok, _, _ = gt.at(gid, times, max_gap)
        gx2, gy2, ok2, _, _ = gt.at(gid, times + vel_half_step, max_gap)
        gx1, gy1, ok1, _, _ = gt.at(gid, times - vel_half_step, max_gap)
        vx = (gx2 - gx1) / (2.0 * vel_half_step)
        vy = (gy2 - gy1) / (2.0 * vel_half_step)
        sp2 = vx ** 2 + vy ** 2
        mask = ok & ok1 & ok2 & (sp2 >= min_speed ** 2)
        if not np.any(mask):
            continue
        rx = track["x"] - gx
        ry = track["y"] - gy
        ts.append(track["t"][mask])
        num.append((rx * vx + ry * vy)[mask])
        den.append(sp2[mask])
    if not ts:
        return 0.0, 0.0, 0
    t = np.concatenate(ts)
    w = np.concatenate(den)                  # weight |v|^2 favours informative samples
    tau = np.concatenate(num) / w            # per-sample tau estimate
    if len(t) < 500:
        return float(np.average(tau, weights=w)), 0.0, len(t)
    W = w.sum()
    Wt = (w * t).sum()
    Wtt = (w * t * t).sum()
    Wy = (w * tau).sum()
    Wty = (w * t * tau).sum()
    d = W * Wtt - Wt * Wt
    if abs(d) < 1e-9:
        return float(Wy / W), 0.0, len(t)
    a = (Wtt * Wy - Wt * Wty) / d
    b = (W * Wty - Wt * Wy) / d
    return float(a), float(b), len(t)


def _estimate_dt(det: dict[int, dict[str, np.ndarray]], gt: GtTable, *,
                 dt_min: float, dt_max: float, dt_step: float, max_gap: float,
                 gate: float, min_overlap: int, sample_tracks: int) -> tuple[float, list[dict[str, Any]]]:
    """Pick the offset minimising the median best-match residual over sampled tracks."""
    tracks = [t for t in det.values() if len(t["t"]) >= min_overlap]
    tracks.sort(key=lambda t: -len(t["t"]))
    tracks = tracks[:sample_tracks]
    grid = np.arange(dt_min, dt_max + 0.5 * dt_step, dt_step)
    curve: list[dict[str, Any]] = []
    for dt in grid:
        tm = TimeMap(float(dt))
        per_track: list[float] = []
        for track in tracks:
            cands = _candidate_gt_ids(gt, track, tm, max_gap, gate)
            best = math.inf
            for gid in cands:
                res = _track_residuals(track, gt, gid, tm, max_gap)
                if len(res) >= min_overlap:
                    best = min(best, float(np.median(res)))
            if best <= gate:
                per_track.append(best)
        med = float(np.median(per_track)) if per_track else math.inf
        curve.append({"dt_sec": round(float(dt), 4), "matched_tracks": len(per_track),
                      "median_residual_m": None if math.isinf(med) else round(med, 4)})

    # Minimise the residual, but only among offsets that still explain most of
    # the traffic: a shift that matches a handful of tracks very well is not a
    # better clock estimate, it is a smaller sample.
    max_matched = max((c["matched_tracks"] for c in curve), default=0)
    if not max_matched:
        return 0.0, curve
    eligible = [c for c in curve
                if c["matched_tracks"] >= 0.8 * max_matched and c["median_residual_m"] is not None]
    if not eligible:
        return 0.0, curve
    return float(min(eligible, key=lambda c: c["median_residual_m"])["dt_sec"]), curve


def _assign_tracks(det: dict[int, dict[str, np.ndarray]], gt: GtTable, tm: TimeMap, *,
                   max_gap: float, gate: float, min_overlap: int,
                   margin_m: float) -> dict[int, tuple[int, float, int]]:
    """Map detector track -> (gt id, median residual, overlap), many-to-one."""
    assigned: dict[int, tuple[int, float, int]] = {}
    for tid, track in det.items():
        if len(track["t"]) < min_overlap:
            continue
        scored: list[tuple[float, int, int]] = []
        for gid in _candidate_gt_ids(gt, track, tm, max_gap, gate):
            res = _track_residuals(track, gt, gid, tm, max_gap)
            if len(res) >= min_overlap:
                scored.append((float(np.median(res)), gid, len(res)))
        if not scored:
            continue
        scored.sort()
        best_med, best_gid, best_n = scored[0]
        if best_med > gate:
            continue
        if len(scored) > 1 and scored[1][0] - best_med < margin_m:
            continue  # ambiguous: two GT identities explain this track equally well
        assigned[tid] = (best_gid, best_med, best_n)
    return assigned


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {}
    return {
        "median_m": round(float(np.median(values)), 4),
        "p90_m": round(float(np.percentile(values, 90)), 4),
        "p95_m": round(float(np.percentile(values, 95)), 4),
        "max_m": round(float(values.max()), 4),
        "rmse_m": round(float(np.sqrt(np.mean(values ** 2))), 4),
    }


def _write_outside_reference(db: str, scene: str, batch: str,
                             pairs: list[dict[str, Any]]) -> int:
    conn = sqlite3.connect(str(Path(db)))
    try:
        conn.execute(
            "UPDATE trajectory_observations"
            " SET outside_reference_x=NULL, outside_reference_y=NULL,"
            "     outside_reference_image_x=NULL, outside_reference_image_y=NULL"
            " WHERE scene_id=? AND batch_id=?",
            (scene, batch),
        )
        conn.executemany(
            "UPDATE trajectory_observations"
            " SET outside_reference_x=?, outside_reference_y=?,"
            "     outside_reference_image_x=?, outside_reference_image_y=?"
            " WHERE scene_id=? AND batch_id=? AND camera_name=? AND local_track_id=? AND frame_index=?",
            [
                (p["gt_world_x"], p["gt_world_y"],
                 p.get("gt_image_x"), p.get("gt_image_y"),
                 scene, batch,
                 p["camera_id"], p["local_track_id"], p["frame_index"])
                for p in pairs
            ],
        )
        conn.commit()
        return conn.total_changes
    finally:
        conn.close()


def run(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(Path(args.out_dir))
    gt_classes = {int(c) for c in split_csv_arg(args.gt_class_ids)} or None
    cameras = split_csv_arg(args.cameras)

    det_conn = connect_db(args.det_db)
    gt_conn = connect_db(args.gt_db)
    try:
        if not cameras:
            cameras = [
                str(r[0]) for r in det_conn.execute(
                    "SELECT DISTINCT camera_name FROM trajectory_observations"
                    " WHERE scene_id=? AND batch_id=? ORDER BY camera_name",
                    (args.det_scene, args.det_batch),
                )
            ]
        pairs: list[dict[str, Any]] = []
        report: dict[str, Any] = {"cameras": {}}
        for camera in cameras:
            gt = _load_gt(gt_conn, args.gt_scene, args.gt_batch, camera, gt_classes)
            det = _load_det(det_conn, args.det_scene, args.det_batch, camera)
            if not gt.ids or not det:
                report["cameras"][camera] = {"error": "no GT or no detections"}
                continue

            if args.dt_sec is None:
                dt, curve = _estimate_dt(
                    det, gt,
                    dt_min=args.dt_min, dt_max=args.dt_max, dt_step=args.dt_step,
                    max_gap=args.max_time_gap, gate=args.dt_gate_m,
                    min_overlap=args.min_overlap, sample_tracks=args.dt_sample_tracks,
                )
            else:
                dt, curve = float(args.dt_sec), []
            tm = TimeMap(dt)

            assigned = _assign_tracks(
                det, gt, tm,
                max_gap=args.max_time_gap, gate=args.gate_m,
                min_overlap=args.min_overlap, margin_m=args.margin_m,
            )

            drift_info: dict[str, Any] = {}
            if args.dt_mode == "affine" and assigned:
                # Two rounds: the drift estimate sharpens once the association
                # itself is computed under the corrected clock.
                for _round in range(2):
                    a, b, n_fit = _fit_time_drift(
                        det, gt, assigned, tm, max_gap=args.max_time_gap)
                    tm = TimeMap(dt, tm.a + a, tm.b + b)
                    assigned = _assign_tracks(
                        det, gt, tm,
                        max_gap=args.max_time_gap, gate=args.gate_m,
                        min_overlap=args.min_overlap, margin_m=args.margin_m,
                    )
                drift_info = {
                    "drift_const_ms": round(1000.0 * tm.a, 1),
                    "drift_slope_ms_per_min": round(60_000.0 * tm.b, 2),
                    "drift_fit_samples": n_fit,
                }
                print(f"[{camera}] affine clock: dt0={dt:+.3f}s a={tm.a*1000:+.0f}ms "
                      f"b={tm.b*60_000:+.2f}ms/min (n={n_fit})")

            residuals: list[float] = []
            dropped_time = 0
            dropped_sanity = 0
            for tid, (gid, _med, _n) in assigned.items():
                track = det[tid]
                dt_v = tm.offsets(track["t"])
                times = track["t"] + dt_v
                gx, gy, ok, giu, giv = gt.at(gid, times, args.max_time_gap)
                res = np.hypot(track["x"] - gx, track["y"] - gy)
                for i in range(len(times)):
                    if not ok[i]:
                        dropped_time += 1
                        continue
                    if res[i] > args.frame_sanity_m:
                        dropped_sanity += 1
                        continue
                    frame = int(track["frame"][i])
                    iu_i = float(giu[i])
                    iv_i = float(giv[i])
                    pairs.append({
                        "scene_id": args.det_scene,
                        "batch_id": args.det_batch,
                        "camera_id": camera,
                        "local_track_id": tid,
                        "frame_index": frame,
                        "timestamp": float(track["t"][i]),
                        "source_detection_id": source_detection_id(
                            args.det_scene, args.det_batch, camera, tid, frame),
                        "event_key": event_key(args.det_scene, camera, tid, frame),
                        "gt_global_id": gid,
                        "gt_world_x": float(gx[i]),
                        "gt_world_y": float(gy[i]),
                        "gt_image_x": None if math.isnan(iu_i) else round(iu_i, 3),
                        "gt_image_y": None if math.isnan(iv_i) else round(iv_i, 3),
                        "dt_applied_sec": round(float(dt_v[i]), 4),
                        "residual_m": round(float(res[i]), 4),
                    })
                    residuals.append(float(res[i]))

            det_rows = sum(len(t["t"]) for t in det.values())
            report["cameras"][camera] = {
                "dt_applied_sec": round(dt, 4),
                "dt_mode": args.dt_mode,
                **drift_info,
                "dt_source": "fixed" if args.dt_sec is not None else "grid_search",
                "dt_curve": curve,
                "gt_ids": len(gt.ids),
                "gt_ids_matched": len({v[0] for v in assigned.values()}),
                "det_tracks": len(det),
                "det_tracks_assigned": len(assigned),
                "det_rows": det_rows,
                "matched_rows": len(residuals),
                "match_rate": round(len(residuals) / det_rows, 4) if det_rows else 0.0,
                "dropped_no_gt_in_time": dropped_time,
                "dropped_sanity_gate": dropped_sanity,
                "baseline_residual": _percentiles(np.asarray(residuals)),
            }
            print(f"[{camera}] dt={dt:+.3f}s  tracks {len(assigned)}/{len(det)}  "
                  f"rows {len(residuals)}/{det_rows}  "
                  f"median={report['cameras'][camera]['baseline_residual'].get('median_m')}m")
    finally:
        det_conn.close()
        gt_conn.close()

    if not pairs:
        raise SystemExit("No detector/GT pairs were established.")

    pairs.sort(key=lambda p: (p["camera_id"], p["local_track_id"], p["frame_index"]))
    pair_path = write_csv(out_dir / "det_gt_pairs.csv", pairs, PAIR_FIELDS)
    all_res = np.asarray([p["residual_m"] for p in pairs])
    report["total_pairs"] = len(pairs)
    report["baseline_residual_overall"] = _percentiles(all_res)
    if args.write_outside_reference:
        changed = _write_outside_reference(args.det_db, args.det_scene, args.det_batch, pairs)
        report["outside_reference_rows_written"] = changed
        print(f"outside_reference written into {args.det_db} ({changed} row updates)")
    report_path = write_json(out_dir / "det_gt_match_report.json", report)
    write_run_metadata(out_dir, "eval_fab2_det_gt_match", args,
                       {"total_pairs": len(pairs), "cameras": cameras})
    print(f"pairs -> {pair_path}")
    print(f"report -> {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det-db", required=True, help="DB holding detector observations")
    ap.add_argument("--det-batch", required=True)
    ap.add_argument("--det-scene", required=True)
    ap.add_argument("--gt-db", required=True, help="DB holding GT-import observations")
    ap.add_argument("--gt-batch", required=True)
    ap.add_argument("--gt-scene", default=None, help="defaults to --det-scene")
    ap.add_argument("--cameras", default="", help="comma separated; default all in det batch")
    ap.add_argument("--gt-class-ids", default="2,5,7",
                    help="restrict GT to these class ids (vehicles), '' for all")
    ap.add_argument("--dt-sec", type=float, default=None,
                    help="fixed video->label offset; omit to grid search")
    ap.add_argument("--dt-mode", choices=("const", "affine"), default="const",
                    help="const: single per-camera offset (historical behaviour); "
                         "affine: additionally fit clock drift dt(t)=dt0+a+b*t from "
                         "moving vehicles (video fps metadata error accumulates over "
                         "long sessions and no constant can absorb it)")
    ap.add_argument("--dt-min", type=float, default=-1.5)
    ap.add_argument("--dt-max", type=float, default=1.5)
    ap.add_argument("--dt-step", type=float, default=0.05)
    ap.add_argument("--dt-gate-m", type=float, default=4.0,
                    help="track counts as matched during offset search below this residual")
    ap.add_argument("--dt-sample-tracks", type=int, default=60,
                    help="longest N detector tracks used for the offset search")
    ap.add_argument("--gate-m", type=float, default=3.0,
                    help="max median track residual for association")
    ap.add_argument("--margin-m", type=float, default=0.75,
                    help="required median gap to the runner-up GT identity")
    ap.add_argument("--min-overlap", type=int, default=8,
                    help="min co-observed samples for a track-level decision")
    ap.add_argument("--max-time-gap", type=float, default=0.12,
                    help="max distance in time to a real GT sample when interpolating")
    ap.add_argument("--frame-sanity-m", type=float, default=10.0,
                    help="drop per-frame pairs beyond this residual (kept generous on purpose)")
    ap.add_argument("--write-outside-reference", action="store_true",
                    help="write matched GT position into the detection DB")
    ap.add_argument("--out-dir", default=str(Path(FAB2_DEFAULT_RESULTS) / "sites" / "lumpi_M1_yolo"))
    args = ap.parse_args()
    if args.gt_scene is None:
        args.gt_scene = args.det_scene
    run(args)


if __name__ == "__main__":
    main()
