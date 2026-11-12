#!/usr/bin/env python3
"""Close the Table S3 detector-row partition from stored clocks.

Unassigned-track rows are counted from tracks absent in a replay of the
stored clock. They are not inferred by subtracting the other three counts.
The replay is a check that the stored assignment counts can be recovered.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np

from common import (
    DET_BATCH,
    GATE_M,
    GT_BATCH,
    GT_CLASSES,
    MARGIN_M,
    MAX_GAP_S,
    MIN_OVERLAP,
    SANITY_M,
    SITES,
    UPSTREAM,
    connect_ro,
    ensure,
    geometry_db,
    label_of,
    match_report,
    origin_db,
    write_json,
)


def load_matcher():
    spec = importlib.util.spec_from_file_location(
        "fab5c_matcher", UPSTREAM / "eval_fab2_det_gt_match.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATCHER = load_matcher()
TimeMap = MATCHER.TimeMap


def load_tracks(conn, site: str, camera: str) -> dict:
    rows = conn.execute(
        """SELECT local_track_id, frame_index, timestamp_sec, world_x, world_y
           FROM trajectory_observations
           WHERE scene_id=? AND batch_id=? AND camera_name=?
             AND world_x IS NOT NULL AND world_y IS NOT NULL
           ORDER BY local_track_id, timestamp_sec""",
        (site, DET_BATCH, camera),
    ).fetchall()
    grouped: dict[int, list] = {}
    for row in rows:
        grouped.setdefault(int(row["local_track_id"]), []).append(row)
    tracks = {}
    for tid, items in grouped.items():
        tracks[tid] = {
            "t": np.asarray([float(item["timestamp_sec"]) for item in items]),
            "frame": np.asarray([int(item["frame_index"]) for item in items]),
            "x": np.asarray([float(item["world_x"]) for item in items]),
            "y": np.asarray([float(item["world_y"]) for item in items]),
        }
    return tracks


def load_gt(conn, site: str, camera: str):
    from clock_check import load_gt as load_gt_rows
    return load_gt_rows(conn, site, camera, None)


def stored_unassigned(conn, site: str, camera: str) -> dict:
    """Rows whose stored match contains no frame of that detector track."""
    rows = conn.execute(
        """SELECT local_track_id, COUNT(*), SUM(outside_reference_x IS NOT NULL)
           FROM trajectory_observations
           WHERE scene_id=? AND batch_id=? AND camera_name=? AND world_x IS NOT NULL
           GROUP BY local_track_id""",
        (site, DET_BATCH, camera),
    ).fetchall()
    detector_rows = sum(int(count) for _tid, count, _matched in rows)
    unassigned_rows = sum(int(count) for _tid, count, matched in rows if not matched)
    unassigned_tracks = sum(1 for _tid, _count, matched in rows if not matched)
    return {
        "detector_rows": detector_rows,
        "detector_tracks": len(rows),
        "unassigned_track_rows": unassigned_rows,
        "unassigned_tracks": unassigned_tracks,
    }


def recount(tracks: dict, gt, assigned: dict, clock) -> dict:
    matched = time_drop = cutoff = 0
    for tid, (gid, _median, _count) in assigned.items():
        track = tracks[tid]
        times = track["t"] + clock.offsets(track["t"])
        gx, gy, ok, _iu, _iv = gt.at(gid, times, MAX_GAP_S)
        residual = np.hypot(track["x"] - gx, track["y"] - gy)
        for index in range(len(track["t"])):
            if not bool(ok[index]):
                time_drop += 1
            elif float(residual[index]) > SANITY_M:
                cutoff += 1
            else:
                matched += 1
    return {"matched": matched, "time": time_drop, "cutoff": cutoff}


def check_site(site: str, report: dict) -> list[dict]:
    det_conn = connect_ro(geometry_db(site))
    gt_conn = connect_ro(origin_db(site))
    rows = []
    try:
        for camera, info in report["cameras"].items():
            if "det_rows" not in info:
                continue
            tracks = load_tracks(det_conn, site, camera)
            gt = load_gt(gt_conn, site, camera)
            stored = stored_unassigned(det_conn, site, camera)
            clock = TimeMap(
                float(info["dt_applied_sec"]),
                float(info.get("drift_const_ms", 0.0)) / 1000.0,
                float(info.get("drift_slope_ms_per_min", 0.0)) / 60_000.0,
            )
            assigned = MATCHER._assign_tracks(
                tracks, gt, clock, max_gap=MAX_GAP_S, gate=GATE_M,
                min_overlap=MIN_OVERLAP, margin_m=MARGIN_M,
            )
            replay = recount(tracks, gt, assigned, clock)
            closed_sum = (
                int(info["matched_rows"]) + int(info["dropped_no_gt_in_time"])
                + int(info["dropped_sanity_gate"]) + stored["unassigned_track_rows"]
            )
            rows.append({
                "site": site,
                "label": label_of(site),
                "camera": camera,
                "stored_detector_rows": int(info["det_rows"]),
                "counted_detector_rows": stored["detector_rows"],
                "stored_matched": int(info["matched_rows"]),
                "stored_time": int(info["dropped_no_gt_in_time"]),
                "stored_cutoff": int(info["dropped_sanity_gate"]),
                "unassigned_track_rows": stored["unassigned_track_rows"],
                "unassigned_tracks": stored["unassigned_tracks"],
                "closed_sum": closed_sum,
                "closed": closed_sum == int(info["det_rows"]) == stored["detector_rows"],
                "replay_matched": replay["matched"],
                "replay_time": replay["time"],
                "replay_cutoff": replay["cutoff"],
                "replay_matches_stored_counts": (
                    replay["matched"] == int(info["matched_rows"])
                    and replay["time"] == int(info["dropped_no_gt_in_time"])
                    and replay["cutoff"] == int(info["dropped_sanity_gate"])
                ),
            })
            print(
                f"[s3] {label_of(site)} {camera} unassigned={stored['unassigned_track_rows']} "
                f"closed={closed_sum == int(info['det_rows'])}",
                flush=True,
            )
    finally:
        det_conn.close()
        gt_conn.close()
    return rows


def write_markdown(rows: list[dict], path) -> None:
    lines = [
        "# Table S3 row closure",
        "",
        "Unassigned-track rows are the detections in tracks that have no stored matched frame. The two dropped columns count only rows inside assigned tracks. This count is not a false-positive label.",
        "",
        "| Recording | Camera | Detector rows | Matched | No GT in time | Cutoff | Unassigned-track rows | Sum | Closed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['camera']} | {row['stored_detector_rows']} | {row['stored_matched']} | "
            f"{row['stored_time']} | {row['stored_cutoff']} | {row['unassigned_track_rows']} | "
            f"{row['closed_sum']} | {row['closed']} |"
        )
    failed = [row for row in rows if not row["closed"]]
    lines += ["", f"Rows checked: {len(rows)}. Rows that do not close: {len(failed)}.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out = ensure()
    rows = []
    for site in SITES:
        report = json.loads(match_report(site).read_text(encoding="utf-8"))
        rows.extend(check_site(site, report))
    write_json(out / "table_s3_closure.json", {"rows": rows})
    write_markdown(rows, out / "table_s3_closure.md")
    print(f"[s3] wrote {out / 'table_s3_closure.md'}", flush=True)


if __name__ == "__main__":
    main()
