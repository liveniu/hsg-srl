#!/usr/bin/env python3
"""Decide whether a recording has one shared ground-truth offset.

A verdict that starts with REFERENCE OFFSET is the only case in which the
common bias is added before localization scoring. A NO COMMON OFFSET file
may still contain a common_bias line; that value is not applied.
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from collections import defaultdict

import numpy as np

from bootstrap import SRC_ROOT
from protocol import DET_BATCH, GT_BATCH, SITES, geometry_db, lumpi_root, origin_db, site_dir, synthehicle_root

PARKED_SPEED = 0.3
COMMON_BIAS_MIN = 0.25
FAULT_MEDIAN_M = 5.0


def camera_positions(site: str, origin_xy: np.ndarray) -> dict[str, np.ndarray]:
    spec = SITES[site]
    sys.path.insert(0, str(SRC_ROOT / "importers"))
    if spec["kind"] == "lumpi":
        import import_lumpi as lumpi

        meta = lumpi.load_meta(lumpi_root())
        calibrations = lumpi.load_all_calibs(
            meta, spec["measurement"], lumpi_root(), z_ground=-2.0, use_test_split=False,
        )
        return {name: calib.T_cam_to_world[:2, 3] - origin_xy for name, calib in calibrations.items()}
    import import_synthehicle as synth

    calibrations = synth.load_all_calibs(
        synthehicle_root() / "calibration",
        spec["town"],
        camera_names=list(spec["cameras"]),
        z_ground=0.0,
    )
    return {name: calib.T_cam_to_world[:2, 3] - origin_xy for name, calib in calibrations.items()}


def main() -> None:
    from utils.scene_geometry_db import load_import_audit_metadata

    sites = sys.argv[1:] or list(SITES)
    for site in sites:
        spec = SITES[site]
        origin = sqlite3.connect(str(origin_db(site)))
        meta = load_import_audit_metadata(origin, site, GT_BATCH)
        origin.close()
        origin_xy = np.asarray(meta.get("origin_world", [0.0, 0.0]), dtype=float)
        positions = camera_positions(site, origin_xy)
        conn = sqlite3.connect(str(geometry_db(site)))
        out = site_dir(site)
        audit_rows, bias_by_cam, fault_units = [], {}, []
        for cam in spec["cameras"]:
            rows = conn.execute(
                """SELECT local_track_id, timestamp_sec, world_x, world_y,
                          outside_reference_x, outside_reference_y
                   FROM trajectory_observations
                   WHERE scene_id=? AND batch_id=? AND camera_name=?
                     AND outside_reference_x IS NOT NULL
                   ORDER BY local_track_id, timestamp_sec""",
                (site, DET_BATCH, cam),
            ).fetchall()
            by_track = defaultdict(list)
            for row in rows:
                by_track[row[0]].append(row[1:])
            radial, tangential, dxs, dys = [], [], [], []
            for samples in by_track.values():
                if len(samples) < 5:
                    continue
                arr = np.asarray(samples, dtype=float)
                timestamp, wx, wy, gx, gy = arr.T
                speed = np.hypot(np.gradient(gx, timestamp), np.gradient(gy, timestamp))
                parked = speed < PARKED_SPEED
                if not parked.any():
                    continue
                rx, ry = wx[parked] - gx[parked], wy[parked] - gy[parked]
                ux, uy = gx[parked] - positions[cam][0], gy[parked] - positions[cam][1]
                norm = np.hypot(ux, uy)
                norm = np.where(norm < 1e-6, 1.0, norm)
                ux, uy = ux / norm, uy / norm
                radial.extend((-(rx * ux + ry * uy)).tolist())
                tangential.extend((rx * (-uy) + ry * ux).tolist())
                dxs.extend(rx.tolist())
                dys.extend(ry.tolist())
            bias = np.array([np.median(dxs), np.median(dys)]) if dxs else np.zeros(2)
            bias_by_cam[cam] = bias
            audit_rows.append({
                "site": site,
                "camera": cam,
                "n_parked": len(dxs),
                "bias_x": round(float(bias[0]), 3),
                "bias_y": round(float(bias[1]), 3),
                "radial_median_m": round(float(np.median(radial)), 3) if radial else None,
                "tangential_median_m": round(float(np.median(tangential)), 3) if tangential else None,
            })
        conn.close()
        common = np.mean([bias_by_cam[cam] for cam in spec["cameras"]], axis=0)
        common_norm = float(np.linalg.norm(common))
        spread = max(float(np.linalg.norm(bias_by_cam[cam] - common)) for cam in spec["cameras"])
        if common_norm > COMMON_BIAS_MIN and common_norm > spread:
            verdict = (
                f"REFERENCE OFFSET: cameras share a common world bias "
                f"({common[0]:+.3f},{common[1]:+.3f}) m"
            )
        else:
            verdict = (
                f"NO COMMON OFFSET (|common|={common_norm:.3f} m, per-camera "
                f"spread {spread:.3f} m)"
            )
        (out / "reference_verdict.txt").write_text(
            verdict + f"\ncommon_bias=({common[0]:+.4f},{common[1]:+.4f})\n"
            f"fault_units={len(fault_units)}\n",
            encoding="utf-8",
        )
        with (out / "reference_audit.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(audit_rows)
        print(f"[audit] {site}: {verdict}", flush=True)


if __name__ == "__main__":
    main()
