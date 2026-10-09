#!/usr/bin/env python3
"""Rescore fixed anchors when the common GT offset is not the HSG estimate.

The published tables add common_bias only where reference_verdict.txt starts
with REFERENCE OFFSET. That vector is the per-camera median of
(HSG world - raw GT) on stationary samples. This script keeps the frozen
detector-to-GT attachment and re-estimates the same verdict from the
bottom-center and beta=0.50 projections. It does not rewrite
reference_verdict.txt or geometry.db, and it does not refit the residual.

Clock attachment is not rerun. On moving samples the script reports the
along-track time that would equate an anchor shift with a clock shift.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from bootstrap import ARTIFACT_ROOT
from db_access import connect_ro, load_adapted, load_bias, load_test_keys, matched_rows, positions
from export_support_localization import support_masks
from protocol import DET_BATCH, GROUPS, MAX_T, SEEDS, SITES, geometry_db, site_dir

PARKED_SPEED = 0.3
COMMON_BIAS_MIN = 0.25
MOVING_SPEED = 1.0
ANCHORS = (("hsg", None), ("b050", 0.50), ("bottom", 1.0))
SCORE_BETAS = (("center", 0.0), ("b050", 0.50), ("b070", 0.70), ("bottom", 1.0))
OUT = ARTIFACT_ROOT / "results" / "offset_sensitivity"


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
    }


def project(box: np.ndarray, homography: np.ndarray, scale: float, beta: float) -> np.ndarray:
    uv = np.column_stack((
        box[:, 0] + box[:, 2] / 2.0,
        box[:, 1] + (1.0 + beta) * box[:, 3] / 2.0,
        np.ones(len(box)),
    ))
    metric = np.diag([1.0 / scale, 1.0 / scale, 1.0]) @ homography
    projected = uv @ metric.T
    return projected[:, :2] / projected[:, 2:3]


def load_attached(site: str) -> dict[str, dict]:
    adapted, _sizes = load_adapted(site)
    conn = connect_ro(geometry_db(site))
    raw = conn.execute(
        """SELECT camera_name, local_track_id, timestamp_sec,
                  bbox_x, bbox_y, bbox_w, bbox_h, world_x, world_y,
                  outside_reference_x, outside_reference_y
           FROM trajectory_observations
           WHERE scene_id=? AND batch_id=? AND outside_reference_x IS NOT NULL
           ORDER BY camera_name, local_track_id, timestamp_sec""",
        (site, DET_BATCH),
    ).fetchall()
    conn.close()
    by_cam: dict[str, list] = defaultdict(list)
    for row in raw:
        cam = row[0]
        if cam not in adapted or cam not in SITES[site]["cameras"]:
            continue
        if any(value is None or not np.isfinite(value) for value in row[2:]):
            continue
        by_cam[cam].append(row[1:])
    out = {}
    for cam, rows in by_cam.items():
        arr = np.asarray(rows, dtype=float)
        homography, scale = adapted[cam]
        box = arr[:, 2:6]
        out[cam] = {
            "tid": arr[:, 0].astype(np.int64),
            "t": arr[:, 1],
            "hsg": arr[:, 6:8],
            "gt": arr[:, 8:10],
            "b050": project(box, homography, scale, 0.50),
            "bottom": project(box, homography, scale, 1.0),
        }
    return out


def verdict_for(points: dict[str, dict], key: str) -> dict:
    per_cam = {}
    clock = []
    for cam, blob in points.items():
        dxs, dys = [], []
        for tid in np.unique(blob["tid"]):
            select = np.flatnonzero(blob["tid"] == tid)
            if select.size < 5:
                continue
            order = select[np.argsort(blob["t"][select])]
            gt = blob["gt"][order]
            timestamp = blob["t"][order]
            speed = np.hypot(np.gradient(gt[:, 0], timestamp), np.gradient(gt[:, 1], timestamp))
            parked = speed < PARKED_SPEED
            if parked.any():
                delta = blob[key][order][parked] - gt[parked]
                dxs.extend(delta[:, 0].tolist())
                dys.extend(delta[:, 1].tolist())
            if key != "hsg":
                moving = speed >= MOVING_SPEED
                if moving.any():
                    tangent = np.column_stack((
                        np.gradient(gt[:, 0], timestamp),
                        np.gradient(gt[:, 1], timestamp),
                    ))
                    tangent = tangent / np.maximum(speed[:, None], 1e-6)
                    along = np.sum((blob["hsg"][order] - blob[key][order]) * tangent, axis=1)
                    clock.extend((along[moving] / speed[moving]).tolist())
        bias = np.array([np.median(dxs), np.median(dys)]) if dxs else np.zeros(2)
        per_cam[cam] = {"n_parked": len(dxs), "bias_full": np.asarray(bias, dtype=float)}
    common = np.mean([per_cam[cam]["bias_full"] for cam in per_cam], axis=0)
    spread = max(float(np.linalg.norm(per_cam[cam]["bias_full"] - common)) for cam in per_cam)
    norm = float(np.linalg.norm(common))
    applied = bool(norm > COMMON_BIAS_MIN and norm > spread)
    common = np.array([float(f"{float(common[0]):.4f}"), float(f"{float(common[1]):.4f}")])
    for cam in per_cam:
        full = per_cam[cam].pop("bias_full")
        per_cam[cam]["bias"] = [float(f"{float(full[0]):.4f}"), float(f"{float(full[1]):.4f}")]
    clock_abs = np.abs(np.asarray(clock, dtype=float)) if clock else np.zeros(0)
    return {
        "per_camera": per_cam,
        "common_bias": [round(float(common[0]), 4), round(float(common[1]), 4)],
        "common_norm_m": round(norm, 4),
        "spread_m": round(spread, 4),
        "applied": applied,
        "clock_equiv_s": {
            "n": int(clock_abs.size),
            "median_abs": None if clock_abs.size == 0 else round(float(np.median(clock_abs)), 4),
            "p90_abs": None if clock_abs.size == 0 else round(float(np.percentile(clock_abs, 90)), 4),
        },
    }


def published_bias_vector(site: str) -> list[float]:
    text = (site_dir(site) / "reference_verdict.txt").read_text(encoding="utf-8")
    line = next(item for item in text.splitlines() if item.startswith("common_bias="))
    body = line.split("=(")[1].rstrip(")")
    return [round(float(part), 4) for part in body.split(",")]


def test_predictions(site: str) -> dict[str, np.ndarray]:
    keys = load_test_keys(site)
    adapted, _sizes = load_adapted(site)
    cameras = set(SITES[site]["cameras"])
    max_t = MAX_T.get(site)
    conn = connect_ro(geometry_db(site))
    raw = conn.execute(
        """SELECT camera_name, local_track_id, frame_index, timestamp_sec,
                  bbox_x, bbox_y, bbox_w, bbox_h, world_x, world_y,
                  outside_reference_x, outside_reference_y
           FROM trajectory_observations
           WHERE scene_id=? AND batch_id=?
             AND world_x IS NOT NULL AND outside_reference_x IS NOT NULL""",
        (site, DET_BATCH),
    ).fetchall()
    conn.close()
    kept = []
    for cam, tid, frame, timestamp, *values in raw:
        if cam not in cameras or cam not in adapted or (cam, int(tid), int(frame)) not in keys:
            continue
        if max_t is not None and float(timestamp) > max_t:
            continue
        if any(value is None for value in values):
            continue
        kept.append((cam, *values))
    coords = np.asarray([row[1:] for row in kept], dtype=float)
    cam_array = np.asarray([row[0] for row in kept])
    pred = {"hsg": coords[:, 4:6].copy(), "gt": coords[:, 6:8].copy()}
    for name, beta in SCORE_BETAS:
        projected = np.empty((len(kept), 2))
        for cam in sorted(set(cam_array.tolist())):
            select = cam_array == cam
            homography, scale = adapted[cam]
            projected[select] = project(coords[select, :4], homography, scale, beta)
        pred[name] = projected
    return pred


def shift_errors(pred: dict[str, np.ndarray], bias: np.ndarray) -> dict[str, np.ndarray]:
    gt = pred["gt"] + bias
    return {name: np.linalg.norm(pred[name] - gt, axis=1) for name in ("hsg", "center", "b050", "b070", "bottom")}


def residual_errors(site: str, biases: dict[str, np.ndarray], masks) -> dict[str, dict[str, np.ndarray]]:
    rows, _by = matched_rows(site)
    if not rows:
        return {}
    published = load_bias(site)
    raw = np.stack([row["gt"] for row in rows]) - published
    checkpoints = ARTIFACT_ROOT / "results" / "recorded" / "checkpoints"
    out = {}
    for seed in SEEDS:
        opened = masks[(site, seed)]
        deploy = {"deploy": "model" if opened else "geo", "S": sorted(opened)}
        xy = positions(site, seed, rows, deploy, checkpoints)
        out[f"s{seed}"] = {
            name: np.linalg.norm(xy - (raw + bias), axis=1) for name, bias in biases.items()
        }
    return out


def main() -> None:
    masks = support_masks()
    per_site = {}
    anchor_errors: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    residual_by_site = {}
    for site in SITES:
        attached = load_attached(site)
        verdicts = {key: verdict_for(attached, key) for key, _beta in ANCHORS}
        published = published_bias_vector(site)
        got = np.asarray(verdicts["hsg"]["common_bias"], dtype=float)
        if float(np.max(np.abs(got - np.asarray(published)))) > 5e-4:
            raise SystemExit(f"{site}: recomputed HSG bias {got.tolist()} != published {published}")
        pred = test_predictions(site)
        treatments = {
            "published_hsg": load_bias(site),
            "none": np.zeros(2),
            "from_bottom": np.asarray(verdicts["bottom"]["common_bias"]) if verdicts["bottom"]["applied"] else np.zeros(2),
            "from_b050": np.asarray(verdicts["b050"]["common_bias"]) if verdicts["b050"]["applied"] else np.zeros(2),
        }
        scored = {name: shift_errors(pred, bias) for name, bias in treatments.items()}
        anchor_errors[site] = scored
        residual_by_site[site] = residual_errors(site, treatments, masks)
        per_site[site] = {
            "n_test": int(pred["gt"].shape[0]),
            "verdicts": verdicts,
            "applied_bias": {name: [round(float(v), 4) for v in bias] for name, bias in treatments.items()},
            "anchors": {
                treatment: {anchor: stats(values) for anchor, values in errors.items()}
                for treatment, errors in scored.items()
            },
        }
        hsg_med = per_site[site]["anchors"]["published_hsg"]["hsg"]["median"]
        b50_med = per_site[site]["anchors"]["published_hsg"]["b050"]["median"]
        print(
            f"[offset] {site} n={pred['gt'].shape[0]} "
            f"published {hsg_med:.3f}/{b50_med:.3f} "
            f"apply hsg={verdicts['hsg']['applied']} "
            f"b050={verdicts['b050']['applied']} bottom={verdicts['bottom']['applied']}",
            flush=True,
        )

    groups = dict(GROUPS)
    groups["LUMPI-real"] = tuple(site for site in SITES if site.startswith("lumpi_"))
    groups["All"] = tuple(SITES)
    pooled = {}
    for group, members in groups.items():
        pooled[group] = {}
        for treatment in ("published_hsg", "none", "from_bottom", "from_b050"):
            pooled[group][treatment] = {
                anchor: stats(np.concatenate([anchor_errors[site][treatment][anchor] for site in members]))
                for anchor in ("hsg", "center", "b050", "b070", "bottom")
            }
    residual_summary = {}
    for site, seeds in residual_by_site.items():
        residual_summary[site] = {}
        for treatment in ("published_hsg", "none", "from_bottom", "from_b050"):
            medians = [float(np.median(seeds[f"s{seed}"][treatment])) for seed in SEEDS]
            residual_summary[site][treatment] = {
                "hsg_srl_median_mean": float(np.mean(medians)),
                "hsg_srl_median_seeds": [round(value, 4) for value in medians],
            }

    payload = {"per_recording": per_site, "pooled": pooled, "hsg_srl": residual_summary}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "offset_sensitivity.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (OUT / "summary.md").write_text(render(per_site, pooled, residual_summary), encoding="utf-8")
    print(f"[offset] -> {OUT / 'summary.md'}", flush=True)


def render(per_site: dict, pooled: dict, residual: dict) -> str:
    lines = [
        "# Reference-offset sensitivity",
        "",
        "Same frozen detector-to-GT attachment. The published column adds the released HSG common bias where the verdict is REFERENCE OFFSET. `none` adds nothing. `from_bottom` and `from_b050` re-estimate that verdict from the bottom-center and beta=0.50 projections, using the same stationary-sample rule (speed < 0.3 m/s, at least five samples, |common| > 0.25 m and greater than the per-camera spread). A treatment that does not fire applies no bias.",
        "",
        "Recomputed HSG common-bias vectors match the released `common_bias=` lines. Clock attachment was not rerun. `clock_equiv_s` is the absolute along-track time that equates the HSG-minus-anchor shift on samples moving at least 1 m/s. The attachment interpolation gap is 0.12 s.",
        "",
        "## Applied common bias",
        "",
        "| Recording | HSG bias (m) | Applied | beta=0.50 bias (m) | Applied | beta=1 bias (m) | Applied | beta=0.50 clock median/p90 (s) | beta=1 clock median/p90 (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for site, blob in per_site.items():
        label = site.replace("lumpi_", "").replace("synth_", "").replace("_O_day", "")
        cells = []
        for key in ("hsg", "b050", "bottom"):
            item = blob["verdicts"][key]
            bias = item["common_bias"]
            cells.append(f"({bias[0]:+.4f}, {bias[1]:+.4f})")
            cells.append("yes" if item["applied"] else "no")
        for key in ("b050", "bottom"):
            clock = blob["verdicts"][key]["clock_equiv_s"]
            cells.append(f"{clock['median_abs']:.4f} / {clock['p90_abs']:.4f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Table III strata, median / RMSE (m)",
        "",
        "Rows are the GT treatment. Columns are pooled over the stratum.",
        "",
    ]
    strata = ("LUMPI-pole", "LUMPI-A", "LUMPI-B", "Synth", "LUMPI-real", "All")
    treatments = (
        ("published_hsg", "Published HSG offset"),
        ("none", "No offset"),
        ("from_b050", "Offset from beta=0.50"),
        ("from_bottom", "Offset from beta=1"),
    )
    for stratum in strata:
        lines += [
            f"### {stratum}",
            "",
            "| GT treatment | HSG | beta=0.50 | beta=0.70 | beta=1 | beta=0 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for key, label in treatments:
            cells = []
            for anchor in ("hsg", "b050", "b070", "bottom", "center"):
                item = pooled[stratum][key][anchor]
                cells.append(f"{item['median']:.3f} / {item['rmse']:.3f}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "## Per-recording median, HSG vs beta=0.50 (m)",
        "",
        "| Recording | Published HSG / beta=0.50 | No offset | From beta=0.50 | From beta=1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for site, blob in per_site.items():
        label = site.replace("lumpi_", "").replace("synth_", "").replace("_O_day", "")
        cells = []
        for key in ("published_hsg", "none", "from_b050", "from_bottom"):
            hsg = blob["anchors"][key]["hsg"]["median"]
            other = blob["anchors"][key]["b050"]["median"]
            cells.append(f"{hsg:.3f} / {other:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## HSG-SRL three-seed mean median under the same GT treatments (m)",
        "",
        "The residual is not retrained. M0/M1 stay on HSG because they have no residual support.",
        "",
        "| Recording | Published | No offset | From beta=0.50 | From beta=1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for site, blob in residual.items():
        label = site.replace("lumpi_", "").replace("synth_", "").replace("_O_day", "")
        cells = [f"{blob[key]['hsg_srl_median_mean']:.3f}" for key in ("published_hsg", "none", "from_b050", "from_bottom")]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
