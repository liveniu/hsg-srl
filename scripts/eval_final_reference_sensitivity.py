#!/usr/bin/env python3
"""Optional LUMPI translation audit. Not a published table.

Published scoring uses the stored footprint centers. This script compares
that reference with the audit translation and two re-estimated translations.

Without flags, rebuild the Markdown readout from the recorded JSON. Use
--cache-root for frozen prediction vectors, or --from-data for the released
geometry databases and existing checkpoints. Neither mode fits a model,
changes correspondences, reruns association, or scores tracking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bootstrap import ARTIFACT_ROOT

OUT = ARTIFACT_ROOT / "results" / "offset_sensitivity"
TREATMENTS = (
    ("published_hsg", "HSG-derived offset"),
    ("none", "Offset removed"),
    ("from_b050", "Offset from midpoint"),
    ("from_bottom", "Offset from bottom center"),
)
SITES = tuple(f"lumpi_M{i}" for i in range(7))
METRICS = ("median", "rmse", "p90")


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"median": float(np.median(values)),
            "rmse": float(np.sqrt(np.mean(values * values))),
            "p90": float(np.percentile(values, 90))}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive(cache_root=None):
    reference_path = OUT / "offset_sensitivity.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    pooled = {name: [[], [], []] for name, _label in TREATMENTS}
    sources = {"reference_treatments": {"file": "offset_sensitivity.json",
                                       "sha256": sha256(reference_path)},
               "recordings": {}}
    masks = None
    if cache_root is None:
        from eval_offset_sensitivity import residual_errors
        from export_support_localization import support_masks
        masks = support_masks()

    total = 0
    for site in SITES:
        source = reference["per_recording"][site]
        biases = {name: np.asarray(source["applied_bias"][name], dtype=float)
                  for name, _label in TREATMENTS}
        recorded_path = ARTIFACT_ROOT / "results" / "recorded" / "loc_support" / f"{site}.npz"
        provenance = {"recorded_localization_sha256": sha256(recorded_path)}
        if cache_root is not None:
            path = Path(cache_root) / f"{site}.npz"
            with np.load(path, allow_pickle=False) as cache:
                xy, gt = cache["xy"], cache["gt"]
                active = cache["opened"].astype(bool) & (cache["depth"] >= -1e-6)
                final = xy[None, :, :] + cache["residual"] * active[None, :, None]
            raw_gt = gt - biases["published_hsg"]
            errors = {f"s{seed}": {name: np.linalg.norm(final[seed] - (raw_gt + bias), axis=1)
                                  for name, bias in biases.items()} for seed in range(3)}
            for name, bias in biases.items():
                geometric = stats(np.linalg.norm(xy - (raw_gt + bias), axis=1))
                for metric in ("median", "rmse"):
                    expected = source["anchors"][name]["hsg"][metric]
                    if not np.isclose(geometric[metric], expected, rtol=0, atol=1e-8):
                        raise ValueError(f"{site} {name}: geometric reference mismatch")
            provenance.update({"prediction_cache": path.name, "prediction_cache_sha256": sha256(path)})
        else:
            errors = residual_errors(site, biases, masks)
            provenance["prediction_source"] = "released geometry DB, frozen camera support, overlap polygons, and checkpoints"

        n = len(errors["s0"]["published_hsg"])
        if n != source["n_test"]:
            raise ValueError(f"{site}: matched cohort changed")
        with np.load(recorded_path, allow_pickle=False) as recorded:
            for seed in range(3):
                actual = stats(errors[f"s{seed}"]["published_hsg"])
                expected = stats(recorded[f"f0_s{seed}"])
                if any(not np.isclose(actual[m], expected[m], rtol=0, atol=2e-5) for m in METRICS):
                    raise ValueError(f"{site} seed {seed}: final deployment does not match recorded localization")
        for name, _label in TREATMENTS:
            for seed in range(3):
                pooled[name][seed].append(errors[f"s{seed}"][name])
        total += n
        sources["recordings"][site] = provenance

    results = {}
    for name, _label in TREATMENTS:
        seeds = [stats(np.concatenate(parts)) for parts in pooled[name]]
        mean = {m: float(np.mean([s[m] for s in seeds])) for m in METRICS}
        baseline = reference["pooled"]["LUMPI-real"][name]["b050"]
        results[name] = {
            "hsg_srl_mean": mean,
            "hsg_srl_seeds": seeds,
            "fixed_midpoint": baseline,
            "reduction_vs_midpoint_percent": {m: 100 * (1 - mean[m] / baseline[m])
                                               for m in ("median", "rmse")},
            "all_seeds_better_than_fixed_midpoint": {
                m: all(s[m] < baseline[m] for s in seeds) for m in ("median", "rmse")},
        }
    return {"scope": "Conditional localization only: seven LUMPI recordings; fixed predictions, support, polygons, matching and clocks. No tracking rescore.",
            "prediction_source": "cached_final_predictions" if cache_root is not None else "released_data_and_frozen_checkpoints",
            "cohort_n": total, "results": results, "sources": sources}


def render(payload):
    lines = ["# Final overlap-gated LUMPI reference sensitivity", "",
             "Same frozen models, support, polygons, matched observations, and clock alignment. "
             "Only the scoring-reference translation changes. Errors pool the seven recordings "
             "within each seed before averaging the three seed statistics. Tracking is not rescored.", "",
             "| Reference treatment | Fixed midpoint median / RMSE (m) | HSG-SRL mean median / RMSE (m) | Median / RMSE reduction (%) | Both errors lower in every seed |",
             "| --- | ---: | ---: | ---: | --- |"]
    for key, label in TREATMENTS:
        row = payload["results"][key]
        pair = lambda item: " / ".join(f"{item[m]:.3f}" for m in ("median", "rmse"))
        reductions = " / ".join(f"{row['reduction_vs_midpoint_percent'][m]:.1f}" for m in ("median", "rmse"))
        every_seed = "Yes" if all(row["all_seeds_better_than_fixed_midpoint"].values()) else "No"
        lines.append(f"| {label} | {pair(row['fixed_midpoint'])} | {pair(row['hsg_srl_mean'])} | {reductions} | {every_seed} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cache-root", type=Path, help="Directory of frozen matched prediction-vector NPZ files")
    group.add_argument("--from-data", action="store_true", help="Recompute using released geometry DBs and existing checkpoints")
    args = parser.parse_args()
    path = OUT / "final_overlap_lumpi.json"
    if args.cache_root is not None or args.from_data:
        payload = derive(args.cache_root)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    dest = OUT / "final_overlap_lumpi.md"
    dest.write_text(render(payload), encoding="utf-8")
    print(f"[reference] {payload['cohort_n']} detections -> {dest}")


if __name__ == "__main__":
    main()
