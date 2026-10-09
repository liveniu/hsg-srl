#!/usr/bin/env python3
"""Write per-detection localization arrays for the published deployment.

Reads the released geometry databases, the test split, and the recorded
checkpoints. A camera uses the residual when it has a retained validation
pair. The output includes the track-cluster index used by the paired
bootstrap. This does not retrain.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from bootstrap import ARTIFACT_ROOT
from db_access import matched_rows, positions
from protocol import SEEDS, SITES

RECORDED = ARTIFACT_ROOT / "results" / "recorded"


def support_masks() -> dict[tuple[str, int], set[str]]:
    rows = list(csv.DictReader((RECORDED / "residual_support.csv").open(encoding="utf-8-sig")))
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["site"], int(row["seed"])), []).append(row)
    masks = {}
    for key, group in grouped.items():
        if group and group[0]["deploy"] == "model":
            masks[key] = {
                row["camera"] for row in group if int(float(row["n_val_pairs"] or 0)) > 0
            }
        else:
            masks[key] = set()
    return masks


def main() -> None:
    sites_root = Path(os.environ.get(
        "HSG_SITE_ROOT",
        str(ARTIFACT_ROOT / "data" / "work" / "sites"),
    ))
    checkpoints = RECORDED / "checkpoints"
    masks = support_masks()
    out = RECORDED / "loc_support"
    out.mkdir(parents=True, exist_ok=True)
    for site in SITES:
        recording = sites_root / site
        db = recording / "geometry.db"
        keys = recording / "test_keys.csv"
        verdict = recording / "reference_verdict.txt"
        if not db.exists() or not keys.exists():
            raise SystemExit(f"missing geometry inputs for {site}: {db}")
        rows, by_track = matched_rows(site, db, keys, verdict if verdict.exists() else None)
        cluster = np.empty(len(rows), np.int32)
        for index, members in enumerate(by_track.values()):
            cluster[members] = index
        gt = np.stack([row["gt"] for row in rows])
        geo = np.linalg.norm(np.stack([row["xy"] for row in rows]) - gt, axis=1)
        payload = {"geo": geo, "cluster": cluster}
        reference = RECORDED / "loc" / f"{site}_loc.npz"
        if reference.exists():
            saved = np.load(reference)["geo"]
            if saved.shape != geo.shape or not np.allclose(saved, geo):
                raise SystemExit(f"{site}: recomputed HSG errors do not match the saved cohort")
        for seed in SEEDS:
            opened = masks[(site, seed)]
            deploy = {"deploy": "model" if opened else "geo", "S": sorted(opened)}
            refined = positions(site, seed, rows, deploy, checkpoints)
            payload[f"f0_s{seed}"] = np.linalg.norm(refined - gt, axis=1)
        np.savez_compressed(out / f"{site}.npz", **payload)
        print(f"[support] {site} n={len(geo)}", flush=True)


if __name__ == "__main__":
    main()
