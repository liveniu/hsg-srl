#!/usr/bin/env python3
"""Check the recorded release against the manuscript's reported figures.

Localization is recomputed from results/recorded/loc_support. Tracking is
the overlap-gated table. This does not train, detect, or rerun association.
"""

from __future__ import annotations

import csv

import numpy as np

from bootstrap import ARTIFACT_ROOT
from build_tables import seed_summary, stats
from protocol import GROUPS, SEEDS

RECORDED = ARTIFACT_ROOT / "results" / "recorded"

# Three-decimal manuscript targets. Tolerances cover that rounding.
LOC = {
    "LUMPI": {"n": 827083, "hsg": (0.604, 0.861, 1.324), "srl": (0.553, 0.835, 1.285)},
    "Ko-PER": {"n": 25122, "hsg": (0.313, 0.579, 0.758), "srl": (0.302, 0.561, 0.703)},
    "Synth": {"n": 83361, "hsg": (0.657, 1.063, 1.615), "srl": (0.669, 1.040, 1.599)},
}
HOTA4 = {
    "LUMPI": (39.18, 41.25),
    "Ko-PER": (59.60, 61.99),
    "Synth": (43.67, 43.39),
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    arrays = {}
    for site in [name for names in GROUPS.values() for name in names]:
        path = RECORDED / "loc_support" / f"{site}.npz"
        if site in arrays or not path.exists():
            continue
        blob = np.load(path)
        arrays[site] = {key: blob[key] for key in blob.files}
    failures = []

    def check(name, got, target, tol):
        if abs(got - target) > tol:
            failures.append(f"{name}: {got:.6f} vs {target:.6f}")

    for group, spec in LOC.items():
        members = GROUPS[group]
        missing = [site for site in members if site not in arrays]
        if missing:
            failures.append(f"missing localization arrays: {missing}")
            continue
        geo = np.concatenate([arrays[site]["geo"] for site in members])
        base = stats(geo)
        check(f"{group} n", base["n"], spec["n"], 0)
        for got, target, label in zip(
            (base["median"], base["rmse"], base["p90"]), spec["hsg"], ("median", "rmse", "p90"),
        ):
            check(f"{group} HSG {label}", got, target, 0.0006)
        seeds = [stats(np.concatenate([arrays[site][f"f0_s{seed}"] for site in members])) for seed in SEEDS]
        for label, target in zip(("median", "rmse", "p90"), spec["srl"]):
            summary = seed_summary([item[label] for item in seeds])
            check(f"{group} HSG-SRL {label}", summary["mean"], target, 0.0015)

    geo_rows = read_csv(RECORDED / "identity_hota25.csv")
    overlap_rows = read_csv(RECORDED / "identity_hota25_overlap.csv")
    for group, (hsg_target, srl_target) in HOTA4.items():
        members = GROUPS[group]
        hsg = []
        for site in members:
            gates = [
                100 * float(row["hota"])
                for row in geo_rows
                if row["site"] == site and row["method"] == "Geo"
            ]
            if len(gates) != 4:
                failures.append(f"{site} geo gates={len(gates)}")
                continue
            hsg.append(float(np.mean(gates)))
        check(f"{group} HSG HOTA4", float(np.mean(hsg)), hsg_target, 0.015)
        seed_means = []
        for seed in SEEDS:
            recording_means = []
            for site in members:
                gates = [
                    100 * float(row["hota"])
                    for row in overlap_rows
                    if row["site"] == site and row["method"] == "HSG-SRL" and int(row["seed"]) == seed
                ]
                if len(gates) != 4:
                    failures.append(f"{site} seed {seed} overlap gates={len(gates)}")
                    continue
                recording_means.append(float(np.mean(gates)))
            seed_means.append(float(np.mean(recording_means)))
        check(f"{group} HSG-SRL HOTA4", float(np.mean(seed_means)), srl_target, 0.02)

    text = (ARTIFACT_ROOT / "results" / "tables" / "computed_tables.md").read_text(encoding="utf-8")
    for snippet in ("0.553", "41.25", "Ko-PER", "overlap"):
        if snippet not in text:
            failures.append(f"computed_tables.md missing {snippet}")
    supplement = ARTIFACT_ROOT / "results" / "tables" / "supplement_controls.md"
    bev = (ARTIFACT_ROOT / "results" / "bevheight" / "score_three.md").read_text(encoding="utf-8")
    if not supplement.exists():
        failures.append("missing supplement_controls.md")
    else:
        body = supplement.read_text(encoding="utf-8")
        for snippet in (
            "0.845 / 0.660 / 0.720",
            "8,287 (32.99%)",
            "0.533 / 0.823 / 1.262",
            "0.563 / 0.830 / 1.258",
            "0.012% / 5.539%",
        ):
            if snippet not in body:
                failures.append(f"supplement_controls.md missing {snippet}")
    for snippet in ("n>5m=15 (0.012%)", "C05, C06, and C07", "med=0.563"):
        if snippet not in bev:
            failures.append(f"score_three.md missing {snippet}")
    if failures:
        raise SystemExit("RELEASE CHECK FAILED\n" + "\n".join(failures))
    print("RELEASE CHECK PASSED")


if __name__ == "__main__":
    main()
