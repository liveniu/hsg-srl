#!/usr/bin/env python3
"""Build the localization, fixed-anchor, and tracking summaries from the recorded results.

This does not retrain. Localization uses results/recorded/loc_support, which
stores overlap-gated residual errors. Tracking uses
results/recorded/identity_hota25_overlap.csv. The older
identity_hota25_support.csv is the camera-support ablation and is still
checked against the saved bins.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from bootstrap import ARTIFACT_ROOT
from intervals import paired_bootstrap
from protocol import BOOT_SEED, GROUPS, SEEDS, SITES

RECORDED = ARTIFACT_ROOT / "results" / "recorded"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
    }


def seed_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)),
        "seeds": [float(value) for value in array],
    }


def load_arrays() -> dict[str, dict]:
    arrays = {}
    for site in SITES:
        path = RECORDED / "loc_support" / f"{site}.npz"
        if not path.exists():
            raise SystemExit(
                f"missing {path}. Run scripts/export_support_localization.py "
                "where the released geometry databases are available."
            )
        blob = np.load(path)
        arrays[site] = {key: blob[key] for key in blob.files}
    return arrays


def clusters_of(cluster: np.ndarray) -> dict[int, np.ndarray]:
    return {int(label): np.flatnonzero(cluster == label) for label in np.unique(cluster)}


def main() -> None:
    arrays = load_arrays()
    tracking_rows = read_csv(RECORDED / "identity_hota25.csv")
    overlap_file = RECORDED / "identity_hota25_overlap.csv"
    if not overlap_file.exists():
        raise SystemExit(f"missing {overlap_file}. Run scripts/assemble_hota_tables.py.")
    support_rows = read_csv(overlap_file)
    geo_track = {
        (row["site"], float(row["gate"])): row
        for row in tracking_rows if row["method"] == "Geo"
    }
    support_track = {
        (row["site"], int(row["seed"]), float(row["gate"])): row
        for row in support_rows if row["method"] == "HSG-SRL"
    }
    ipm = json.loads((RECORDED / "ipm_baseline.json").read_text(encoding="utf-8"))

    groups = dict(GROUPS)
    groups["LUMPI-real"] = tuple(site for site in SITES if site.startswith("lumpi_"))

    lines = [
        "# Tables rebuilt inside the artifact",
        "",
        "Localization and tracking use the published deployment: a trained residual",
        "on a supported camera only where the fixed HSG point lies in the frozen overlap union.",
        "",
        "## Localization, HSG-SRL is the three-seed mean +/- sample SD",
        "",
        "| Stratum | IPM median | HSG median | HSG-SRL median | HSG p90 | HSG-SRL p90 | HSG RMSE | HSG-SRL RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    # HSG is not in ipm_baseline; it is the geo column of the support arrays.
    strata = ("LUMPI", "LUMPI-pole", "LUMPI-A", "LUMPI-B", "Ko-PER", "Synth")
    labels = {
        "LUMPI": "LUMPI",
        "LUMPI-pole": "LUMPI-pole",
        "LUMPI-A": "LUMPI-A",
        "LUMPI-B": "LUMPI-B",
        "Ko-PER": "Ko-PER",
        "Synth": "Synthehicle",
    }
    for stratum in strata:
        members = groups[stratum]
        base = stats(np.concatenate([arrays[site]["geo"] for site in members]))
        seeds = []
        for seed in SEEDS:
            seeds.append(stats(np.concatenate([arrays[site][f"f0_s{seed}"] for site in members])))
        median = seed_summary([item["median"] for item in seeds])
        p90 = seed_summary([item["p90"] for item in seeds])
        rmse = seed_summary([item["rmse"] for item in seeds])
        ipm_median = ipm["pooled"][stratum]["bottom"]["median"]
        lines.append(
            f"| {labels[stratum]} | {ipm_median:.3f} | {base['median']:.3f} | {median['mean']:.3f} +/- {median['sd']:.3f} "
            f"| {base['p90']:.3f} | {p90['mean']:.3f} +/- {p90['sd']:.3f} "
            f"| {base['rmse']:.3f} | {rmse['mean']:.3f} +/- {rmse['sd']:.3f} |"
        )

    lines += [
        "",
        "## Fixed anchors, median / RMSE (m)",
        "",
        "| Anchor | LUMPI | Ko-PER | Synthehicle |",
        "| --- | ---: | ---: | ---: |",
    ]
    anchor_names = (
        ("center", "Box-center projection"),
        ("bottom", "Bottom-center projection (IPM)"),
        ("alpha", "Sensitivity only"),
        ("shape", "Shape only"),
        ("b050", "Fixed interior midpoint"),
        ("b070", "STMC-derived fixed anchor"),
    )
    anchor_strata = ("LUMPI", "Ko-PER", "Synth")
    hsg_cells = []
    for stratum in anchor_strata:
        values = np.concatenate([arrays[site]["geo"] for site in groups[stratum]])
        summary = stats(values)
        hsg_cells.append(f"{summary['median']:.3f} / {summary['rmse']:.3f}")
    lines.append("| HSG (per-detection) | " + " | ".join(hsg_cells) + " |")
    for key, label in anchor_names:
        cells = []
        for stratum in anchor_strata:
            data = ipm["pooled"][stratum][key]
            cells.append(f"{data['median']:.3f} / {data['rmse']:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Recording-macro TrackEval BEV-HOTA",
        "",
        "| Data / method | 0.5 m | 1 m | 1.5 m | 2 m | mean of four |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    track_groups = {
        "LUMPI": groups["LUMPI"],
        "Ko-PER": groups["Ko-PER"],
        "Synthehicle": groups["Synth"],
    }
    for label, members in track_groups.items():
        geo_vals = {}
        f0_vals = {}
        for gate in (0.5, 1.0, 1.5, 2.0):
            geo_vals[gate] = float(np.mean([100 * float(geo_track[(site, gate)]["hota"]) for site in members]))
            seed_means = []
            for seed in SEEDS:
                seed_means.append(float(np.mean([
                    100 * float(support_track[(site, seed, gate)]["hota"])
                    for site in members
                ])))
            f0_vals[gate] = seed_summary(seed_means)
        geo_four = float(np.mean([geo_vals[gate] for gate in (0.5, 1.0, 1.5, 2.0)]))
        f0_four = seed_summary([
            float(np.mean([f0_vals[gate]["seeds"][seed] for gate in (0.5, 1.0, 1.5, 2.0)]))
            for seed in range(3)
        ])
        lines.append(
            f"| {label} / HSG | " + " | ".join(f"{geo_vals[gate]:.2f}" for gate in (0.5, 1.0, 1.5, 2.0))
            + f" | {geo_four:.2f} |"
        )
        lines.append(
            f"| {label} / HSG-SRL | " + " | ".join(
                f"{f0_vals[gate]['mean']:.2f} +/- {f0_vals[gate]['sd']:.2f}" for gate in (0.5, 1.0, 1.5, 2.0)
            ) + f" | {f0_four['mean']:.2f} +/- {f0_four['sd']:.2f} |"
        )

    lines += [
        "",
        "## Per-recording median and seed-0 bootstrap interval for HSG-SRL minus HSG",
        "",
        "| Recording | HSG | HSG-SRL mean +/- SD | Seed-0 difference [interval] |",
        "| --- | ---: | ---: | ---: |",
    ]
    archived = {}
    for site in SITES:
        archived[site] = paired_bootstrap(
            clusters_of(arrays[site]["cluster"]),
            arrays[site]["geo"],
            arrays[site]["f0_s0"],
            rng=np.random.default_rng(BOOT_SEED),
        )
    for site in SITES:
        base = stats(arrays[site]["geo"])["median"]
        medians = [stats(arrays[site][f"f0_s{seed}"])["median"] for seed in SEEDS]
        summary = seed_summary(medians)
        median = archived[site]["median"]
        label = site.replace("lumpi_", "").replace("synth_", "").replace("_O_day", "")
        lines.append(
            f"| {label} | {base:.3f} | {summary['mean']:.3f} +/- {summary['sd']:.3f} "
            f"| {median['d']:+.3f} [{median['lo']:+.3f}, {median['hi']:+.3f}] |"
        )

    def macro_metric(members, gate, method, field):
        if method == "geo":
            return float(np.mean([100 * float(geo_track[(site, gate)][field]) for site in members]))
        seeds = []
        for seed in SEEDS:
            seeds.append(float(np.mean([
                100 * float(support_track[(site, seed, gate)][field])
                for site in members
            ])))
        return seed_summary(seeds)

    lines += [
        "",
        "## Detection and association accuracy",
        "",
        "| Data / gate | HSG DetA | HSG-SRL DetA | HSG AssA | HSG-SRL AssA |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, members in track_groups.items():
        for gate in (0.5, 1.0, 1.5, 2.0):
            deta_f = macro_metric(members, gate, "f0", "deta")
            assa_f = macro_metric(members, gate, "f0", "assa")
            lines.append(
                f"| {label} / {gate:g} m | {macro_metric(members, gate, 'geo', 'deta'):.2f} "
                f"| {deta_f['mean']:.2f} +/- {deta_f['sd']:.2f} "
                f"| {macro_metric(members, gate, 'geo', 'assa'):.2f} "
                f"| {assa_f['mean']:.2f} +/- {assa_f['sd']:.2f} |"
            )
    lines += [
        "",
        "## Localization accuracy of retained matches",
        "",
        "| Data / gate | HSG LocA | HSG-SRL LocA |",
        "| --- | ---: | ---: |",
    ]
    for label, members in track_groups.items():
        for gate in (0.5, 1.0, 1.5, 2.0):
            loca_f = macro_metric(members, gate, "f0", "loca")
            lines.append(
                f"| {label} / {gate:g} m | {macro_metric(members, gate, 'geo', 'loca'):.2f} "
                f"| {loca_f['mean']:.2f} +/- {loca_f['sd']:.2f} |"
            )
    lines += [
        "",
        "## Per-recording BEV-HOTA",
        "",
        "| Recording | HSG 0.5 m | HSG-SRL 0.5 m | HSG 2 m | HSG-SRL 2 m |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for site in SITES:
        label = site.replace("lumpi_", "").replace("synth_", "").replace("_O_day", "")
        cells = []
        for gate in (0.5, 2.0):
            cells.append(f"{100 * float(geo_track[(site, gate)]['hota']):.2f}")
            refined = seed_summary([
                100 * float(support_track[(site, seed, gate)]["hota"])
                for seed in SEEDS
            ])
            cells.append(f"{refined['mean']:.2f} +/- {refined['sd']:.2f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    dest = ARTIFACT_ROOT / "results" / "tables" / "computed_tables.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    real = stats(np.concatenate([arrays[site]["f0_s0"] for site in groups["LUMPI-real"]]))
    print(f"[tables] -> {dest}")
    print(f"[tables] real seed-0 median {real['median']:.3f} n={real['n']}")


if __name__ == "__main__":
    main()
