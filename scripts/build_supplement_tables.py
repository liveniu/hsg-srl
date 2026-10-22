#!/usr/bin/env python3
"""Write the supplement controls that are not in the main computed tables.

S9 and S10 are recomputed from the geometry databases: the same residual with
and without the overlap mask, plus inside and outside the frozen polygons.
S11 scores the pair-symmetric checkpoints on the overlap mask. S16 reads the
stored BEVHeight matched-subset arrays and the published tracking tables.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from bootstrap import ARTIFACT_ROOT, SRC_ROOT
from db_access import camera_polygons, deploy_file, matched_rows, positions
from protocol import SEEDS, SITES

import sys
sys.path.insert(0, str(SRC_ROOT))
from geometry.overlap_gate import inside_mask  # noqa: E402

RECORDED = ARTIFACT_ROOT / "results" / "recorded"
OUT = RECORDED / "supplement"
BH = ARTIFACT_ROOT / "results" / "bevheight"
LABEL = {
    "ko_per": "Ko-PER",
    **{f"lumpi_M{i}": f"M{i}" for i in range(7)},
    **{f"synth_Town0{i}_O_day": f"Town0{i}" for i in range(1, 6)},
}
MVC_SITES = ("lumpi_M6", "lumpi_M2", "ko_per", "synth_Town05_O_day")
BH_SITES = ("lumpi_M6", "lumpi_M3", "synth_Town05_O_day")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"no rows for {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stat3(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "p90": float(np.percentile(values, 90)),
    }


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def hota4(rows, site: str, method: str, seed: str) -> float:
    gates = [
        100 * float(row["hota"])
        for row in rows
        if row["site"] == site and row["method"] == method and str(row.get("seed") or "") == seed
    ]
    if len(gates) != 4:
        raise SystemExit(f"{site} {method} seed {seed!r}: {len(gates)} gates")
    return float(np.mean(gates))


def full_hota(site: str, seed: int) -> float:
    if site == "ko_per":
        blob = json.loads((RECORDED / "hota_sources" / f"ko_per__full__s{seed}.json").read_text(encoding="utf-8"))
        return 100 * float(np.mean([float(row["hota"]) for row in blob["rows"]]))
    return hota4(read_csv(RECORDED / "identity_hota25_support.csv"), site, "HSG-SRL", str(seed))


def membership(site: str, rows: list[dict]) -> np.ndarray:
    polygons = camera_polygons(site)
    xy = np.stack([row["xy"] for row in rows])
    mask = np.zeros(len(rows), dtype=bool)
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["cam"]].append(index)
    for camera, indexes in grouped.items():
        chosen = np.asarray(indexes, dtype=int)
        mask[chosen] = inside_mask(xy[chosen], polygons.get(camera, []))
    return mask


def spatial_tables() -> tuple[list[dict], list[dict]]:
    overlap_rows = read_csv(RECORDED / "identity_hota25_overlap.csv")
    recording = []
    regional = []
    for site in SITES:
        rows, _tracks = matched_rows(site)
        gt = np.stack([row["gt"] for row in rows])
        geo_xy = np.stack([row["xy"] for row in rows])
        geo = np.linalg.norm(geo_xy - gt, axis=1)
        inside = membership(site, rows)
        base = stat3(geo)
        for seed in SEEDS:
            deploy = json.loads(deploy_file(site, seed).read_text(encoding="utf-8"))
            full_xy = positions(site, seed, rows, deploy, spatial=False)
            overlap_xy = positions(site, seed, rows, deploy, spatial=True)
            full = np.linalg.norm(full_xy - gt, axis=1)
            final = np.linalg.norm(overlap_xy - gt, axis=1)
            full_s, final_s = stat3(full), stat3(final)
            recording.append({
                "site": site,
                "seed": seed,
                "n": base["n"],
                "full_median": full_s["median"],
                "full_rmse": full_s["rmse"],
                "full_p90": full_s["p90"],
                "overlap_median": final_s["median"],
                "overlap_rmse": final_s["rmse"],
                "overlap_p90": final_s["p90"],
                "full_hota4": full_hota(site, seed),
                "overlap_hota4": hota4(overlap_rows, site, "HSG-SRL", str(seed)),
            })
            for region, selected in (("inside", inside), ("outside", ~inside)):
                if not np.any(selected):
                    continue
                regional.append({
                    "site": site,
                    "seed": seed,
                    "region": region,
                    "n": int(np.sum(selected)),
                    "fraction": float(np.mean(selected)),
                    "hsg_median": float(np.median(geo[selected])),
                    "hsg_rmse": float(np.sqrt(np.mean(geo[selected] ** 2))),
                    "full_median": float(np.median(full[selected])),
                    "full_rmse": float(np.sqrt(np.mean(full[selected] ** 2))),
                    "overlap_median": float(np.median(final[selected])),
                    "overlap_rmse": float(np.sqrt(np.mean(final[selected] ** 2))),
                })
        print(f"[supplement] spatial {site} n={base['n']} inside={float(np.mean(inside)):.4f}", flush=True)
    return recording, regional


def symmetric_table(recording_rows: list[dict]) -> list[dict]:
    import torch
    from pipeline.geometry_guided_residual_mlp import GeometryGuidedResidualMLP

    out = []
    asymmetric = {(row["site"], int(row["seed"])): row for row in recording_rows}
    for site in MVC_SITES:
        rows, _tracks = matched_rows(site)
        gt = np.stack([row["gt"] for row in rows])
        geo = np.linalg.norm(np.stack([row["xy"] for row in rows]) - gt, axis=1)
        base = stat3(geo)
        for seed in SEEDS:
            path = RECORDED / "checkpoints" / f"{site}_symmetric_seed{seed}.pt"
            blob = torch.load(path, map_location="cpu", weights_only=False)
            model = GeometryGuidedResidualMLP(
                in_dim=int(blob["in_dim"]), hidden=64, max_residual_m=float(blob["max_residual_m"]),
            )
            model.load_state_dict(blob["state_dict"])
            model.eval()
            deploy = json.loads(deploy_file(site, seed).read_text(encoding="utf-8"))
            features = torch.from_numpy(np.stack([row["feat"] for row in rows]).astype(np.float32))
            with torch.no_grad():
                residual = model(features).cpu().numpy().astype(np.float64)
            opened = set(deploy.get("S") or [])
            xy = np.stack([row["xy"] for row in rows])
            sym_xy = xy.copy()
            from db_access import camera_polygons
            polygons = camera_polygons(site)
            grouped = defaultdict(list)
            for index, row in enumerate(rows):
                if row["cam"] in opened:
                    grouped[row["cam"]].append(index)
            for camera, indexes in grouped.items():
                chosen = np.asarray(indexes, dtype=int)
                chosen = chosen[inside_mask(xy[chosen], polygons.get(camera, []))]
                sym_xy[chosen] = sym_xy[chosen] + residual[chosen]
            sym = stat3(np.linalg.norm(sym_xy - gt, axis=1))
            published = asymmetric[(site, seed)]
            out.append({
                "site": site,
                "seed": seed,
                "n": base["n"],
                "hsg_median": base["median"],
                "hsg_rmse": base["rmse"],
                "hsg_p90": base["p90"],
                "asymmetric_median": published["overlap_median"],
                "asymmetric_rmse": published["overlap_rmse"],
                "asymmetric_p90": published["overlap_p90"],
                "symmetric_median": sym["median"],
                "symmetric_rmse": sym["rmse"],
                "symmetric_p90": sym["p90"],
            })
        print(f"[supplement] mvc {site}", flush=True)
    return out


def bevheight_rows() -> list[dict]:
    overlap = read_csv(RECORDED / "identity_hota25_overlap.csv")
    geo = read_csv(RECORDED / "identity_hota25.csv")
    remerge = (BH / "hota_remerge.md").read_text(encoding="utf-8")
    rows = []
    for site in BH_SITES:
        blob = np.load(BH / f"{site}_bevheight_subset.npz")
        entry = {"site": site, "n": int(blob["hsg"].shape[0])}
        for key in ("hsg", "overlap_seed0", "bevheight"):
            err = np.asarray(blob[key], dtype=np.float64)
            summary = stat3(err)
            entry[f"{key}_median"] = summary["median"]
            entry[f"{key}_rmse"] = summary["rmse"]
            entry[f"{key}_p90"] = summary["p90"]
            entry[f"{key}_above5"] = int(np.sum(err > 5))
            entry[f"{key}_frac5"] = float(np.mean(err > 5))
        entry["hsg_hota4"] = hota4(geo, site, "Geo", "")
        entry["srl_hota4"] = hota4(overlap, site, "HSG-SRL", "0")
        marker = f"## {site}"
        block = remerge.split(marker, 1)[1].split("## ", 1)[0]
        bev_line = [line for line in block.splitlines() if line.startswith("- BEV-HOTA")][0]
        entry["bevheight_hota4"] = float(bev_line.split()[-1])
        rows.append(entry)
    return rows


def slash(rows, key: str, digits: int = 3) -> str:
    return " / ".join(fmt(float(row[key]), digits) for row in rows)


def write_markdown(recording, regional, mvc, bev) -> None:
    lines = [
        "# Supplement controls rebuilt in the artifact",
        "",
        "S9 and S10 recompute localization from the released geometry databases.",
        "The residual is the published F0 checkpoint. HSG-SRL applies it only inside",
        "the frozen overlap union. The row without spatial gating uses the same",
        "camera support on every box. HOTA4 is the published tracking replay.",
        "",
        "## S9. Spatial-gating control",
        "",
        "| Recording | Deployment rule | Median / m | RMSE / m | p90 / m | HOTA4 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    by_site = defaultdict(list)
    for row in recording:
        by_site[row["site"]].append(row)
    for site in SITES:
        seeds = by_site[site]
        for rule, prefix in (("Without spatial gating", "full"), ("HSG-SRL", "overlap")):
            lines.append(
                f"| {LABEL[site]} | {rule} | {slash(seeds, prefix + '_median')} | "
                f"{slash(seeds, prefix + '_rmse')} | {slash(seeds, prefix + '_p90')} | "
                f"{slash(seeds, prefix + '_hota4', 2)} |"
            )
    lines += [
        "",
        "## S10. Inside and outside the overlap union",
        "",
        "| Recording | Region | Matched count / fraction | HSG median | No-spatial median | Final median | HSG RMSE | No-spatial RMSE | Final RMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    region_rows = defaultdict(list)
    for row in regional:
        region_rows[(row["site"], row["region"])].append(row)
    for site in SITES:
        for region in ("inside", "outside"):
            seeds = region_rows.get((site, region))
            if not seeds:
                continue
            count = int(seeds[0]["n"])
            fraction = 100 * float(seeds[0]["fraction"])
            lines.append(
                f"| {LABEL[site]} | {region} | {count:,} ({fraction:.2f}%) | "
                f"{fmt(float(seeds[0]['hsg_median']))} | {slash(seeds, 'full_median')} | {slash(seeds, 'overlap_median')} | "
                f"{fmt(float(seeds[0]['hsg_rmse']))} | {slash(seeds, 'full_rmse')} | {slash(seeds, 'overlap_rmse')} |"
            )
    lines += [
        "",
        "## S11. Pair-symmetric MVC control",
        "",
        "Symmetric checkpoints are results/recorded/checkpoints/*_symmetric_seed*.pt.",
        "Both learned columns use the overlap mask and the published camera support.",
        "",
        "| Recording | Seed | HSG | Sensitivity-guided MVC | Pair-symmetric MVC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in mvc:
        def cell(prefix: str) -> str:
            return " / ".join(fmt(float(row[f"{prefix}_{name}"])) for name in ("median", "rmse", "p90"))
        lines.append(
            f"| {LABEL[row['site']]} | {row['seed']} | {cell('hsg')} | {cell('asymmetric')} | {cell('symmetric')} |"
        )
    lines += [
        "",
        "## S16. Supervised BEVHeight reference",
        "",
        "Localization uses the stored matched-subset arrays. HSG-SRL is seed 0 with",
        "C05 included on M6 and the overlap mask applied. Tracking HOTA4 is HSG /",
        "seed-0 HSG-SRL / the mixed BEVHeight replay.",
        "",
        "| Recording | Matched | HSG loc. | HSG-SRL seed-0 loc. | BEVHeight loc. | >5 m: SRL / BEVHeight | HOTA4 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in bev:
        def loc(prefix: str) -> str:
            return " / ".join(fmt(float(row[f"{prefix}_{name}"])) for name in ("median", "rmse", "p90"))
        rate = f"{100 * row['overlap_seed0_frac5']:.3f}% / {100 * row['bevheight_frac5']:.3f}%"
        hota = f"{row['hsg_hota4']:.1f} / {row['srl_hota4']:.1f} / {row['bevheight_hota4']:.1f}"
        lines.append(
            f"| {LABEL[row['site']]} | {row['n']:,} | {loc('hsg')} | {loc('overlap_seed0')} | {loc('bevheight')} | {rate} | {hota} |"
        )
    dest = ARTIFACT_ROOT / "results" / "tables" / "supplement_controls.md"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[supplement] -> {dest}")


def write_score_three(bev) -> None:
    lines = [
        "# BEVHeight reference on M6, M3, and Town05",
        "",
        "Attachment is detection order: the first unused test box containing the projected ground point.",
        "Localization is the stored matched subset. HSG-SRL seed 0 uses retained validation-pair",
        "support and the frozen overlap mask. On M6 that support set is C05, C06, and C07.",
        "A deployment file containing only C06 and C07 is not this comparison.",
        "Full-set HOTA is the published overlap-gated recording score. BEVHeight tracking is in hota_remerge.md.",
        "",
    ]
    for row in bev:
        lines += [
            f"## {row['site']}",
            f"- matched subset: {row['n']}",
            (
                f"- HSG on attached subset: n={row['n']} med={row['hsg_median']:.3f} "
                f"rmse={row['hsg_rmse']:.3f} p90={row['hsg_p90']:.3f} "
                f"n>5m={row['hsg_above5']} ({100 * row['hsg_frac5']:.3f}%)"
            ),
            (
                f"- HSG-SRL seed 0 on attached subset: n={row['n']} med={row['overlap_seed0_median']:.3f} "
                f"rmse={row['overlap_seed0_rmse']:.3f} p90={row['overlap_seed0_p90']:.3f} "
                f"n>5m={row['overlap_seed0_above5']} ({100 * row['overlap_seed0_frac5']:.3f}%)"
            ),
            (
                f"- BEVHeight on attached subset: n={row['n']} med={row['bevheight_median']:.3f} "
                f"rmse={row['bevheight_rmse']:.3f} p90={row['bevheight_p90']:.3f} "
                f"n>5m={row['bevheight_above5']} ({100 * row['bevheight_frac5']:.3f}%)"
            ),
            f"- full-set HOTA HSG: {row['hsg_hota4']:.1f}",
            f"- full-set HOTA HSG-SRL seed 0: {row['srl_hota4']:.1f}",
            f"- full-set HOTA BEVHeight mixed replay: {row['bevheight_hota4']:.1f}",
            "",
        ]
    path = BH / "score_three.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[supplement] -> {path}")


def main() -> None:
    recording, regional = spatial_tables()
    write_csv(OUT / "spatial_recording.csv", recording)
    write_csv(OUT / "spatial_regions.csv", regional)
    mvc = symmetric_table(recording)
    write_csv(OUT / "mvc.csv", mvc)
    bev = bevheight_rows()
    write_csv(OUT / "bevheight_subset.csv", bev)
    write_markdown(recording, regional, mvc, bev)
    write_score_three(bev)


if __name__ == "__main__":
    main()
