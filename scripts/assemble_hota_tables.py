#!/usr/bin/env python3
"""Build the published overlap-gated HOTA table from vendored score files.

results/recorded/hota_sources/<site>__overlap__s<seed>.json and
ko_per__geo__s0.json are the association replays used by the manuscript.
M0 and M1 have no residual model, so their overlap rows copy geometry-only
HSG. This does not rerun fusion.

Bin hashes are not in the score files. scripts/export_overlap_bins.py fills
them after a replay from the geometry databases. This command keeps an
existing hash when the rebuilt row has the same scores, so rerunning it does
not disconnect the rows from results/recorded/bins.
"""

from __future__ import annotations

import csv
import json

from bootstrap import ARTIFACT_ROOT
from protocol import SEEDS, SITES

RECORDED = ARTIFACT_ROOT / "results" / "recorded"
SOURCES = RECORDED / "hota_sources"
FIELDS = [
    "site", "method", "seed", "gate", "alpha", "hota", "bev_hota",
    "deta", "assa", "loca", "tp", "fp", "fn", "input_sha256",
]
NO_MODEL = {"lumpi_M0", "lumpi_M1"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict) -> tuple[str, str, str, str]:
    seed = row.get("seed")
    seed_text = "" if seed in ("", None) else str(int(float(seed)))
    return (row["site"], row["method"], seed_text, f"{float(row['gate']):g}")


def score_text(row: dict) -> tuple[str, ...]:
    return tuple(f"{float(row[name]):.12g}" for name in ("hota", "deta", "assa", "loca")) + tuple(
        str(int(float(row[name]))) for name in ("tp", "fp", "fn")
    )


def existing_hashes(path) -> dict[tuple, tuple[str, tuple[str, ...]]]:
    """Map each stored row to its bin hash and the scores that hash was checked against."""
    if not path.exists():
        return {}
    out = {}
    for row in read_csv(path):
        digest = (row.get("input_sha256") or "").strip()
        if digest:
            out[row_key(row)] = (digest, score_text(row))
    return out


def carry_hashes(rows: list[dict], stored: dict) -> int:
    kept = 0
    for row in rows:
        found = stored.get(row_key(row))
        if found and found[1] == score_text(row):
            row["input_sha256"] = found[0]
            kept += 1
    return kept


def from_json(path, method: str) -> list[dict]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in blob["rows"]:
        rows.append({
            "site": blob["site"],
            "method": method,
            "seed": "" if method == "Geo" else int(blob["seed"]),
            "gate": item["gate"],
            "alpha": item["alpha"],
            "hota": item["hota"],
            "bev_hota": item.get("bev_hota", ""),
            "deta": item["deta"],
            "assa": item["assa"],
            "loca": item["loca"],
            "tp": item["tp"],
            "fp": item["fp"],
            "fn": item["fn"],
            "input_sha256": "",
        })
    return rows


def main() -> None:
    geo_path = RECORDED / "identity_hota25.csv"
    overlap_path = RECORDED / "identity_hota25_overlap.csv"
    stored_geo = existing_hashes(geo_path)
    stored_overlap = existing_hashes(overlap_path)

    geo_rows = [row for row in read_csv(geo_path) if row["site"] != "ko_per"]
    koper_geo = from_json(SOURCES / "ko_per__geo__s0.json", "Geo")
    kept_geo = carry_hashes(koper_geo, stored_geo)
    geo_rows.extend(koper_geo)
    write_csv(geo_path, geo_rows)

    overlap = []
    for site in SITES:
        if site in NO_MODEL:
            base = [row for row in geo_rows if row["site"] == site and row["method"] == "Geo"]
            if len(base) != 4:
                raise SystemExit(f"{site} geometry rows={len(base)}")
            for seed in SEEDS:
                for row in base:
                    copied = dict(row)
                    copied["method"] = "HSG-SRL"
                    copied["seed"] = seed
                    overlap.append(copied)
            continue
        for seed in SEEDS:
            path = SOURCES / f"{site}__overlap__s{seed}.json"
            if not path.exists():
                raise SystemExit(f"missing {path}")
            got = from_json(path, "HSG-SRL")
            if len(got) != 4:
                raise SystemExit(f"{path} has {len(got)} gates")
            overlap.extend(got)
    kept_overlap = carry_hashes(overlap, stored_overlap)
    write_csv(overlap_path, overlap)
    print(f"[hota] geo={len(geo_rows)} overlap={len(overlap)} "
          f"hashes kept: geo {kept_geo}/{len(koper_geo)}, overlap {kept_overlap}/{len(overlap)}")
    missing = [row for row in overlap if not row["input_sha256"]]
    if missing:
        print(f"[hota] {len(missing)} overlap rows have no bin hash; run scripts/export_overlap_bins.py "
              "from the geometry databases to fill them")


if __name__ == "__main__":
    main()
