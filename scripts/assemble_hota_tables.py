#!/usr/bin/env python3
"""Build the published overlap-gated HOTA table from vendored score files.

results/recorded/hota_sources/<site>__overlap__s<seed>.json and
ko_per__geo__s0.json are the association replays used by the manuscript.
M0 and M1 have no residual model, so their overlap rows copy geometry-only
HSG. This does not rerun fusion. Saved 10 Hz bins still check the geometry
table and the camera-support ablation; they are not the overlap-gated rows.
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
    geo_rows = [row for row in read_csv(geo_path) if row["site"] != "ko_per"]
    geo_rows.extend(from_json(SOURCES / "ko_per__geo__s0.json", "Geo"))
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
    write_csv(RECORDED / "identity_hota25_overlap.csv", overlap)
    print(f"[hota] geo={len(geo_rows)} overlap={len(overlap)}")


if __name__ == "__main__":
    main()
