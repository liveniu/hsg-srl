#!/usr/bin/env python3
"""Write the tracking-source table from files inside this repository.

Paths in the source column are relative to the artifact root. The table is
results/recorded/tracking_decomposition_recording.csv.
"""

from __future__ import annotations

import csv
import hashlib
import json

from bootstrap import ARTIFACT_ROOT

RECORDED = ARTIFACT_ROOT / "results" / "recorded"
BINS = RECORDED / "bins"
FIELDS = ["site", "method", "seed", "gate", "hota", "deta", "assa", "loca", "source"]


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bin_index() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(BINS.glob("*.npz")):
        found.setdefault(file_hash(path), []).append(f"results/recorded/bins/{path.name}")
    return found


def choose_bin(paths: list[str], method: str, seed: str) -> str:
    if method == "hsg":
        preferred = [path for path in paths if path.endswith("_geo25.npz")]
    elif method == "overlap":
        preferred = [path for path in paths if path.endswith(f"_overlap_s{seed}.npz")]
    else:
        preferred = [path for path in paths if "_overlap_" not in path and not path.endswith("_geo25.npz")]
    return preferred[0] if preferred else paths[0]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def from_csv(path, method: str, hashes: dict[str, str]) -> list[dict]:
    rows = []
    for row in read_csv(path):
        if method == "hsg" and row["method"] != "Geo":
            continue
        if method != "hsg" and row["method"] != "HSG-SRL":
            continue
        digest = (row.get("input_sha256") or "").strip()
        matches = hashes.get(digest) or []
        if not matches:
            raise SystemExit(f"{path.name} {row['site']} gate {row['gate']} has no bin in this repository")
        source = choose_bin(matches, method, str(row.get("seed") or "0"))
        rows.append({
            "site": row["site"],
            "method": method,
            "seed": 0 if method == "hsg" else int(row["seed"]),
            "gate": float(row["gate"]),
            "hota": 100 * float(row["hota"]),
            "deta": 100 * float(row["deta"]),
            "assa": 100 * float(row["assa"]),
            "loca": 100 * float(row["loca"]),
            "source": source,
        })
    return rows


def from_koper_full(hashes: dict[str, str]) -> list[dict]:
    rows = []
    for seed in (0, 1, 2):
        path = RECORDED / "hota_sources" / f"ko_per__full__s{seed}.json"
        blob = json.loads(path.read_text(encoding="utf-8"))
        source = f"results/recorded/hota_sources/{path.name}"
        for item in blob["rows"]:
            rows.append({
                "site": "ko_per",
                "method": "full_support",
                "seed": seed,
                "gate": float(item["gate"]),
                "hota": 100 * float(item["hota"]),
                "deta": 100 * float(item["deta"]),
                "assa": 100 * float(item["assa"]),
                "loca": 100 * float(item["loca"]),
                "source": source,
            })
    return rows


def main() -> None:
    hashes = bin_index()
    rows = []
    rows.extend(from_csv(RECORDED / "identity_hota25.csv", "hsg", hashes))
    rows.extend(from_csv(RECORDED / "identity_hota25_support.csv", "full_support", hashes))
    rows.extend(from_koper_full(hashes))
    rows.extend(from_csv(RECORDED / "identity_hota25_overlap.csv", "overlap", hashes))
    dest = RECORDED / "tracking_decomposition_recording.csv"
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    outside = [row["source"] for row in rows if not row["source"].startswith("results/")]
    if outside:
        raise SystemExit(f"sources outside this repository: {outside[:5]}")
    print(f"[sources] {len(rows)} rows -> {dest}")


if __name__ == "__main__":
    main()
