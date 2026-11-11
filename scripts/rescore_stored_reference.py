#!/usr/bin/env python3
"""Put saved TrackEval bins on the stored footprint centers.

Older bins for recordings whose verdict starts with REFERENCE OFFSET stored
ground truth after that audit translation. Predictions stay. This script
subtracts the audit vector, rewrites those bins, and updates the three
published HOTA tables. A second run checks that the bins already match the
tables and does not shift them again.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

from bootstrap import ARTIFACT_ROOT
from db_access import audit_bias
from evaluation.hota_trackeval import hota_trackeval
from protocol import SITES

RECORDED = ARTIFACT_ROOT / "results" / "recorded"
BINS = RECORDED / "bins"
MARKER = RECORDED / "SCORING_REFERENCE.txt"
TABLES = (
    "identity_hota25.csv",
    "identity_hota25_overlap.csv",
    "identity_hota25_support.csv",
)
FIELDS = ("hota", "deta", "assa", "loca", "tp", "fp", "fn")
MARKER_TEXT = (
    "Published scoring uses the stored footprint centers.\n"
    "reference_verdict.txt is an audit record and is not added to ground truth.\n"
)


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack(blob, prefix: str) -> dict:
    times, identities, xy = (blob[f"{prefix}_{suffix}"] for suffix in ("t", "id", "xy"))
    out: dict = {}
    for timestamp, identity, point in zip(times, identities, xy):
        out.setdefault(int(timestamp), {})[int(identity)] = (float(point[0]), float(point[1]))
    return out


def shift_gt(gt_bins: dict, bias: np.ndarray) -> dict:
    shifted = {}
    for timestamp, frame in gt_bins.items():
        shifted[timestamp] = {
            identity: (point[0] - float(bias[0]), point[1] - float(bias[1]))
            for identity, point in frame.items()
        }
    return shifted


def score_blob(blob, bias: np.ndarray | None) -> dict[float, dict]:
    gt = unpack(blob, "gt")
    pred = unpack(blob, "pr")
    if bias is not None:
        gt = shift_gt(gt, bias)
    rows = hota_trackeval(gt, pred)
    bev = float(np.mean([row["hota"] for row in rows]))
    scored = {}
    for row in rows:
        row["bev_hota"] = bev
        scored[float(row["gate"])] = row
    return scored


def gap(scored: dict, row: dict) -> float:
    actual = scored[float(row["gate"])]
    return max(abs(float(actual[field]) - float(row[field])) for field in FIELDS)


def affected() -> dict[str, np.ndarray]:
    found = {}
    for site in SITES:
        bias = audit_bias(site)
        if float(np.linalg.norm(bias)) > 0.0:
            found[site] = bias
    if not found:
        raise SystemExit("no recording has an audit translation to remove")
    return found


def main() -> None:
    sites = affected()
    print("[reference] audit translation present for " + ", ".join(sites), flush=True)
    tables = {name: read_csv(RECORDED / name) for name in TABLES}
    by_hash = {file_sha256(path): path for path in BINS.glob("*.npz")}
    needed: dict[str, str] = {}
    for _name, (fields, rows) in tables.items():
        del fields
        for row in rows:
            if row["site"] not in sites:
                continue
            digest = (row.get("input_sha256") or "").strip()
            if not digest:
                raise SystemExit(f"{row['site']} {row['method']} has no bin hash")
            if digest not in by_hash:
                raise SystemExit(f"missing bin {digest} for {row['site']}")
            previous = needed.get(digest)
            if previous not in (None, row["site"]):
                raise SystemExit(f"bin {digest} is shared by {previous} and {row['site']}")
            needed[digest] = row["site"]

    migrated = {}
    if MARKER.exists():
        for digest, site in needed.items():
            path = by_hash[digest]
            with np.load(path) as blob:
                stored = score_blob(blob, None)
            rows = [
                row for _name, (_fields, table) in tables.items()
                for row in table if (row.get("input_sha256") or "").strip() == digest
            ]
            stored_gap = max(gap(stored, row) for row in rows)
            if stored_gap > 1e-8:
                raise SystemExit(f"{path.name} disagrees with the tables by {stored_gap:.3e}")
            print(f"[reference] {path.name} already matches the tables", flush=True)
        print("[reference] bins already use stored footprint centers")
        return

    for digest, site in needed.items():
        path = by_hash[digest]
        with np.load(path) as blob:
            stored = score_blob(blob, None)
            arrays = {key: np.array(blob[key]) for key in blob.files}
        rows = [
            row for _name, (_fields, table) in tables.items()
            for row in table if (row.get("input_sha256") or "").strip() == digest
        ]
        stored_gap = max(gap(stored, row) for row in rows)
        if stored_gap > 1e-8:
            raise SystemExit(
                f"{path.name} disagrees with the current tables by {stored_gap:.3e}; "
                "refusing to shift an unrecognized bin"
            )
        shifted = score_blob(arrays, sites[site])
        migrated[digest] = {"path": path, "arrays": arrays, "bias": sites[site], "scores": shifted}
        print(f"[reference] {path.name} will drop the audit translation", flush=True)

    new_hash = {}
    for digest, item in migrated.items():
        path = item["path"]
        arrays = item["arrays"]
        bias = item["bias"]
        arrays["gt_xy"] = np.asarray(arrays["gt_xy"], dtype=np.float64) - bias.reshape(1, 2)
        np.savez_compressed(path, **arrays)
        rewritten = file_sha256(path)
        with np.load(path) as blob:
            check = score_blob(blob, None)
        worst = 0.0
        for gate, row in item["scores"].items():
            for field in FIELDS:
                worst = max(worst, abs(float(check[gate][field]) - float(row[field])))
        if worst > 1e-8:
            raise SystemExit(f"{path.name} rewrite did not reproduce the shifted score ({worst})")
        new_hash[digest] = (rewritten, item["scores"])
        print(f"[reference] rewrote {path.name}", flush=True)

    for name, (fields, rows) in tables.items():
        for row in rows:
            digest = (row.get("input_sha256") or "").strip()
            if digest not in new_hash:
                continue
            rewritten, scores = new_hash[digest]
            updated = scores[float(row["gate"])]
            for field in FIELDS + ("bev_hota",):
                row[field] = updated[field]
            row["input_sha256"] = rewritten
        write_csv(RECORDED / name, fields, rows)
        print(f"[reference] updated {name}", flush=True)
    MARKER.write_text(MARKER_TEXT, encoding="utf-8")
    print(f"[reference] shifted {len(migrated)} bins")


if __name__ == "__main__":
    main()
