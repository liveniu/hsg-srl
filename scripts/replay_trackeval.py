#!/usr/bin/env python3
"""Replay saved 10 Hz bins through the included HOTA implementation.

This checks the published tracking tables. It does not retrain a model or
choose a camera mask.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys

import numpy as np

from bootstrap import ARTIFACT_ROOT, SRC_ROOT

sys.path.insert(0, str(SRC_ROOT))
from evaluation.hota_trackeval import _self_check, hota_trackeval  # noqa: E402

RECORDED = ARTIFACT_ROOT / "results" / "recorded"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def unpack(blob, prefix):
    times, ids, xy = (blob[f"{prefix}_{suffix}"] for suffix in ("t", "id", "xy"))
    out = {}
    for timestamp, identity, point in zip(times, ids, xy):
        out.setdefault(int(timestamp), {})[int(identity)] = tuple(point)
    return out


def main() -> None:
    _self_check()
    bins = list((RECORDED / "bins").glob("*.npz"))
    if not bins:
        raise SystemExit("results/recorded/bins is empty; the scoring inputs were not exported")
    by_hash = {hashlib.sha256(path.read_bytes()).hexdigest(): path for path in bins}
    rows = read_csv(RECORDED / "identity_hota25.csv")
    rows += read_csv(RECORDED / "identity_hota25_support.csv")
    maximum = 0.0
    checked = 0
    for row in rows:
        path = by_hash[row["input_sha256"]]
        with np.load(path) as blob:
            gt, pred = unpack(blob, "gt"), unpack(blob, "pr")
        scores = {item["gate"]: item for item in hota_trackeval(gt, pred)}
        actual = scores[float(row["gate"])]
        for field in ("hota", "deta", "assa", "loca", "tp", "fp", "fn"):
            maximum = max(maximum, abs(actual[field] - float(row[field])))
        checked += 1
    result = {"rows_checked": checked, "max_abs_difference": maximum}
    if maximum > 1e-12:
        raise SystemExit(json.dumps(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
