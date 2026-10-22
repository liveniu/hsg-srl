#!/usr/bin/env python3
"""Replay overlap-gated tracking and store the 10 Hz bins used by TrackEval.

The published scores stay in identity_hota25_overlap.csv. This command fills
their bin hashes after checking that the new replay matches those scores.
Ko-PER geometry-only rows in identity_hota25.csv are filled the same way.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys

from bootstrap import ARTIFACT_ROOT
from protocol import SEEDS, SITES

RECORDED = ARTIFACT_ROOT / "results" / "recorded"
BINS = RECORDED / "bins"
FIELDS = [
    "site", "method", "seed", "gate", "alpha", "hota", "bev_hota",
    "deta", "assa", "loca", "tp", "fp", "fn", "input_sha256",
]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay(site: str, seed: int | None) -> tuple[str, dict[float, float]]:
    cmd = [sys.executable, "-u", str(ARTIFACT_ROOT / "scripts" / "eval_tracking.py"), "--site", site, "--save-bins"]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    completed = subprocess.run(cmd, cwd=str(ARTIFACT_ROOT))
    if completed.returncode != 0:
        raise SystemExit(f"replay {site} seed {seed} failed")
    name = f"{site}_geo25.npz" if seed is None else f"{site}_overlap_s{seed}.npz"
    manifest = json.loads((BINS / name).with_suffix(".json").read_text(encoding="utf-8"))
    scores = {float(gate): float(value) for gate, value in manifest["hota"].items()}
    digest = str(manifest["sha256"])
    if file_hash(BINS / name) != digest:
        raise SystemExit(f"{name} hash does not match its manifest")
    print(f"[export] {site} seed {seed} {digest[:12]}", flush=True)
    return digest, scores


def assign(rows, site: str, method: str, seed: str, digest: str, scores: dict[float, float]) -> None:
    found = 0
    for row in rows:
        if row["site"] != site or row["method"] != method:
            continue
        if method != "Geo" and str(row["seed"]) != str(seed):
            continue
        gate = float(row["gate"])
        if abs(float(row["hota"]) - scores[gate]) > 1e-8:
            raise SystemExit(
                f"{site} {method} seed {seed} gate {gate}: replay {scores[gate]} != published {row['hota']}"
            )
        row["input_sha256"] = digest
        found += 1
    if found != 4:
        raise SystemExit(f"{site} {method} seed {seed}: updated {found} gates")


def main() -> None:
    overlap = read_csv(RECORDED / "identity_hota25_overlap.csv")
    geo = read_csv(RECORDED / "identity_hota25.csv")
    for site in SITES:
        for seed in SEEDS:
            name = f"{site}_overlap_s{seed}.npz"
            path = BINS / name
            digest = file_hash(path) if path.exists() else ""
            current = {
                row["input_sha256"]
                for row in overlap
                if row["site"] == site and str(row["seed"]) == str(seed)
            }
            if digest and current == {digest}:
                print(f"[export] reuse {name}", flush=True)
                continue
            digest, scores = replay(site, seed)
            assign(overlap, site, "HSG-SRL", str(seed), digest, scores)
            write_csv(RECORDED / "identity_hota25_overlap.csv", overlap)
    name = "ko_per_geo25.npz"
    path = BINS / name
    digest = file_hash(path) if path.exists() else ""
    current = {
        row["input_sha256"]
        for row in geo
        if row["site"] == "ko_per" and row["method"] == "Geo"
    }
    if not (digest and current == {digest}):
        digest, scores = replay("ko_per", None)
        assign(geo, "ko_per", "Geo", "", digest, scores)
        write_csv(RECORDED / "identity_hota25.csv", geo)
    print("[export] overlap bins and Ko-PER geometry bins are hashed")


if __name__ == "__main__":
    main()
