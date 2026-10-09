#!/usr/bin/env python3
"""Chronological 60/10/30 split on unique timestamps.

Writes dataset_split_manifest.csv and train/val/test key lists. Ground-truth
coordinates are not the split key.
"""

from __future__ import annotations

import csv
import subprocess
import sys

from bootstrap import ARTIFACT_ROOT, SRC_ROOT
from protocol import DET_BATCH, SITES, geometry_db, site_dir


def export_keys(site: str) -> None:
    manifest = site_dir(site) / "manifest" / "dataset_split_manifest.csv"
    shutil_copy = manifest
    published = site_dir(site) / "dataset_split_manifest.csv"
    published.write_text(shutil_copy.read_text(encoding="utf-8"), encoding="utf-8")
    counts = {"train": 0, "val": 0, "test": 0}
    handles = {}
    writers = {}
    for split in counts:
        path = site_dir(site) / f"{split}_keys.csv"
        handle = path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=("camera_id", "local_track_id", "frame_index"))
        writer.writeheader()
        handles[split] = handle
        writers[split] = writer
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row.get("split")
            if split not in writers:
                continue
            if row.get("batch_id") and row["batch_id"] != DET_BATCH:
                continue
            writers[split].writerow({
                "camera_id": row["camera_id"],
                "local_track_id": row["local_track_id"],
                "frame_index": row["frame_index"],
            })
            counts[split] += 1
    for handle in handles.values():
        handle.close()
    print(f"[split] {site} keys {counts}", flush=True)


def main() -> None:
    sites = sys.argv[1:] or list(SITES)
    script = SRC_ROOT / "evaluation" / "fab2_manifest.py"
    for site in sites:
        if site not in SITES:
            raise SystemExit(f"unknown recording {site}")
        out = site_dir(site)
        pairs = out / "det_gt_pairs.csv"
        if not pairs.exists():
            raise SystemExit(f"run attach_ground_truth.py before the split ({site})")
        cmd = [
            sys.executable, str(script),
            "--db", str(geometry_db(site)),
            "--batch", DET_BATCH,
            "--scene", site,
            "--gt-id-csv", str(pairs),
            "--split-mode", "time",
            "--train-frac", "0.6",
            "--val-frac", "0.1",
            "--out-dir", str(out),
        ]
        print(f"[split] {site}", flush=True)
        completed = subprocess.run(cmd, cwd=str(ARTIFACT_ROOT))
        if completed.returncode != 0:
            raise SystemExit(f"split {site} failed")
        export_keys(site)


if __name__ == "__main__":
    main()
