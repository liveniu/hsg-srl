#!/usr/bin/env python3
"""Run the published chain for one recording, or print the stage order.

Stages, in order:

1. import_recordings.py
2. detect_and_merge.py
3. attach_ground_truth.py
4. audit_reference.py
5. make_time_split.py
6. build_overlap.py
7. freeze_pairs.py
8. train_recording.py for seeds 0, 1, and 2
9. eval_fixed_anchors.py
10. eval_localization.py
11. eval_tracking.py

Licensed videos stay outside this package. This command runs YOLO26l and
builds a new geometry database. Published tables use the released databases
at https://doi.org/10.5281/zenodo.23008421, because a new detection does not
match that detection set.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from bootstrap import ARTIFACT_ROOT
from protocol import SITES


def run(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(ARTIFACT_ROOT / "scripts" / script), *args]
    print("+", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(ARTIFACT_ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, choices=sorted(SITES))
    parser.add_argument("--model", default="")
    parser.add_argument("--from-stage", default="import")
    args = parser.parse_args()
    stages = ["import", "detect", "attach", "audit", "split", "overlap", "freeze", "train", "anchors", "localize", "track"]
    if args.from_stage not in stages:
        raise SystemExit(f"--from-stage must be one of {stages}")
    start = stages.index(args.from_stage)
    site = args.site
    if start <= 0:
        run("import_recordings.py", [site])
    if start <= 1:
        extra = ["--site", site]
        if args.model:
            extra += ["--model", args.model]
        run("detect_and_merge.py", extra)
    if start <= 2:
        run("attach_ground_truth.py", [site])
    if start <= 3:
        run("audit_reference.py", [site])
    if start <= 4:
        run("make_time_split.py", [site])
    if start <= 5:
        run("build_overlap.py", [site])
    if start <= 6:
        run("freeze_pairs.py", [site])
    if start <= 7:
        for seed in (0, 1, 2):
            run("train_recording.py", ["--site", site, "--seed", str(seed)])
    if start <= 8:
        run("eval_fixed_anchors.py", [site])
    if start <= 9:
        run("eval_localization.py", ["--site", site])
    if start <= 10:
        run("eval_tracking.py", ["--site", site])
        for seed in (0, 1, 2):
            run("eval_tracking.py", ["--site", site, "--seed", str(seed)])


if __name__ == "__main__":
    main()
