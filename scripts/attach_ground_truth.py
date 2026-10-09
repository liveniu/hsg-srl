#!/usr/bin/env python3
"""Attach dataset ground truth to detector tracks on the video clock.

The matcher is the same detector-to-GT tool used to freeze
outside_reference_x/y before residual fitting. It searches an affine clock
offset on the stored baseline world coordinates. Later HSG and residual
variants do not recompute the attachment.
"""

from __future__ import annotations

import subprocess
import sys

from bootstrap import ARTIFACT_ROOT, SRC_ROOT
from protocol import DET_BATCH, GT_BATCH, SITES, geometry_db, match_report, origin_db, site_dir


def main() -> None:
    sites = sys.argv[1:] or list(SITES)
    script = SRC_ROOT / "evaluation" / "eval_fab2_det_gt_match.py"
    for site in sites:
        if site not in SITES:
            raise SystemExit(f"unknown recording {site}")
        out = site_dir(site)
        out.mkdir(parents=True, exist_ok=True)
        cameras = ",".join(SITES[site]["cameras"])
        cmd = [
            sys.executable, str(script),
            "--det-db", str(geometry_db(site)),
            "--det-batch", DET_BATCH,
            "--det-scene", site,
            "--gt-db", str(origin_db(site)),
            "--gt-batch", GT_BATCH,
            "--cameras", cameras,
            "--dt-mode", "affine",
            "--write-outside-reference",
            "--out-dir", str(out),
        ]
        print(f"[attach] {site}", flush=True)
        completed = subprocess.run(cmd, cwd=str(ARTIFACT_ROOT))
        if completed.returncode != 0:
            raise SystemExit(f"attach {site} failed")
        if not match_report(site).exists():
            raise SystemExit(f"attach {site} did not write {match_report(site).name}")


if __name__ == "__main__":
    main()
