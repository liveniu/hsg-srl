#!/usr/bin/env python3
"""Import one downloaded recording into an origin SQLite database.

LUMPI uses the dataset homography (--h-source model). Synthehicle daytime
towns use the visible tight box (--bbox-source gt_tight) and the ground
height fitted from the 3-D boxes (--z-ground auto). Ko-PER Sequence1a-1d
are concatenated on one camera-clock epoch, and the front-bumper bottom
center is shifted backward by half the vehicle length. Official test clips
are not imported: they do not support this protocol.
"""

from __future__ import annotations

import subprocess
import sys

from bootstrap import ARTIFACT_ROOT, SRC_ROOT
from protocol import GT_BATCH, SITES, koper_root, lumpi_root, origin_db, synthehicle_root


def command(site: str) -> list[str]:
    spec = SITES[site]
    db = origin_db(site)
    db.parent.mkdir(parents=True, exist_ok=True)
    if spec["kind"] == "koper":
        script = SRC_ROOT / "importers" / "import_koper.py"
        return [
            sys.executable, str(script), "import",
            "--all", "--together",
            "--db", str(db),
            "--batch", GT_BATCH,
            "--h-source", "model",
            "--data-root", str(koper_root()),
        ]
    if spec["kind"] == "lumpi":
        script = SRC_ROOT / "importers" / "import_lumpi.py"
        return [
            sys.executable, str(script), "import",
            "--measurement", str(spec["measurement"]),
            "--db", str(db),
            "--batch", GT_BATCH,
            "--h-source", "model",
            "--data-root", str(lumpi_root()),
        ]
    script = SRC_ROOT / "importers" / "import_synthehicle.py"
    return [
        sys.executable, str(script), "import",
        "--scene", spec["scene"],
        "--db", str(db),
        "--batch", GT_BATCH,
        "--h-source", "model",
        "--bbox-source", "gt_tight",
        "--z-ground", "auto",
        "--data-root", str(synthehicle_root()),
    ]


def main() -> None:
    sites = sys.argv[1:] or list(SITES)
    for site in sites:
        if site not in SITES:
            raise SystemExit(f"unknown recording {site}")
        db = origin_db(site)
        if db.exists():
            print(f"[import] {site}: {db.name} exists, skipped")
            continue
        cmd = command(site)
        print(f"[import] {site}", flush=True)
        completed = subprocess.run(cmd, cwd=str(ARTIFACT_ROOT))
        if completed.returncode != 0:
            raise SystemExit(f"import {site} failed with code {completed.returncode}")


if __name__ == "__main__":
    main()
