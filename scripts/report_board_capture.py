#!/usr/bin/env python3
"""Print the stored board-fit residuals.

The six scenes are optional local-frame captures. They are not part of the
LUMPI or Synthehicle geometry databases. This command does not read images.
Re-fitting from the stills is documented in docs/REPRODUCTION.md.
"""

from __future__ import annotations

import json
from collections import OrderedDict

from bootstrap import ARTIFACT_ROOT

RESIDUALS = ARTIFACT_ROOT / "results" / "board_capture" / "fit_residuals.json"
SCALES = {
    "xizi_scene_01": 15,
    "xizi_scene_02": 15,
    "xizi_scene_03": 15,
    "xizi_scene_04": 20,
    "xizi_scene_05": 15,
    "xizi_scene_07": 40,
}


def main() -> None:
    rows = json.loads(RESIDUALS.read_text(encoding="utf-8"))
    by_scene: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        if row.get("record_type") != "camera_refine" or row.get("status") != "ok":
            continue
        by_scene.setdefault(row["scene_id"], []).append(row)
    missing = [scene for scene in SCALES if scene not in by_scene]
    if missing:
        raise SystemExit(f"missing camera fits: {missing}")
    print("Board-fit RMS before per-camera refinement, in BEV-grid pixels.")
    print("Scene ids xizi_scene_* in the JSON are these six captures. The fits use manual corners, not ruler measurements.")
    print()
    print("| Scene | Cameras | Camera graph depth | BEV pixels per meter | Fit RMS before | Fit RMS after |")
    print("| --- | --- | --- | ---: | --- | --- |")
    for scene, scale in SCALES.items():
        cameras = by_scene[scene]
        label = scene.replace("xizi_scene_", "")
        names = ", ".join(row["name"] for row in cameras)
        depths = ", ".join(str(int(row["board_or_camera_hop"])) for row in cameras)
        before = ", ".join(f"{float(row['reproj_rms_before_px']):.3f}" for row in cameras)
        after = ", ".join(f"{float(row['reproj_rms_after_px']):.3f}" for row in cameras)
        print(f"| {label} | {names} | {depths} | {scale} | {before} | {after} |")


if __name__ == "__main__":
    main()
