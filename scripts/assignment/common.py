"""Read-only paths for the clock-source check and the Table S3 row closure.

Geometry databases, split manifests, and seed-0 checkpoints are read from
this package. Results are written under results/controls/. The matcher is
the package copy of eval_fab2_det_gt_match.py.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ARTIFACT_ROOT / "src" / "evaluation"

for path in (ARTIFACT_ROOT / "src", ARTIFACT_ROOT / "scripts", UPSTREAM):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from protocol import DET_BATCH, GT_BATCH, MAX_T, SITES, geometry_db, match_report, origin_db  # noqa: E402

RESULTS = ARTIFACT_ROOT / "results" / "controls"
SITES_CLOCK = ("lumpi_M2", "lumpi_M6", "ko_per")
BOOT_DRAWS = 1000
BOOT_SEED = 20260921

DT_MIN = -1.5
DT_MAX = 1.5
DT_STEP = 0.05
DT_GATE_M = 4.0
DT_SAMPLE_TRACKS = 60
GATE_M = 3.0
MARGIN_M = 0.75
MIN_OVERLAP = 8
MAX_GAP_S = 0.12
SANITY_M = 10.0
MIN_AFFINE_SAMPLES = 500
GT_CLASSES = {2, 5, 7}


def connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def ensure(*parts: str) -> Path:
    path = RESULTS.joinpath(*parts) if parts else RESULTS
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload) -> None:
    def convert(obj):
        if isinstance(obj, np.floating):
            value = float(obj)
            return value if np.isfinite(value) else None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"not serializable: {type(obj)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=convert) + "\n", encoding="utf-8")


def label_of(site: str) -> str:
    if site == "ko_per":
        return "Ko-PER"
    return site.replace("lumpi_", "")


def test_start(site: str) -> float:
    path = ARTIFACT_ROOT / "data" / "work" / "sites" / site / "dataset_split_manifest.csv"
    import csv
    stamps = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                stamps.append(float(row["timestamp"]))
    if not stamps:
        raise RuntimeError(f"{site} has no test timestamps")
    return min(stamps)


def load_test_keys(site: str) -> set[tuple[str, int, int]]:
    import csv
    path = ARTIFACT_ROOT / "data" / "work" / "sites" / site / "test_keys.csv"
    keys = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            cam = row.get("camera_id") or row.get("camera_name") or ""
            keys.add((cam, int(row["local_track_id"]), int(row["frame_index"])))
    return keys


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def median(values: np.ndarray) -> float:
    return float(np.median(values))
