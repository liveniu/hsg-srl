"""Recording list and the constants used by the paper scripts.

Library argparse defaults inside src/pipeline still describe an earlier
residual setup (1.5 m bound, 3 m pair gate, residual weight 0.05). Those
defaults are not the published method. Every paper stage below passes the
values in this file.
"""

from __future__ import annotations

import os
from pathlib import Path

from bootstrap import ARTIFACT_ROOT

GT_BATCH = "l0hn_gt_001"
DET_BATCH = "l0hn_yolo_001"
CLASSES = "2,5,7"
IMGSZ = "384,640"
LUMPI_WINDOW_SEC = 441.5
M6_WINDOW_SEC = 360.0

S0_M = 60.0
ETA = 0.75
R_MAX_M = 1.0
MIN_TRAIN_PAIRS = 32
PAIR_DIST_M = 2.0
TEACHER_KAPPA = 0.20
TEACHER_GAP = 0.05
EPOCHS = 80
BATCH = 256
LR = 1e-3
LAMBDA_MAG = 0.10
LAMBDA_MEAN = 1.0
LAMBDA_PAIR = 0.50
LAMBDA_TEMP = 0.02
TEMPORAL_REUSE = 512
SEEDS = (0, 1, 2)
BOOT_DRAWS = 1000
BOOT_SEED = 20260921

# Final tracking replay. These flags are passed on the command line and do
# not read configs/sites/*/trajectory_merge.
PAPER_MERGE_SCORE = 0.70
PAPER_MERGE_DISTANCE_M = 2.5

WORK = ARTIFACT_ROOT / "data" / "work" / "sites"
EXTERNAL = ARTIFACT_ROOT / "data" / "external"


def site_dir(site: str) -> Path:
    return WORK / site


def geometry_db(site: str) -> Path:
    return site_dir(site) / "geometry.db"


def origin_db(site: str) -> Path:
    return site_dir(site) / "origin.db"


def split_keys(site: str, split: str) -> Path:
    return site_dir(site) / f"{split}_keys.csv"


def verdict_path(site: str) -> Path:
    return site_dir(site) / "reference_verdict.txt"


def match_report(site: str) -> Path:
    return site_dir(site) / "det_gt_match_report.json"


def overlap_path(site: str) -> Path:
    return site_dir(site) / "overlap.json"


def python_exe() -> str:
    return os.environ.get("HSG_PYTHON", "") or "python"


def lumpi_root() -> Path:
    env = os.environ.get("HSGSRL_LUMPI_ROOT", "").strip()
    return Path(env) if env else EXTERNAL / "LUMPI"


def synthehicle_root() -> Path:
    env = os.environ.get("HSGSRL_SYNTHEHICLE_ROOT", "").strip()
    return Path(env) if env else EXTERNAL / "synthehicle_core"


def _lumpi(measurement: int, cameras: tuple[str, ...]) -> dict:
    return {
        "kind": "lumpi",
        "measurement": measurement,
        "cameras": cameras,
        "yaml": f"lumpi_M{measurement}.yaml",
        "scene": f"Town-unused",
    }


def _synth(town: str, n_cams: int) -> dict:
    return {
        "kind": "synth",
        "town": town,
        "scene": f"{town}-O-day",
        "cameras": tuple(f"C{i:02d}" for i in range(1, n_cams + 1)),
        "yaml": f"synth_{town}_O_day.yaml",
    }


SITES: dict[str, dict] = {
    "lumpi_M0": _lumpi(0, ("C08", "C09", "C10")),
    "lumpi_M1": _lumpi(1, ("C08", "C09", "C10")),
    "lumpi_M2": _lumpi(2, ("C08", "C09", "C10")),
    "lumpi_M3": _lumpi(3, ("C08", "C09", "C10")),
    "lumpi_M4": _lumpi(4, ("C06", "C07")),
    "lumpi_M5": _lumpi(5, ("C05", "C06", "C07")),
    "lumpi_M6": _lumpi(6, ("C05", "C06", "C07")),
    "synth_Town01_O_day": _synth("Town01", 4),
    "synth_Town02_O_day": _synth("Town02", 3),
    "synth_Town03_O_day": _synth("Town03", 8),
    "synth_Town04_O_day": _synth("Town04", 4),
    "synth_Town05_O_day": _synth("Town05", 6),
}

GROUPS = {
    "LUMPI-pole": ("lumpi_M0", "lumpi_M1"),
    "LUMPI-A": ("lumpi_M2", "lumpi_M3"),
    "LUMPI-B": ("lumpi_M4", "lumpi_M5", "lumpi_M6"),
    "Synth": tuple(name for name in SITES if name.startswith("synth_")),
}

MAX_T = {"lumpi_M6": M6_WINDOW_SEC}
MAX_T.update({name: LUMPI_WINDOW_SEC for name in SITES if name.startswith("lumpi_") and name != "lumpi_M6"})

_LUMPI_25 = round(LUMPI_WINDOW_SEC * 25.012)
_LUMPI_50 = round(LUMPI_WINDOW_SEC * 50.0)
_LUMPI_30 = round(LUMPI_WINDOW_SEC * 30.0)
MAX_FRAMES: dict[tuple[str, str], int] = {
    ("lumpi_M6", "C05"): 10800,
    ("lumpi_M6", "C06"): 10800,
    ("lumpi_M6", "C07"): 10800,
}
for _site in ("lumpi_M0", "lumpi_M1", "lumpi_M2", "lumpi_M3"):
    MAX_FRAMES[(_site, "C08")] = _LUMPI_25
    MAX_FRAMES[(_site, "C09")] = _LUMPI_50
    MAX_FRAMES[(_site, "C10")] = _LUMPI_50
for _cam in SITES["lumpi_M4"]["cameras"]:
    MAX_FRAMES[("lumpi_M4", _cam)] = _LUMPI_30
for _cam in SITES["lumpi_M5"]["cameras"]:
    MAX_FRAMES[("lumpi_M5", _cam)] = _LUMPI_30

OBS_TABLES = (
    "trajectory_observations",
    "trajectory_global_tracks",
    "trajectory_merges",
    "trajectory_runtime_stats",
    "trajectory_camera_config",
    "trajectory_batches",
    "trajectory_batch_stages",
)

BEVHEIGHT_SITES = ("lumpi_M6", "lumpi_M3", "synth_Town05_O_day")
ORIN_SITES = ("lumpi_M6", "synth_Town05_O_day")
