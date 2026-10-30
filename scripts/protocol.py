"""Recording list and the constants used by the paper scripts.

scripts/train_recording.py fits the frozen pairs and does not call
src/pipeline/geometry_guided_residual_mlp.py. That module's train subcommand
is a separate database trainer: its numeric defaults match the published
loss, and it still requires --split-manifest. It rejects a run that changes
the published loss values. Passing those values does not make it the paper fit.
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

# Final tracking replay. These flags are the published association gates and match MatchConfig.
# They are still passed on the command line. A site YAML trajectory_merge
# block is the pseudo-identity rebuild, not this replay, and it replaces a
# flag only when that flag is left at its default.
PAPER_MERGE_SCORE = 0.70
PAPER_MERGE_DISTANCE_M = 2.5
PAPER_MERGE_TIME_GAP_S = 1.2
PAPER_MERGE_MIN_SAMPLES = 5
PAPER_MERGE_SAMPLE_COUNT = 12
PAPER_MERGE_TIME_SIGMA_S = 0.6
PAPER_MERGE_DISTANCE_SIGMA_M = 0.75
PAPER_MERGE_SHAPE_SIGMA = 0.35
PAPER_MERGE_TIME_WEIGHT = 0.25
PAPER_MERGE_DISTANCE_WEIGHT = 0.45
PAPER_MERGE_SHAPE_WEIGHT = 0.30

# Association used only while building pseudo-identities from a fresh
# detection. These are the values in configs/sites/*/trajectory_merge.
# They are not the final-tracking gates. The released geometry databases already contain
# the trajectories produced with these gates.
GEOMETRY_MERGE = {
    "lumpi": {
        "score": 0.70,
        "max_distance_m": 2.5,
        "distance_sigma_m": 1.5,
        "shape_sigma": 0.50,
    },
    "synth": {
        "score": 0.75,
        "max_distance_m": 2.0,
        "distance_sigma_m": 1.0,
        "shape_sigma": 0.35,
    },
    # Ko-PER is a real intersection. Pseudo-identities use the real-data gates.
    # Final tracking still uses paper_merge_args.
    "koper": {
        "score": 0.70,
        "max_distance_m": 2.5,
        "distance_sigma_m": 1.5,
        "shape_sigma": 0.50,
    },
}

# Fresh from-video graphs only. Published scoring reads configs/overlap and
# does not apply this threshold again. The frozen LUMPI and Synthehicle graphs
# were built at 1000 m^2: that cutoff is why Town03 keeps 27 of 28 edges
# (C02-C04 covers 995.58 m^2) and Town04 keeps 5 of 6 (C02-C04 covers
# 538.42 m^2). Ko-PER's only edge covers 579.35 m^2, so its builder threshold
# is 500 m^2. A positive-area rule would add those two Synthehicle edges and
# would not regenerate the frozen graphs.
AREA_MIN_M2 = {
    "lumpi": 1000.0,
    "synth": 1000.0,
    "koper": 500.0,
}


def paper_merge_args() -> list[str]:
    """Published final-tracking flags for a fusion replay."""
    return [
        "--no-stitch",
        "--merge-score-threshold", f"{PAPER_MERGE_SCORE:.2f}",
        "--merge-max-distance", f"{PAPER_MERGE_DISTANCE_M:.1f}",
        "--merge-max-time-gap", f"{PAPER_MERGE_TIME_GAP_S:.1f}",
        "--merge-min-samples", str(PAPER_MERGE_MIN_SAMPLES),
        "--merge-sample-count", str(PAPER_MERGE_SAMPLE_COUNT),
        "--merge-time-sigma", f"{PAPER_MERGE_TIME_SIGMA_S:.2f}",
        "--merge-distance-sigma", f"{PAPER_MERGE_DISTANCE_SIGMA_M:.2f}",
        "--merge-shape-sigma", f"{PAPER_MERGE_SHAPE_SIGMA:.2f}",
        "--merge-time-weight", f"{PAPER_MERGE_TIME_WEIGHT:.2f}",
        "--merge-distance-weight", f"{PAPER_MERGE_DISTANCE_WEIGHT:.2f}",
        "--merge-shape-weight", f"{PAPER_MERGE_SHAPE_WEIGHT:.2f}",
    ]


def geometry_merge_args(kind: str) -> list[str]:
    """Explicit pseudo-identity gates for a from-video geometry rebuild."""
    spec = GEOMETRY_MERGE[kind]
    return [
        "--merge-score-threshold", f"{spec['score']:.2f}",
        "--merge-max-distance", f"{spec['max_distance_m']:.1f}",
        "--merge-distance-sigma", f"{spec['distance_sigma_m']:.2f}",
        "--merge-shape-sigma", f"{spec['shape_sigma']:.2f}",
    ]

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


def koper_root() -> Path:
    env = os.environ.get("HSGSRL_KOPER_ROOT", "").strip()
    return Path(env) if env else EXTERNAL / "ko_per"


def _lumpi(measurement: int, cameras: tuple[str, ...]) -> dict:
    return {
        "kind": "lumpi",
        "measurement": measurement,
        "cameras": cameras,
        "yaml": f"lumpi_M{measurement}.yaml",
        "scene": f"Town-unused",
    }


def _koper() -> dict:
    """Ko-PER Sequence1a-1d as one recording and one chronological split."""
    return {
        "kind": "koper",
        "cameras": ("SK1", "SK4"),
        "yaml": "",
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
    "ko_per": _koper(),
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
    "LUMPI": tuple(name for name in SITES if name.startswith("lumpi_")),
    "Ko-PER": ("ko_per",),
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
ORIN_SITES = ("lumpi_M6", "synth_Town05_O_day", "ko_per")
