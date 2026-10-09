"""Site layout for the three-recording BEVHeight reference pack.

Paths point at this artifact. The notes from the older four-site packer are
not used.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[2]
WORK = ARTIFACT / "data" / "work" / "sites"
EXTERNAL = ARTIFACT / "data" / "external"
PACK_WORK = Path(__file__).resolve().parent / "step2_pack" / "work"
PACK_OUT = Path(__file__).resolve().parent / "step2_pack" / "outputs"

CARLA_TO_KITTI = (
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


@dataclass(frozen=True)
class SiteSpec:
    site: str
    kind: str
    dataset: str
    cameras: tuple[str, ...]
    table_cameras: tuple[str, ...]
    measurement: int | None = None
    scene_name: str | None = None
    data_rel: str | None = None
    calib_rel: str | None = None
    z_ground: float = 0.0
    image_wh: tuple[int, int] | None = None


SITE_SPECS = {
    "lumpi_M6": SiteSpec(
        "lumpi_M6", "lumpi", "LUMPI", ("C05", "C06", "C07"), ("C05", "C06", "C07"),
        measurement=6, z_ground=-2.0,
    ),
    "lumpi_M3": SiteSpec(
        "lumpi_M3", "lumpi", "LUMPI", ("C08", "C09", "C10"), ("C08", "C09", "C10"),
        measurement=3, z_ground=-2.0,
    ),
    "synth_Town05_O_day": SiteSpec(
        "synth_Town05_O_day", "synth", "Synthehicle",
        ("C01", "C02", "C03", "C04", "C05", "C06"),
        ("C01", "C02", "C03", "C04", "C05", "C06"),
        scene_name="Town05-O-day",
        data_rel="data/external/synthehicle_core/train/Town05-O-day",
        calib_rel="data/external/synthehicle_core/calibration/overlapping/Town05",
        image_wh=(1920, 1080),
    ),
}
ALL_SITES = tuple(SITE_SPECS)


def require_freeze() -> None:
    missing = [site for site in SITE_SPECS if not (WORK / site / "origin.db").exists()]
    if missing:
        raise SystemExit(f"import the recordings before packing BEVHeight: {missing}")


def assert_write(path: Path, allowed_roots: tuple[Path, ...]) -> Path:
    dest = path.resolve()
    for root in allowed_roots:
        root = root.resolve()
        if dest == root or root in dest.parents:
            return dest
    raise RuntimeError(f"refusing to write outside the pack directory: {dest}")


def video_path(spec: SiteSpec, camera: str) -> Path:
    if spec.kind == "lumpi":
        return (
            EXTERNAL / "LUMPI" / f"Measurement{spec.measurement}" / "cam"
            / str(int(camera[1:])) / "video.mp4"
        )
    if spec.data_rel is None:
        raise ValueError(f"missing data path for {spec.site}")
    return ARTIFACT / spec.data_rel / camera / "out_rgb" / "video.mp4"


def split_manifest(site: str) -> Path:
    return WORK / site / "dataset_split_manifest.csv"


def load_time_maps(site: str) -> dict[str, tuple[float, float]]:
    report = json.loads((WORK / site / "det_gt_match_report.json").read_text(encoding="utf-8"))
    clocks = {}
    for camera, row in report["cameras"].items():
        if "error" in row:
            continue
        clocks[camera] = (
            float(row["dt_applied_sec"]) + float(row.get("drift_const_ms", 0.0)) / 1000.0,
            float(row.get("drift_slope_ms_per_min", 0.0)) / 60000.0,
        )
    return clocks


def label_time(t_video: float, ab: tuple[float, float]) -> float:
    offset, slope = ab
    return float(t_video) + offset + slope * float(t_video)


def load_split_frames(site: str) -> dict[str, dict[tuple[str, int], dict]]:
    out = {"train": {}, "val": {}, "test": {}}
    with split_manifest(site).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row.get("split") or ""
            if split not in out:
                continue
            key = (row["camera_id"], int(row["frame_index"]))
            if key in out[split]:
                continue
            out[split][key] = {
                "timestamp": float(row["timestamp"]),
                "camera_id": row["camera_id"],
                "frame_index": int(row["frame_index"]),
                "split": split,
            }
    return out


def parse_sites(text: str) -> list[str]:
    sites = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [item for item in sites if item not in SITE_SPECS]
    if unknown:
        raise SystemExit(f"unknown sites: {unknown}")
    return sites
