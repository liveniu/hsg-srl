#!/usr/bin/env python3
"""Detect Ko-PER Sequence1a-1d as one continuous video per camera.

BoT-SORT keeps its state across the four files. Timestamps use the same
camera-clock epoch as the concatenated ground truth. A box on the frame
border is dropped, and a track is stored only after three in-frame hits.
This path does not reproduce the published tables: YOLO26l detections are
not identical across machines. Use the released geometry.db for those tables.

The database must already contain the imported calibration. If geometry.db
is missing, this script copies origin.db. It refuses to append a second
detection batch unless --replace is set.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

from bootstrap import ARTIFACT_ROOT, SRC_ROOT
from protocol import DET_BATCH, koper_root, origin_db, site_dir

sys.path.insert(0, str(SRC_ROOT))
from pipeline.geometry_guided_residual_mlp import compute_geometry_base, xywh_to_xyxy  # noqa: E402
from pipeline.multi_camera_trajectory_fusion import is_bbox_clipped, utc_now  # noqa: E402
from utils.scene_geometry_db import save_batch_camera_frame_wh  # noqa: E402

SCENE = "ko_per"
FOLDER = {"SK1": "KAB_SK_1_undist", "SK4": "KAB_SK_4_undist"}
SEQUENCES = ("Sequence1a", "Sequence1b", "Sequence1c", "Sequence1d")
MARGIN_PX = 3
MARGIN_RATIO = 0.01
MIN_STREAK = 3


def _only_mat(seq_dir: Path, prefix: str) -> Path:
    matches = sorted(seq_dir.glob(f"{prefix}_*.mat"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{seq_dir} {prefix}_*.mat count={len(matches)}")
    return matches[0]


def clocks(data_root: Path) -> tuple[float, dict]:
    found = {}
    for sequence in SEQUENCES:
        mat = sio.loadmat(_only_mat(data_root / sequence, "1"), squeeze_me=True, struct_as_record=False)
        found[sequence] = {}
        for camera, folder in FOLDER.items():
            found[sequence][camera] = np.asarray(
                getattr(mat["result"], folder).time, dtype=np.float64,
            ).ravel()
    epoch = min(float(arr[0]) for seq in found.values() for arr in seq.values())
    return epoch, found


def frames_for(camera: str, found: dict, data_root: Path) -> list[tuple[int, float, Path]]:
    epoch = min(float(arr[0]) for seq in found.values() for arr in seq.values())
    base = 0
    rows = []
    for sequence in SEQUENCES:
        seq_dir = data_root / sequence
        clock = found[sequence][camera]
        mat = sio.loadmat(_only_mat(seq_dir, "1"), squeeze_me=True, struct_as_record=False)
        stream = getattr(mat["result"], FOLDER[camera])
        seen: set[str] = set()
        for index, item in enumerate(np.asarray(stream.data, dtype=object).ravel()):
            name = str(item.image)
            if name in seen:
                continue
            seen.add(name)
            path = seq_dir / FOLDER[camera] / name
            if not path.is_file():
                continue
            rows.append((base + index + 1, (float(clock[index]) - epoch) * 1e-6, path))
        base += int(clock.shape[0])
    return rows


def prepare_db(replace: bool) -> Path:
    dest = site_dir(SCENE) / "geometry.db"
    if not dest.exists():
        source = origin_db(SCENE)
        if not source.exists():
            raise SystemExit(f"missing {source}. Import Ko-PER before detection.")
        shutil.copyfile(source, dest)
    conn = sqlite3.connect(dest)
    existing = conn.execute(
        "SELECT COUNT(*) FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
        (SCENE, DET_BATCH),
    ).fetchone()[0]
    conn.close()
    if existing and not replace:
        raise SystemExit(
            f"{dest} already has {existing} {DET_BATCH} rows. "
            "Refusing to mix a new detection with the released database. Pass --replace to overwrite that batch."
        )
    if existing and replace:
        conn = sqlite3.connect(dest)
        conn.execute(
            "DELETE FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
            (SCENE, DET_BATCH),
        )
        conn.commit()
        conn.close()
    return dest


def homography(conn: sqlite3.Connection, camera: str) -> tuple[np.ndarray, float, tuple[int, int]]:
    row = conn.execute(
        "SELECT homography, scale_px_per_meter FROM camera_calibrations WHERE scene_id=? AND camera_name=?",
        (SCENE, camera),
    ).fetchone()
    if row is None:
        raise SystemExit(f"{SCENE} has no calibration for {camera}")
    matrix = np.frombuffer(row[0], dtype=np.float64).reshape(3, 3).copy()
    wh = conn.execute(
        "SELECT infer_w, infer_h FROM trajectory_camera_config WHERE scene_id=? AND camera_name=? LIMIT 1",
        (SCENE, camera),
    ).fetchone()
    return matrix, float(row[1]), (int(wh[0]), int(wh[1])) if wh and wh[0] else (656, 494)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("HSG_YOLO_WEIGHTS", str(ARTIFACT_ROOT / "data" / "models" / "yolo26l.pt")))
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"missing detector weights: {model_path}")
    trt = os.environ.get("HSG_TRT_BIN", "").strip()
    if trt:
        os.environ["PATH"] = trt + os.pathsep + os.environ.get("PATH", "")
    from ultralytics import YOLO

    data_root = koper_root()
    db_path = prepare_db(args.replace)
    _epoch, found = clocks(data_root)
    conn = sqlite3.connect(db_path)
    model = YOLO(str(model_path))
    tracker = ARTIFACT_ROOT / "configs" / "botsort_static.yaml"
    now = utc_now()
    for camera in ("SK1", "SK4"):
        matrix, scale, wh = homography(conn, camera)
        frames = frames_for(camera, found, data_root)
        streaks: dict[int, int] = {}
        written = 0
        print(f"[detect] {camera} frames={len(frames)}", flush=True)
        for frame_index, timestamp_sec, path in frames:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            wh = (int(image.shape[1]), int(image.shape[0]))
            results = model.track(
                image, persist=True, conf=0.25, iou=0.7, classes=[2, 5, 7],
                imgsz=[384, 640], tracker=str(tracker), device="0", verbose=False,
            )
            result = results[0] if isinstance(results, list) else results
            boxes = getattr(result, "boxes", None)
            present: dict[int, tuple] = {}
            if boxes is not None and getattr(boxes, "id", None) is not None:
                names = getattr(model, "names", {}) or {}
                xyxy = boxes.xyxy.detach().cpu().numpy()
                ids = boxes.id.detach().cpu().numpy().astype(int)
                confs = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                for box, track_id, conf, cls_id in zip(xyxy, ids, confs, classes):
                    present[int(track_id)] = (
                        box, float(conf), int(cls_id), names.get(int(cls_id), str(int(cls_id))),
                    )
            for track_id in list(streaks):
                if track_id not in present:
                    del streaks[track_id]
            rows = []
            for track_id, (box, conf, cls_id, name) in present.items():
                if is_bbox_clipped(box, wh, margin_px=MARGIN_PX, margin_ratio=MARGIN_RATIO):
                    streaks[track_id] = 0
                    continue
                streaks[track_id] = streaks.get(track_id, 0) + 1
                if streaks[track_id] < MIN_STREAK:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box]
                anchor, world, _feat, _risk = compute_geometry_base(
                    xywh_to_xyxy((x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))),
                    matrix, scale, wh,
                )
                rows.append((
                    SCENE, DET_BATCH, camera, track_id, track_id, frame_index, timestamp_sec,
                    float(anchor[0]), float(anchor[1]), float(world[0]), float(world[1]),
                    x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1),
                    conf, cls_id, name, now,
                ))
            if rows:
                conn.executemany(
                    """INSERT INTO trajectory_observations
                       (scene_id, batch_id, camera_name, local_track_id, global_id,
                        frame_index, timestamp_sec, image_x, image_y, world_x, world_y,
                        bbox_x, bbox_y, bbox_w, bbox_h, confidence, class_id, class_name,
                        recorded_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    rows,
                )
                written += len(rows)
            if frame_index % 500 == 0:
                conn.commit()
                print(f"[detect] {camera} frame {frame_index} obs {written}", flush=True)
        save_batch_camera_frame_wh(conn, SCENE, DET_BATCH, {camera: wh})
        conn.commit()
        print(f"[detect] {camera} kept={written}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
