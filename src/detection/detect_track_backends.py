"""Detection/tracking backends for multi-camera fusion.

Default path remains Ultralytics YOLO + built-in BoT-SORT (``model.track``).
Optional ``mmdet_yolox`` path runs a pure-PyTorch MMDet YOLOX detector and feeds
boxes into an *independent* Ultralytics BoT-SORT instance per camera — without
calling ``YOLO.track()``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml


class _DetResults:
    """Minimal Results-like object accepted by ultralytics BOTSORT.update()."""

    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)
        w = self.xyxy[:, 2] - self.xyxy[:, 0]
        h = self.xyxy[:, 3] - self.xyxy[:, 1]
        cx = self.xyxy[:, 0] + w / 2.0
        cy = self.xyxy[:, 1] + h / 2.0
        self.xywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, idx: Any) -> "_DetResults":
        return _DetResults(self.xyxy[idx], self.conf[idx], self.cls[idx])


def _load_botsort_args(tracker: str | None) -> SimpleNamespace:
    """Load BoT-SORT hyper-params from Ultralytics tracker YAML (or defaults)."""
    defaults = {
        "tracker_type": "botsort",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "fuse_score": True,
        "gmc_method": "none",  # fixed cameras; sparseOptFlow needs Results.xyxy only
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.8,
        "with_reid": False,
        "model": "auto",
    }
    if not tracker:
        return SimpleNamespace(**defaults)

    raw = Path(str(tracker))
    candidates = [raw]
    if not raw.is_file():
        try:
            import ultralytics

            candidates.append(
                Path(ultralytics.__file__).resolve().parent
                / "cfg"
                / "trackers"
                / raw.name
            )
        except ImportError:
            pass
        # project-relative
        from utils.project_paths import resolve_project_path

        candidates.append(resolve_project_path(raw))

    payload: dict[str, Any] = {}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            break
        except (OSError, yaml.YAMLError):
            continue

    merged = dict(defaults)
    for key in defaults:
        if key in payload:
            merged[key] = payload[key]
    return SimpleNamespace(**merged)


def _parse_ultralytics_result(result: Any, model: Any) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "id", None) is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    ids = boxes.id.detach().cpu().numpy().astype(int)
    confs = (
        boxes.conf.detach().cpu().numpy()
        if getattr(boxes, "conf", None) is not None
        else np.full(len(ids), np.nan, dtype=np.float32)
    )
    clss = (
        boxes.cls.detach().cpu().numpy().astype(int)
        if getattr(boxes, "cls", None) is not None
        else np.full(len(ids), -1, dtype=np.int32)
    )
    names = getattr(model, "names", {}) or {}
    parsed: list[dict[str, Any]] = []
    for box, track_id, conf, cls_id in zip(xyxy, ids, confs, clss):
        class_name = names.get(int(cls_id), str(int(cls_id))) if int(cls_id) >= 0 else None
        parsed.append(
            {
                "xyxy": box.astype(np.float32),
                "track_id": int(track_id),
                "confidence": None if not np.isfinite(conf) else float(conf),
                "class_id": None if int(cls_id) < 0 else int(cls_id),
                "class_name": class_name,
            }
        )
    return parsed


class UltralyticsTrackBackend:
    """Wraps an Ultralytics YOLO model; uses built-in ``model.track`` + BoT-SORT."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.names = getattr(model, "names", {}) or {}
        self.backend_name = "ultralytics"

    def track_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        conf: float,
        iou: float,
        classes: list[int] | None,
        imgsz: int | list[int] | None,
        device: str | None,
        tracker: str | None,
    ) -> list[dict[str, Any]]:
        track_kwargs: dict[str, Any] = {
            "persist": True,
            "conf": conf,
            "iou": iou,
            "verbose": False,
        }
        if tracker:
            track_kwargs["tracker"] = tracker
        if classes is not None:
            track_kwargs["classes"] = classes
        if imgsz:
            track_kwargs["imgsz"] = imgsz
        if device:
            track_kwargs["device"] = device
        results = self.model.track(frame_bgr, **track_kwargs)
        result = results[0] if isinstance(results, list) else results
        return _parse_ultralytics_result(result, self.model)


class MmdetYoloxBotsortBackend:
    """MMDet YOLOX detector + independent per-camera BoT-SORT."""

    def __init__(self, detector: Any, tracker: Any) -> None:
        self.detector = detector
        self.tracker = tracker
        self.names = dict(getattr(detector, "names", {}) or {})
        self.backend_name = "mmdet_yolox"

    def track_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        conf: float,
        iou: float,
        classes: list[int] | None,
        imgsz: int | None,
        device: str | None,
        tracker: str | None,
    ) -> list[dict[str, Any]]:
        del imgsz, device, tracker  # detector constructed with fixed imgsz/device
        xyxy, scores, labels = self.detector.detect(frame_bgr, conf=conf, iou=iou)
        if classes is not None and self.detector.num_classes > 1:
            keep = np.isin(labels, np.asarray(classes, dtype=np.int32))
            xyxy, scores, labels = xyxy[keep], scores[keep], labels[keep]
        # Single-class Synthehicle weights: ignore COCO id filters (2,5,7, …).
        det = _DetResults(xyxy, scores, labels.astype(np.float32))
        online = self.tracker.update(det, frame_bgr)
        if online is None or len(online) == 0:
            return []
        online = np.asarray(online, dtype=np.float32)
        parsed: list[dict[str, Any]] = []
        for row in online:
            # [x1,y1,x2,y2,track_id,score,cls,idx]
            box = row[:4].astype(np.float32)
            track_id = int(row[4])
            score = float(row[5])
            cls_id = int(row[6])
            parsed.append(
                {
                    "xyxy": box,
                    "track_id": track_id,
                    "confidence": score if np.isfinite(score) else None,
                    "class_id": cls_id if cls_id >= 0 else None,
                    "class_name": self.names.get(cls_id, str(cls_id)),
                }
            )
        return parsed


def _load_ultralytics_yolo(model_path: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            'ultralyticspip install -r requirements.txt'
        ) from exc
    return YOLO(model_path)


def load_ultralytics_backends(model_path: str, camera_names: list[str]) -> dict[str, Any]:
    if not camera_names:
        return {}
    base_model = _load_ultralytics_yolo(model_path)
    backends: dict[str, Any] = {
        camera_names[0]: UltralyticsTrackBackend(base_model)
    }
    for name in camera_names[1:]:
        backends[name] = UltralyticsTrackBackend(copy.deepcopy(base_model))
    return backends


def load_mmdet_yolox_botsort_backends(
    model_path: str,
    camera_names: list[str],
    *,
    device: str | None,
    imgsz: int | None,
    conf: float,
    iou: float,
    tracker: str | None,
    class_name: str = "vehicle",
) -> dict[str, Any]:
    from ultralytics.trackers.bot_sort import BOTSORT, BOTrack

    from detection.mmdet_yolox_infer import MmdetYoloxDetector

    if not camera_names:
        return {}
    detector = MmdetYoloxDetector(
        model_path,
        device=device,
        imgsz=imgsz,
        score_thr=conf,
        nms_iou=iou,
        class_name=class_name,
    )
    botsort_args = _load_botsort_args(tracker)
    BOTrack.reset_id()
    backends: dict[str, Any] = {}
    for name in camera_names:
        # Fresh BoT-SORT state per camera (same reason Ultralytics deepcopies YOLO).
        # Note: BoT-SORT track IDs use a process-wide counter; fusion namespaces by camera.
        backends[name] = MmdetYoloxBotsortBackend(
            detector,
            BOTSORT(copy.deepcopy(botsort_args)),
        )
    print(
        f"  [mmdet_yolox] checkpoint={model_path} "
        f"num_classes={detector.num_classes} imgsz={detector.img_scale[0]} "
        f"score_thr={detector.score_thr} nms_iou={detector.nms_iou} "
        f"device={detector.device} cameras={len(camera_names)}"
    )
    if detector.num_classes == 1:
        print(
            '[mmdet_yolox] --classes COCO id'
            f" class_id=0 name={class_name!r}"
        )
    return backends


def load_track_backends(
    args: Any,
    camera_names: list[str],
) -> dict[str, Any]:
    """Factory: choose Ultralytics or MMDet-YOLOX+BoT-SORT from ``args.detect_backend``."""
    backend = str(getattr(args, "detect_backend", "ultralytics") or "ultralytics").lower()
    if backend in {"ultralytics", "yolo", "default"}:
        return load_ultralytics_backends(args.model, camera_names)
    if backend in {"mmdet_yolox", "mmdet", "yolox"}:
        return load_mmdet_yolox_botsort_backends(
            args.model,
            camera_names,
            device=getattr(args, "device", None),
            imgsz=getattr(args, "imgsz", None),
            conf=float(getattr(args, "conf", 0.25)),
            iou=float(getattr(args, "iou", 0.7)),
            tracker=getattr(args, "tracker", None),
            class_name=str(getattr(args, "mmdet_class_name", "vehicle") or "vehicle"),
        )
    raise ValueError(
        f" --detect-backend={backend!r}: ultralytics, mmdet_yolox"
    )


def backend_class_names(backends: dict[str, Any]) -> dict[int, str]:
    if not backends:
        return {}
    first = next(iter(backends.values()))
    return dict(getattr(first, "names", {}) or {})
