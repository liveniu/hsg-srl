#!/usr/bin/env python3
"""Bounded ground-coordinate residual for HSG-SRL.

Paper training is scripts/train_recording.py. That script fits the frozen
pairs, writes the F0 checkpoint and deploy file, and does not call this
command line. F0 is the filename tag for this residual, not a second method.

This train subcommand is a separate database trainer. Its numeric defaults
now match the published loss, but it still requires --split-manifest and
mines pairs from the geometry database. Supplying those numbers does not
make the command the paper fit. There is no camera-translation stage in the
published method; --camera-translation-json is not used by the paper scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.multi_camera_bev_stitch import (  # noqa: E402
    GEOMETRY_VERSION,
    load_calibration,
)
from pipeline.multi_camera_trajectory_fusion import (  # noqa: E402
    DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
    DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
    DEFAULT_ANCHOR_SHAPE_ETA,
    RISK_SCALAR_MODES,
    RiskScalar,
    adapt_homography_to_resolution,
    box_anchor,
    clamped_width_height_ratio_from_bbox,
    compute_risk_scalar,
    image_point_to_world,
    is_bbox_clipped,
    local_box_extent_meters,
)
from utils.g2_preflight_diagnostic import (  # noqa: E402
    diagnose_low_order_bias,
    format_low_order_preflight_report,
)
from utils.scene_geometry_db import (  # noqa: E402
    load_scene_overlap_pairs,
)
from utils.trajectory_batches import (  # noqa: E402
    finish_batch_stage,
    require_trajectory_batch_schema,
    start_batch_stage,
)
from utils.project_paths import resolve_project_path  # noqa: E402

FEATURE_VERSION = "fab1_metric_jacobian_robust_shape_v2"

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


                                                                         


@dataclass(frozen=True)
class AnchorFeatures:
    'MLP F P_geo\n\n ``xyxy``/``homography``/``scale`` \n ``t*`` bbox ``compute_geometry_base``\n ``feature``/``world_geo`` \n ``valid_for_mvc=False`` // MVC teacher'

    camera_name: str
    timestamp_sec: float
    global_id: int
    local_track_id: int
    feature: np.ndarray
    world_geo: tuple[float, float]
    image_anchor: tuple[float, float]
    frame_wh: tuple[int, int]
    horizon_confidence: float = 1.0
    xyxy: np.ndarray | None = None
    homography: np.ndarray | None = None
    scale: float | None = None
    risk: RiskScalar | None = None
    valid_for_mvc: bool = True
    scene_id: str = ""
    frame_index: int | None = None


@dataclass
class MVCPair:
    obs_a: AnchorFeatures
    obs_b: AnchorFeatures
    sample_weight: float = 1.0
    loss_mode_override: str | None = None
    require_kappa_gap: bool | None = None


@dataclass
class TemporalPair:
    prev: AnchorFeatures
    curr: AnchorFeatures


@dataclass
class RuntimeObservation:
    'Controller'

    camera_name: str
    global_id: int
    local_track_id: int
    timestamp_sec: float
    xyxy: np.ndarray
    homography: np.ndarray
    scale: float
    frame_wh: tuple[int, int]
    clipped: bool = False
    scene_id: str = ""
    frame_index: int | None = None


@dataclass(frozen=True)
class ResidualPrediction:
    'g2 BEV'

    image_anchor_geo: tuple[float, float]
    world_geo: tuple[float, float]
    delta_m: tuple[float, float]
    world_final: tuple[float, float]
    camera_translation_m: tuple[float, float] = (0.0, 0.0)
    world_corrected_geo: tuple[float, float] | None = None


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 80
    batch_size: int = 256
    lambda_reg: float = 0.10
    lambda_smooth: float = 0.02
    lambda_mean_delta: float = 1.0
    lambda_pair_center: float = 0.50
    max_residual_m: float = 1.0
    max_pair_dist_m: float | None = 2.0
    max_time_gap_sec: float = 0.35
    anchor_mode: str = "multiplicative_gating_offsetted"
                                          
    risk_scalar_mode: str = "metric_jacobian"
    s0_m: float = 60.0
    feature_l0_m: float = 10.0
    anchor_shape_eta: float = DEFAULT_ANCHOR_SHAPE_ETA
    anchor_aspect_ratio_min: float = DEFAULT_ANCHOR_ASPECT_RATIO_MIN
    anchor_aspect_ratio_max: float = DEFAULT_ANCHOR_ASPECT_RATIO_MAX
    sensitivity_clip_m: float | None = None
    extent_clip_m: float | None = None
    pixel_grazing_d0: float = 50.0 / 1080.0
                                                       
    motion_compensation: str = "linear_interpolate"
                                                                   
                                                  
    time_offsets_sec: dict[str, float] = field(default_factory=dict)
    loss_mode: str = "asymmetric_teacher"
    min_teacher_confidence: float = 0.20
    min_teacher_confidence_gap: float = 0.05
    teacher_confidence_power: float = 1.0
    student_weight_floor: float = 0.05
    reg_confidence_floor: float = 0.25
    loss_eps: float = 1e-6

    def __post_init__(self) -> None:
        valid = {"asymmetric_teacher", "symmetric_weighted"}
        if self.loss_mode not in valid:
            raise ValueError(f"loss_mode  {sorted(valid)} : {self.loss_mode}")
        if self.risk_scalar_mode not in RISK_SCALAR_MODES:
            raise ValueError(
                f"risk_scalar_mode  {RISK_SCALAR_MODES} : {self.risk_scalar_mode}"
            )
        valid_mc = {"off", "nearest", "linear_interpolate"}
        if self.motion_compensation not in valid_mc:
            raise ValueError(f"motion_compensation  {sorted(valid_mc)} ")
        if self.min_teacher_confidence < 0 or self.min_teacher_confidence > 1:
            raise ValueError('min_teacher_confidence [0, 1]')
        if self.min_teacher_confidence_gap < 0 or self.min_teacher_confidence_gap > 1:
            raise ValueError('min_teacher_confidence_gap [0, 1]')
        if self.teacher_confidence_power <= 0:
            raise ValueError('teacher_confidence_power > 0')
        if self.student_weight_floor < 0:
            raise ValueError('student_weight_floor >= 0')
        if self.reg_confidence_floor < 0:
            raise ValueError('reg_confidence_floor >= 0')
        if self.lambda_mean_delta < 0:
            raise ValueError('lambda_mean_delta >= 0')
        if self.lambda_pair_center < 0:
            raise ValueError('lambda_pair_center >= 0')
        if self.max_pair_dist_m is not None and self.max_pair_dist_m <= 0:
            raise ValueError('max_pair_dist_m > 0')
        if self.s0_m <= 0:
            raise ValueError('s0_m > 0')
        if self.feature_l0_m <= 0:
            raise ValueError('feature_l0_m > 0')
        if not 0.0 <= self.anchor_shape_eta <= 1.0:
            raise ValueError('anchor_shape_eta [0, 1]')
        if self.anchor_aspect_ratio_min <= 0 or self.anchor_aspect_ratio_max <= 0:
            raise ValueError('anchor_aspect_ratio_min/max > 0')
        if self.anchor_aspect_ratio_min > self.anchor_aspect_ratio_max:
            raise ValueError('anchor_aspect_ratio_min <= anchor_aspect_ratio_max')


@dataclass
class OnlineLearnConfig:
    enabled: bool = False
    every_frames: int = 300
    epochs: int = 3
    min_pairs: int = 64
    lr_scale: float = 0.3
    max_buffer_obs: int = 8000
    max_time_gap_sec: float = 0.35
    max_pairs_per_learn: int = 2048

    def __post_init__(self) -> None:
        if self.every_frames < 1:
            raise ValueError('every_frames >= 1')
        if self.epochs < 1:
            raise ValueError('epochs >= 1')
        if self.min_pairs < 2:
            raise ValueError('min_pairs >= 2')
        if self.lr_scale <= 0:
            raise ValueError('lr_scale > 0')
        if self.max_buffer_obs < 1:
            raise ValueError('max_buffer_obs >= 1')
        if self.max_time_gap_sec <= 0:
            raise ValueError('max_time_gap_sec > 0')
        if self.max_pairs_per_learn < 1:
            raise ValueError('max_pairs_per_learn >= 1')


                                                                       


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError('PyTorchpip install torch') from _TORCH_IMPORT_ERROR


def xywh_to_xyxy(bbox_xywh: tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = bbox_xywh
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def build_feature_vector(
    xyxy: np.ndarray,
    homography: np.ndarray,
    frame_wh: tuple[int, int],
    *,
    scale: float,
    risk: RiskScalar,
    anchor_xy: tuple[float, float] | None = None,
    feature_l0_m: float = 10.0,
    aspect_ratio_min: float = DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
    aspect_ratio_max: float = DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
    extent_clip_m: float | None = None,
) -> np.ndarray:
    'F = [cx/W, cy/H, w_m/L0, h_m/L0, clamp(w/h), risk_alpha, gate]fab1/v2 \n\n 3/4 ``local_box_extent_meters``\n 6/7 ``risk``\n camera_id / scene_id / residual \n label-free preflight \n MVC pair MLP'
    fw, fh = frame_wh
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w_m, h_m = local_box_extent_meters(
        xyxy, homography, scale, anchor_xy=anchor_xy, extent_clip_m=extent_clip_m,
    )
    l0 = max(float(feature_l0_m), 1e-6)
    gate = 4.0 * risk.alpha * (1.0 - risk.alpha)
    slimness = clamped_width_height_ratio_from_bbox(
        xyxy,
        aspect_ratio_min=aspect_ratio_min,
        aspect_ratio_max=aspect_ratio_max,
    )
    feat = np.array(
        [cx / fw, cy / fh, w_m / l0, h_m / l0, slimness, risk.alpha, gate],
        dtype=np.float32,
    )
    return feat


def compute_geometry_base(
    xyxy: np.ndarray,
    homography: np.ndarray,
    scale: float,
    frame_wh: tuple[int, int],
    *,
    risk_scalar_mode: str = "metric_jacobian",
    s0_m: float = 60.0,
    feature_l0_m: float = 10.0,
    pixel_grazing_d0: float = 50.0 / 1080.0,
    anchor_mode: str = "multiplicative_gating_offsetted",
    anchor_shape_eta: float = DEFAULT_ANCHOR_SHAPE_ETA,
    anchor_aspect_ratio_min: float = DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
    anchor_aspect_ratio_max: float = DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
    sensitivity_clip_m: float | None = None,
    extent_clip_m: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float], np.ndarray, RiskScalar]:
    '(image_anchor, world_geo, feature, risk)\n\n `kappa=1-risk.alpha`\n anchor_beta ``box_anchor``/``gated_anchor_beta``'
    risk = compute_risk_scalar(
        xyxy, homography, frame_wh,
        risk_scalar_mode=risk_scalar_mode, scale=scale, s0_m=s0_m,
        grazing_d0=pixel_grazing_d0, sensitivity_clip_m=sensitivity_clip_m,
    )
    anchor_xy = box_anchor(
        xyxy,
        anchor_mode,
        homography=homography,
        frame_wh=frame_wh,
        scale=scale,
        risk_scalar_mode=risk_scalar_mode,
        s0_m=s0_m,
        grazing_d0=pixel_grazing_d0,
        sensitivity_clip_m=sensitivity_clip_m,
        shape_eta=anchor_shape_eta,
        aspect_ratio_min=anchor_aspect_ratio_min,
        aspect_ratio_max=anchor_aspect_ratio_max,
        risk=risk,
    )
    world_geo = image_point_to_world(homography, anchor_xy, scale)
    feat = build_feature_vector(
        xyxy, homography, frame_wh,
        scale=scale, risk=risk, anchor_xy=anchor_xy,
        feature_l0_m=feature_l0_m,
        aspect_ratio_min=anchor_aspect_ratio_min,
        aspect_ratio_max=anchor_aspect_ratio_max,
        extent_clip_m=extent_clip_m,
    )
    return anchor_xy, world_geo, feat, risk


def interpolate_anchor_feature(
    before: AnchorFeatures,
    after: AnchorFeatures,
    t: float,
    *,
    risk_scalar_mode: str = "metric_jacobian",
    s0_m: float = 60.0,
    feature_l0_m: float = 10.0,
    pixel_grazing_d0: float = 50.0 / 1080.0,
    anchor_mode: str = "multiplicative_gating_offsetted",
    anchor_shape_eta: float = DEFAULT_ANCHOR_SHAPE_ETA,
    anchor_aspect_ratio_min: float = DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
    anchor_aspect_ratio_max: float = DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
    sensitivity_clip_m: float | None = None,
    extent_clip_m: float | None = None,
) -> AnchorFeatures | None:
    '``before``/``after`` ``t``\n\n bbox ``compute_geometry_base`` \n ``feature``/``world_geo``α/κ bbox \n ``xyxy``/``homography``/``scale``\n ``before``/``after`` ``None``'
    if before.xyxy is None or after.xyxy is None:
        return None
    if before.homography is None or before.scale is None:
        return None
    if before.camera_name != after.camera_name:
        return None
    t0, t1 = before.timestamp_sec, after.timestamp_sec
    if t1 <= t0:
        return None
    frac = max(0.0, min(1.0, (t - t0) / (t1 - t0)))
    xyxy_interp = before.xyxy + frac * (after.xyxy - before.xyxy)
    anchor_xy, world_geo, feat, risk = compute_geometry_base(
        xyxy_interp,
        before.homography,
        before.scale,
        before.frame_wh,
        risk_scalar_mode=risk_scalar_mode,
        s0_m=s0_m,
        feature_l0_m=feature_l0_m,
        pixel_grazing_d0=pixel_grazing_d0,
        anchor_mode=anchor_mode,
        anchor_shape_eta=anchor_shape_eta,
        anchor_aspect_ratio_min=anchor_aspect_ratio_min,
        anchor_aspect_ratio_max=anchor_aspect_ratio_max,
        sensitivity_clip_m=sensitivity_clip_m,
        extent_clip_m=extent_clip_m,
    )
    return AnchorFeatures(
        camera_name=before.camera_name,
        timestamp_sec=t,
        global_id=before.global_id,
        local_track_id=before.local_track_id,
        feature=feat,
        world_geo=world_geo,
        image_anchor=anchor_xy,
        frame_wh=before.frame_wh,
        horizon_confidence=risk.kappa,
        xyxy=xyxy_interp,
        homography=before.homography,
        scale=before.scale,
        risk=risk,
        valid_for_mvc=(risk.valid_ground_side and risk.valid_projection),
        scene_id=before.scene_id,
        frame_index=None,
    )


class GeometryDataView:
    'MVC /'

    @staticmethod
    def observation_to_features(
        obs: RuntimeObservation,
        *,
        risk_scalar_mode: str = "metric_jacobian",
        s0_m: float = 60.0,
        feature_l0_m: float = 10.0,
        pixel_grazing_d0: float = 50.0 / 1080.0,
        anchor_mode: str = "multiplicative_gating_offsetted",
        anchor_shape_eta: float = DEFAULT_ANCHOR_SHAPE_ETA,
        anchor_aspect_ratio_min: float = DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
        anchor_aspect_ratio_max: float = DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
        sensitivity_clip_m: float | None = None,
        extent_clip_m: float | None = None,
    ) -> AnchorFeatures:
        anchor_xy, world_geo, feat, risk = compute_geometry_base(
            obs.xyxy,
            obs.homography,
            obs.scale,
            obs.frame_wh,
            risk_scalar_mode=risk_scalar_mode,
            s0_m=s0_m,
            feature_l0_m=feature_l0_m,
            pixel_grazing_d0=pixel_grazing_d0,
            anchor_mode=anchor_mode,
            anchor_shape_eta=anchor_shape_eta,
            anchor_aspect_ratio_min=anchor_aspect_ratio_min,
            anchor_aspect_ratio_max=anchor_aspect_ratio_max,
            sensitivity_clip_m=sensitivity_clip_m,
            extent_clip_m=extent_clip_m,
        )
        return AnchorFeatures(
            camera_name=obs.camera_name,
            timestamp_sec=obs.timestamp_sec,
            global_id=obs.global_id,
            local_track_id=obs.local_track_id,
            feature=feat,
            world_geo=world_geo,
            image_anchor=anchor_xy,
            frame_wh=obs.frame_wh,
            horizon_confidence=risk.kappa,
            xyxy=np.asarray(obs.xyxy, dtype=np.float64),
            homography=obs.homography,
            scale=obs.scale,
            risk=risk,
            valid_for_mvc=(risk.valid_ground_side and risk.valid_projection and not obs.clipped),
            scene_id=obs.scene_id,
            frame_index=obs.frame_index,
        )

    @staticmethod
    def load_camera_calibrations(
        conn: sqlite3.Connection,
        scene_id: str,
        camera_names: list[str],
    ) -> dict[str, tuple[np.ndarray, float]]:
        '``geometry_version == GEOMETRY_VERSION``\n\n update.md\n cla1'
        out: dict[str, tuple[np.ndarray, float]] = {}
        for name in camera_names:
            cal = load_calibration(
                conn, name, scene_id, require_geometry_version=GEOMETRY_VERSION,
            )
            if cal is None:
                raise RuntimeError(f": scene={scene_id} camera={name}")
            homography, scale = cal
            out[name] = (homography.astype(np.float64), float(scale))
        return out

    @staticmethod
    def iter_observations_from_db(
        conn: sqlite3.Connection,
        scene_id: str,
        batch_id: str,
        camera_names: list[str],
        calibrations: dict[str, tuple[np.ndarray, float]],
        *,
        frame_wh_by_camera: dict[str, tuple[int, int]] | None = None,
        risk_scalar_mode: str = "metric_jacobian",
        s0_m: float = 60.0,
        feature_l0_m: float = 10.0,
        pixel_grazing_d0: float = 50.0 / 1080.0,
        anchor_mode: str = "multiplicative_gating_offsetted",
        anchor_shape_eta: float = DEFAULT_ANCHOR_SHAPE_ETA,
        anchor_aspect_ratio_min: float = DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
        anchor_aspect_ratio_max: float = DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
        sensitivity_clip_m: float | None = None,
        extent_clip_m: float | None = None,
        clip_margin_px: int = 3,
        clip_margin_ratio: float = 0.01,
    ) -> Iterator[AnchorFeatures]:
        placeholders = ",".join("?" for _ in camera_names)
        rows = conn.execute(
            f"""
            SELECT camera_name, global_id, local_track_id, frame_index, timestamp_sec,
                   bbox_x, bbox_y, bbox_w, bbox_h,
                   image_x, image_y, world_x, world_y
             FROM trajectory_observations
             WHERE scene_id = ? AND batch_id = ? AND camera_name IN ({placeholders})
             ORDER BY timestamp_sec
            """,
            (scene_id, batch_id, *camera_names),
        ).fetchall()

        default_wh = (1920, 1080)
        for row in rows:
            cam = str(row[0])
            bbox = (float(row[5]), float(row[6]), float(row[7]), float(row[8]))
            xyxy = xywh_to_xyxy(bbox)
            H, scale = calibrations[cam]
            wh = (frame_wh_by_camera or {}).get(cam, default_wh)
            if is_bbox_clipped(
                xyxy,
                wh,
                margin_px=clip_margin_px,
                margin_ratio=clip_margin_ratio,
            ):
                continue
            _anchor_xy, _world_geo, feat, risk = compute_geometry_base(
                xyxy,
                H,
                scale,
                wh,
                risk_scalar_mode=risk_scalar_mode,
                s0_m=s0_m,
                feature_l0_m=feature_l0_m,
                pixel_grazing_d0=pixel_grazing_d0,
                anchor_mode=anchor_mode,
                anchor_shape_eta=anchor_shape_eta,
                anchor_aspect_ratio_min=anchor_aspect_ratio_min,
                anchor_aspect_ratio_max=anchor_aspect_ratio_max,
                sensitivity_clip_m=sensitivity_clip_m,
                extent_clip_m=extent_clip_m,
            )
                                                            
                                                                   
                                                                
            anchor_xy = (float(row[9]), float(row[10]))
            world_geo = (float(row[11]), float(row[12]))
            yield AnchorFeatures(
                camera_name=cam,
                timestamp_sec=float(row[4]),
                global_id=int(row[1]),
                local_track_id=int(row[2]),
                feature=feat,
                world_geo=world_geo,
                image_anchor=anchor_xy,
                frame_wh=wh,
                horizon_confidence=risk.kappa,
                xyxy=xyxy,
                homography=H,
                scale=scale,
                risk=risk,
                valid_for_mvc=(risk.valid_ground_side and risk.valid_projection),
                scene_id=scene_id,
                frame_index=int(row[3]) if row[3] is not None else None,
            )

    @staticmethod
    def _find_time_bracket(
        sorted_obs: list[AnchorFeatures],
        t: float,
        max_gap_sec: float,
        max_extrapolation_sec: float = 0.0,
    ) -> tuple[AnchorFeatures, AnchorFeatures] | None:
        '``t`` \n\n ``max_gap_sec`` ``2×max_gap_sec`` \n \n ``max_extrapolation_sec`` ``t`` \n ``0.0`` bracketing``t`` ``[sorted_obs[0].t, sorted_obs[-1].t]``\n ``None``mine_mvc_pairs ``0.0``\n hold-first/hold-last'
        n = len(sorted_obs)
        if n < 2:
            return None
        if t <= sorted_obs[0].timestamp_sec:
            gap = sorted_obs[0].timestamp_sec - t
            if max_extrapolation_sec <= 0.0 or gap > max_extrapolation_sec:
                return None
            return (sorted_obs[0], sorted_obs[1])
        if t >= sorted_obs[-1].timestamp_sec:
            gap = t - sorted_obs[-1].timestamp_sec
            if max_extrapolation_sec <= 0.0 or gap > max_extrapolation_sec:
                return None
            return (sorted_obs[-2], sorted_obs[-1])
        for i in range(n - 1):
            a, b = sorted_obs[i], sorted_obs[i + 1]
            if a.timestamp_sec <= t <= b.timestamp_sec:
                if (b.timestamp_sec - a.timestamp_sec) > 2 * max_gap_sec:
                    return None
                return a, b
        return None

    @staticmethod
    def mine_mvc_pairs(
        observations: list[AnchorFeatures],
        overlap_camera_pairs: set[tuple[str, str]],
        *,
        max_time_gap_sec: float = 0.35,
        motion_compensation: str = "linear_interpolate",
        time_offsets_sec: dict[tuple[str, str], float] | None = None,
        risk_scalar_mode: str = "metric_jacobian",
        s0_m: float = 60.0,
        feature_l0_m: float = 10.0,
        pixel_grazing_d0: float = 50.0 / 1080.0,
        anchor_mode: str = "multiplicative_gating_offsetted",
        anchor_shape_eta: float = DEFAULT_ANCHOR_SHAPE_ETA,
        anchor_aspect_ratio_min: float = DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
        anchor_aspect_ratio_max: float = DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
        sensitivity_clip_m: float | None = None,
        extent_clip_m: float | None = None,
    ) -> list[MVCPair]:
        'MVC pair time ``t*``\n\n ``motion_compensation``:\n - ``linear_interpolate`` ``t*``\n bbox \n ** bracketing**``t*`` \n —— hold-first/\n hold-last \n ``2×max_time_gap_sec`` \n - ``nearest`` / ``off``cla1 (ix) \n\n ``time_offsets_sec````{(cam_a, cam_b): offset_sec}`` \n ``offset`` cam_b cam_a \n //``valid_for_mvc=False``'
        offsets = time_offsets_sec or {}
        by_gid: dict[int, list[AnchorFeatures]] = defaultdict(list)
        for obs in observations:
            if not obs.valid_for_mvc:
                continue
            by_gid[obs.global_id].append(obs)

        pairs: list[MVCPair] = []
        for obs_list in by_gid.values():
            if len({o.camera_name for o in obs_list}) < 2:
                continue
            for cam_a, cam_b in overlap_camera_pairs:
                list_a = sorted(
                    (o for o in obs_list if o.camera_name == cam_a), key=lambda o: o.timestamp_sec
                )
                list_b = sorted(
                    (o for o in obs_list if o.camera_name == cam_b), key=lambda o: o.timestamp_sec
                )
                if not list_a or not list_b:
                    continue
                offset = offsets.get((cam_a, cam_b), -offsets.get((cam_b, cam_a), 0.0))
                for oa in list_a:
                    best: AnchorFeatures | None = None
                    best_dt = max_time_gap_sec + 1.0
                    for ob in list_b:
                        dt = abs(oa.timestamp_sec - (ob.timestamp_sec - offset))
                        if dt < best_dt:
                            best_dt = dt
                            best = ob
                    if best is None or best_dt > max_time_gap_sec:
                        continue
                    if motion_compensation in ("nearest", "off"):
                        pairs.append(MVCPair(obs_a=oa, obs_b=best))
                        continue

                    t_star = 0.5 * (oa.timestamp_sec + (best.timestamp_sec - offset))
                    bracket_a = GeometryDataView._find_time_bracket(list_a, t_star, max_time_gap_sec)
                    bracket_b = GeometryDataView._find_time_bracket(
                        list_b, t_star + offset, max_time_gap_sec,
                    )
                    if bracket_a is None or bracket_b is None:
                        continue
                    interp_a = interpolate_anchor_feature(
                        *bracket_a, t_star,
                        risk_scalar_mode=risk_scalar_mode, s0_m=s0_m, feature_l0_m=feature_l0_m,
                        pixel_grazing_d0=pixel_grazing_d0, anchor_mode=anchor_mode,
                        anchor_shape_eta=anchor_shape_eta,
                        anchor_aspect_ratio_min=anchor_aspect_ratio_min,
                        anchor_aspect_ratio_max=anchor_aspect_ratio_max,
                        sensitivity_clip_m=sensitivity_clip_m, extent_clip_m=extent_clip_m,
                    )
                    interp_b = interpolate_anchor_feature(
                        *bracket_b, t_star + offset,
                        risk_scalar_mode=risk_scalar_mode, s0_m=s0_m, feature_l0_m=feature_l0_m,
                        pixel_grazing_d0=pixel_grazing_d0, anchor_mode=anchor_mode,
                        anchor_shape_eta=anchor_shape_eta,
                        anchor_aspect_ratio_min=anchor_aspect_ratio_min,
                        anchor_aspect_ratio_max=anchor_aspect_ratio_max,
                        sensitivity_clip_m=sensitivity_clip_m, extent_clip_m=extent_clip_m,
                    )
                    if interp_a is None or interp_b is None:
                        continue
                    if not (interp_a.valid_for_mvc and interp_b.valid_for_mvc):
                        continue
                    interp_b = replace(interp_b, timestamp_sec=t_star)
                    pairs.append(MVCPair(obs_a=interp_a, obs_b=interp_b))
        return pairs

    @staticmethod
    def mine_temporal_pairs(
        observations: list[AnchorFeatures],
        *,
        max_frame_gap_sec: float = 0.2,
    ) -> list[TemporalPair]:
        groups: dict[tuple[str, int], list[AnchorFeatures]] = defaultdict(list)
        for obs in observations:
            if not obs.valid_for_mvc:
                continue
            groups[(obs.camera_name, obs.local_track_id)].append(obs)
        out: list[TemporalPair] = []
        for items in groups.values():
            items.sort(key=lambda o: o.timestamp_sec)
            for prev, curr in zip(items, items[1:]):
                if curr.timestamp_sec - prev.timestamp_sec <= max_frame_gap_sec:
                    out.append(TemporalPair(prev=prev, curr=curr))
        return out

    @staticmethod
    def default_overlap_pairs(camera_names: list[str]) -> set[tuple[str, str]]:
        """Adjacent-camera chain. Not the published full overlap graph."""
        names = sorted(camera_names)
        return {tuple(sorted((names[i], names[i + 1]))) for i in range(len(names) - 1)}


                                                                   


if nn is not None:

    class GeometryGuidedResidualMLP(nn.Module):
        '3 MLPF (7) → ΔP (2) BEV'

        def __init__(self, in_dim: int = 7, hidden: int = 64, max_residual_m: float = 1.0):
            super().__init__()
            self.max_residual_m = max_residual_m
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 2),
            )
            linear_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
            for m in linear_layers[:-1]:
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
            # Fresh checkpoints must be a pure-geometry fallback: ΔP = 0.
            nn.init.zeros_(linear_layers[-1].weight)
            nn.init.zeros_(linear_layers[-1].bias)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self.net(features)) * self.max_residual_m

else:
    GeometryGuidedResidualMLP = None  # type: ignore[misc, assignment]


class ResidualMLPModel:
    'horizon-aware MVC'

    def __init__(self, mlp: GeometryGuidedResidualMLP, cfg: TrainConfig):
        self.mlp = mlp
        self.cfg = cfg

    @classmethod
    def create(cls, cfg: TrainConfig | None = None) -> ResidualMLPModel:
        _require_torch()
        cfg = cfg or TrainConfig()
        mlp = GeometryGuidedResidualMLP(max_residual_m=cfg.max_residual_m)
        return cls(mlp, cfg)

    def predict_delta(self, features: np.ndarray | torch.Tensor) -> np.ndarray:
        _require_torch()
        self.mlp.eval()
        if isinstance(features, np.ndarray):
            t = torch.from_numpy(features.astype(np.float32))
            if t.ndim == 1:
                t = t.unsqueeze(0)
        else:
            t = features
        with torch.no_grad():
            delta = self.mlp(t).cpu().numpy()
        return delta.reshape(-1, 2) if delta.ndim == 2 else delta

    def _pair_teacher_confidence(self, pair: MVCPair) -> float:
        return max(pair.obs_a.horizon_confidence, pair.obs_b.horizon_confidence)

    def _pair_teacher_confidence_gap(self, pair: MVCPair) -> float:
        return abs(pair.obs_a.horizon_confidence - pair.obs_b.horizon_confidence)

    def _pair_learning_score(self, pair: MVCPair) -> float:
        omega_a = pair.obs_a.horizon_confidence
        omega_b = pair.obs_b.horizon_confidence
        teacher_conf = max(omega_a, omega_b)
        asymmetry = abs(omega_a - omega_b)
        both_view_support = min(omega_a, omega_b)
        return teacher_conf * (0.25 + asymmetry) + 0.10 * both_view_support

    def _pair_selection_key(self, pair: MVCPair) -> tuple[Any, ...]:
        scene = str(pair.obs_a.scene_id or pair.obs_b.scene_id or "")
        t_mid = 0.5 * (float(pair.obs_a.timestamp_sec) + float(pair.obs_b.timestamp_sec))
        cam_a, cam_b = sorted((str(pair.obs_a.camera_name), str(pair.obs_b.camera_name)))
        return (
            -self._pair_learning_score(pair),
            scene,
            round(t_mid, 6),
            cam_a,
            cam_b,
            int(pair.obs_a.local_track_id),
            int(pair.obs_b.local_track_id),
            int(pair.obs_a.frame_index or -1),
            int(pair.obs_b.frame_index or -1),
        )

    def select_mvc_pairs(
        self,
        pairs: list[MVCPair],
        *,
        max_pairs: int | None = None,
    ) -> list[MVCPair]:
        'teacher near/far pair\n\n ``require_kappa_gap=False`` pairNEAR_NEAR teacher\n |Δκ|'
        filtered = []
        for p in pairs:
            if self._pair_teacher_confidence(p) < self.cfg.min_teacher_confidence:
                continue
            req_gap = True if p.require_kappa_gap is None else bool(p.require_kappa_gap)
            if req_gap and self._pair_teacher_confidence_gap(p) < self.cfg.min_teacher_confidence_gap:
                continue
            filtered.append(p)
        filtered.sort(key=self._pair_selection_key)
        if max_pairs is None or len(filtered) <= max_pairs:
            return filtered
        return filtered[:max_pairs]

    def _weighted_mean(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=self.cfg.loss_eps)

    @staticmethod
    def _percentile(values: np.ndarray, q: float) -> float:
        if values.size == 0:
            return 0.0
        return float(np.percentile(values, q))

    def evaluate_mvc_pairs(self, pairs: list[MVCPair]) -> dict[str, float]:
        """Evaluate geometric consistency before/after residual prediction.

        This is a training-health diagnostic, not a held-out metric: pairs come
        from the same MVC supervision set. It is still useful for catching
        saturated residual fields or pair sets that exceed the bounded residual
        branch's theoretical closure range.
        """
        if not pairs:
            return {}

        feat_a = np.stack([p.obs_a.feature for p in pairs]).astype(np.float32)
        feat_b = np.stack([p.obs_b.feature for p in pairs]).astype(np.float32)
        geo_a = np.asarray([p.obs_a.world_geo for p in pairs], dtype=np.float64)
        geo_b = np.asarray([p.obs_b.world_geo for p in pairs], dtype=np.float64)
        omega_a = np.asarray([p.obs_a.horizon_confidence for p in pairs], dtype=np.float64)
        omega_b = np.asarray([p.obs_b.horizon_confidence for p in pairs], dtype=np.float64)

        delta_a = self.predict_delta(feat_a).astype(np.float64)
        delta_b = self.predict_delta(feat_b).astype(np.float64)
        pred_a = geo_a + delta_a
        pred_b = geo_b + delta_b

        geo_dist = np.linalg.norm(geo_a - geo_b, axis=1)
        final_dist = np.linalg.norm(pred_a - pred_b, axis=1)
        improvement = geo_dist - final_dist
        delta_all = np.vstack([delta_a, delta_b])
        delta_norm = np.linalg.norm(delta_all, axis=1)
        max_component = np.max(np.abs(delta_all), axis=1)

        r_max = max(float(self.cfg.max_residual_m), 1e-9)
        vector_bound = float(np.sqrt(2.0) * r_max)
        pair_close_bound = 2.0 * vector_bound
        teacher_conf = np.maximum(omega_a, omega_b)
        omega_gap = np.abs(omega_a - omega_b)

        return {
            "pair_count": float(len(pairs)),
            "geo_mean": float(np.mean(geo_dist)),
            "geo_p50": self._percentile(geo_dist, 50),
            "geo_p90": self._percentile(geo_dist, 90),
            "geo_max": float(np.max(geo_dist)),
            "geo_over_pair_bound_frac": float(np.mean(geo_dist > pair_close_bound)),
            "final_mean": float(np.mean(final_dist)),
            "final_p50": self._percentile(final_dist, 50),
            "final_p90": self._percentile(final_dist, 90),
            "final_max": float(np.max(final_dist)),
            "improvement_mean": float(np.mean(improvement)),
            "improved_frac": float(np.mean(improvement > 0.0)),
            "worse_frac": float(np.mean(improvement < 0.0)),
            "delta_norm_mean": float(np.mean(delta_norm)),
            "delta_norm_p90": self._percentile(delta_norm, 90),
            "delta_norm_max": float(np.max(delta_norm)),
            "component_sat_frac": float(np.mean(max_component > 0.95 * r_max)),
            "vector_sat_frac": float(np.mean(delta_norm > 0.95 * vector_bound)),
            "teacher_conf_mean": float(np.mean(teacher_conf)),
            "omega_gap_mean": float(np.mean(omega_gap)),
            "pair_close_bound": pair_close_bound,
        }

    @staticmethod
    def _fmt_stats(stats: dict[str, float], *keys: str) -> str:
        return "/".join(f"{stats.get(k, 0.0):.3f}" for k in keys)

    def _print_training_summary(
        self,
        before: dict[str, float],
        after: dict[str, float],
    ) -> None:
        if not after:
            return
        print(
            "  [g2 summary] P_geo dist mean/p50/p90/max="
            f"{self._fmt_stats(after, 'geo_mean', 'geo_p50', 'geo_p90', 'geo_max')}m; "
            f"over_bound={after.get('geo_over_pair_bound_frac', 0.0) * 100:.1f}% "
            f"(bound={after.get('pair_close_bound', 0.0):.2f}m)"
        )
        print(
            "  [g2 summary] P_final dist mean before->after="
            f"{before.get('final_mean', 0.0):.3f}->{after.get('final_mean', 0.0):.3f}m; "
            "p90 before->after="
            f"{before.get('final_p90', 0.0):.3f}->{after.get('final_p90', 0.0):.3f}m; "
            f"improved={after.get('improved_frac', 0.0) * 100:.1f}% "
            f"worse={after.get('worse_frac', 0.0) * 100:.1f}%"
        )
        print(
            "  [g2 summary] residual norm mean/p90/max="
            f"{self._fmt_stats(after, 'delta_norm_mean', 'delta_norm_p90', 'delta_norm_max')}m; "
            f"component_sat={after.get('component_sat_frac', 0.0) * 100:.1f}% "
            f"vector_sat={after.get('vector_sat_frac', 0.0) * 100:.1f}%; "
            f"teacher={after.get('teacher_conf_mean', 0.0):.3f} "
            f"gap={after.get('omega_gap_mean', 0.0):.3f}"
        )

        final_mean = after.get("final_mean", 0.0)
        geo_mean = after.get("geo_mean", 0.0)
        improved = after.get("improved_frac", 0.0)
        sat = after.get("component_sat_frac", 0.0)
        over_bound = after.get("geo_over_pair_bound_frac", 0.0)
        if final_mean <= geo_mean * 0.85 and improved >= 0.65 and sat < 0.20 and over_bound < 0.10:
            verdict = "healthy"
        elif final_mean > geo_mean or sat > 0.50 or over_bound > 0.30:
            verdict = "risky"
        else:
            verdict = "mixed"
        print(f"  [g2 summary] expected_effect={verdict}")

    def horizon_aware_mvc_loss(
        self,
        pairs: list[MVCPair],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        'stop-gradient ablation baseline'
        feat_a = torch.from_numpy(np.stack([p.obs_a.feature for p in pairs]))
        feat_b = torch.from_numpy(np.stack([p.obs_b.feature for p in pairs]))
        geo_a = torch.tensor([p.obs_a.world_geo for p in pairs], dtype=torch.float32)
        geo_b = torch.tensor([p.obs_b.world_geo for p in pairs], dtype=torch.float32)
        omega_a = torch.tensor(
            [p.obs_a.horizon_confidence for p in pairs], dtype=torch.float32
        )
        omega_b = torch.tensor(
            [p.obs_b.horizon_confidence for p in pairs], dtype=torch.float32
        )

        delta_a = self.mlp(feat_a)
        delta_b = self.mlp(feat_b)
        pred_a = geo_a + delta_a
        pred_b = geo_b + delta_b

        omega_a = torch.clamp(omega_a, 0.0, 1.0)
        omega_b = torch.clamp(omega_b, 0.0, 1.0)
        teacher_conf_a = omega_a ** self.cfg.teacher_confidence_power
        teacher_conf_b = omega_b ** self.cfg.teacher_confidence_power
        teacher_conf = torch.maximum(teacher_conf_a, teacher_conf_b)

        mixed = any(
            p.loss_mode_override is not None or abs(float(p.sample_weight) - 1.0) > 1e-12
            for p in pairs
        )
        if (not mixed) and self.cfg.loss_mode == "symmetric_weighted":
            diff2 = torch.sum((pred_a - pred_b) ** 2, dim=1)
            pair_weight = omega_a * omega_b
            loss_mvc = self._weighted_mean(diff2, pair_weight)
            stats_weight = pair_weight
            loss_asym = torch.mean(torch.abs(omega_a - omega_b))
        elif not mixed:
            teacher = (
                teacher_conf_a.unsqueeze(1) * pred_a
                + teacher_conf_b.unsqueeze(1) * pred_b
            ) / (teacher_conf_a + teacher_conf_b + self.cfg.loss_eps).unsqueeze(1)
            teacher = teacher.detach()

            residual_need_a = 1.0 - omega_a
            residual_need_b = 1.0 - omega_b
            student_weight_a = teacher_conf * (residual_need_a + self.cfg.student_weight_floor)
            student_weight_b = teacher_conf * (residual_need_b + self.cfg.student_weight_floor)
            loss_a = torch.sum((pred_a - teacher) ** 2, dim=1)
            loss_b = torch.sum((pred_b - teacher) ** 2, dim=1)
            loss_mvc = (
                torch.sum(student_weight_a * loss_a + student_weight_b * loss_b)
                / torch.clamp(
                    torch.sum(student_weight_a + student_weight_b),
                    min=self.cfg.loss_eps,
                )
            )
            stats_weight = 0.5 * (student_weight_a + student_weight_b)
            loss_asym = torch.mean(torch.abs(omega_a - omega_b))
        else:
            teacher = (
                teacher_conf_a.unsqueeze(1) * pred_a
                + teacher_conf_b.unsqueeze(1) * pred_b
            ) / (teacher_conf_a + teacher_conf_b + self.cfg.loss_eps).unsqueeze(1)
            teacher = teacher.detach()
            residual_need_a = 1.0 - omega_a
            residual_need_b = 1.0 - omega_b
            student_weight_a = teacher_conf * (residual_need_a + self.cfg.student_weight_floor)
            student_weight_b = teacher_conf * (residual_need_b + self.cfg.student_weight_floor)
            loss_a = torch.sum((pred_a - teacher) ** 2, dim=1)
            loss_b = torch.sum((pred_b - teacher) ** 2, dim=1)
            asym = (
                student_weight_a * loss_a + student_weight_b * loss_b
            ) / torch.clamp(student_weight_a + student_weight_b, min=self.cfg.loss_eps)
            diff2 = torch.sum((pred_a - pred_b) ** 2, dim=1)
            is_sym = torch.tensor(
                [
                    (p.loss_mode_override or self.cfg.loss_mode) == "symmetric_weighted"
                    for p in pairs
                ],
                dtype=torch.bool,
            )
            sample_w = torch.tensor(
                [float(p.sample_weight) for p in pairs], dtype=torch.float32
            )
            per_pair = torch.where(is_sym, diff2, asym)
            pair_weight = sample_w * torch.where(
                is_sym, omega_a * omega_b, 0.5 * (student_weight_a + student_weight_b)
            )
            loss_mvc = self._weighted_mean(per_pair, pair_weight)
            stats_weight = pair_weight
            loss_asym = torch.mean(torch.abs(omega_a - omega_b))

        reg_a = torch.sum(delta_a ** 2, dim=1)
        reg_b = torch.sum(delta_b ** 2, dim=1)
        reg_weight_a = omega_a + self.cfg.reg_confidence_floor
        reg_weight_b = omega_b + self.cfg.reg_confidence_floor
        loss_reg = 0.5 * (
            self._weighted_mean(reg_a, reg_weight_a)
            + self._weighted_mean(reg_b, reg_weight_b)
        )
        delta_all = torch.cat([delta_a, delta_b], dim=0)
        mean_delta_vec = torch.mean(delta_all, dim=0)
        loss_mean_delta = torch.sum(mean_delta_vec ** 2)
        pair_center_shift = 0.5 * (delta_a + delta_b)
        pair_center_error = torch.sum(pair_center_shift ** 2, dim=1)
        pair_center_weight = torch.minimum(teacher_conf_a, teacher_conf_b)
        loss_pair_center = self._weighted_mean(pair_center_error, pair_center_weight)
        total = (
            loss_mvc
            + self.cfg.lambda_reg * loss_reg
            + self.cfg.lambda_mean_delta * loss_mean_delta
            + self.cfg.lambda_pair_center * loss_pair_center
        )
        stats = {
            "loss_mvc": float(loss_mvc.item()),
            "loss_reg": float(loss_reg.item()),
            "loss_mean_delta": float(loss_mean_delta.item()),
            "loss_pair_center": float(loss_pair_center.item()),
            "mean_delta_norm": float(torch.linalg.norm(mean_delta_vec).item()),
            "mean_loss_weight": float(stats_weight.mean().item()),
            "mean_teacher_confidence": float(teacher_conf.mean().item()),
            "mean_omega_gap": float(loss_asym.item()),
        }
        return total, stats

    def temporal_smooth_loss(self, pairs: list[TemporalPair]) -> torch.Tensor:
        if not pairs:
            return torch.tensor(0.0)
        ft0 = torch.from_numpy(np.stack([p.prev.feature for p in pairs]))
        ft1 = torch.from_numpy(np.stack([p.curr.feature for p in pairs]))
        return torch.mean((self.mlp(ft1) - self.mlp(ft0)) ** 2)

    def train_offline(
        self,
        mvc_pairs: list[MVCPair],
        temporal_pairs: list[TemporalPair],
        *,
        epochs: int | None = None,
        lr: float | None = None,
    ) -> dict[str, float]:
        mvc_pairs = self.select_mvc_pairs(mvc_pairs)
        if not mvc_pairs:
            raise RuntimeError('MVC')
        before_summary = self.evaluate_mvc_pairs(mvc_pairs)
        epochs = epochs if epochs is not None else self.cfg.epochs
        lr = lr if lr is not None else self.cfg.lr
        opt = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self.mlp.train()

        n = len(mvc_pairs)
        last_stats: dict[str, float] = {}
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            total_loss = 0.0
            steps = 0
            for start in range(0, n, self.cfg.batch_size):
                idx = perm[start : start + self.cfg.batch_size]
                batch = [mvc_pairs[i] for i in idx]
                loss, stats = self.horizon_aware_mvc_loss(batch)
                if temporal_pairs and self.cfg.lambda_smooth > 0:
                    sample = temporal_pairs[: min(512, len(temporal_pairs))]
                    loss = loss + self.cfg.lambda_smooth * self.temporal_smooth_loss(sample)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
                steps += 1
                last_stats = stats
            if (epoch + 1) % 10 == 0 or epoch == 0 or epochs <= 5:
                print(
                    f"  [g2 train] epoch {epoch + 1}/{epochs}  "
                    f"loss={total_loss / max(steps, 1):.4f}  "
                    f"mvc_pairs={n}  w_mean={last_stats.get('mean_loss_weight', 0):.3f}  "
                    f"teacher={last_stats.get('mean_teacher_confidence', 0):.3f}  "
                    f"mean_delta={last_stats.get('mean_delta_norm', 0):.3f}m  "
                    f"mean_loss={last_stats.get('loss_mean_delta', 0):.5f}  "
                    f"center_loss={last_stats.get('loss_pair_center', 0):.5f}  "
                    f"lambda_mean={self.cfg.lambda_mean_delta:.3f}  "
                    f"lambda_center={self.cfg.lambda_pair_center:.3f}"
                )
        last_stats["final_loss"] = total_loss / max(steps, 1)
        after_summary = self.evaluate_mvc_pairs(mvc_pairs)
        self._print_training_summary(before_summary, after_summary)
        last_stats.update({f"summary_{k}": v for k, v in after_summary.items()})
        return last_stats


def save_checkpoint(
    path: Path,
    model: GeometryGuidedResidualMLP,
    meta: dict[str, Any],
) -> None:
    _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "feature_version": FEATURE_VERSION}
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(
    path: Path,
    *,
    require_feature_version: bool = True,
) -> tuple[GeometryGuidedResidualMLP, dict[str, Any]]:
    'checkpoint ``meta.feature_version`` fab1 \n\n fab1 cla1 checkpoint 7 \n alpha fab1 alpha'
    _require_torch()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = blob.get("meta", {})
    feature_version = meta.get("feature_version")
    if require_feature_version and feature_version != FEATURE_VERSION:
        raise RuntimeError(
            f"checkpoint {path}  feature_version={feature_version!r}  "
            f"{FEATURE_VERSION!r} fab1  cla1 checkpoint "
            '/'
        )
    max_res = float(meta.get("max_residual_m", 0.8))
    model = GeometryGuidedResidualMLP(max_residual_m=max_res)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, meta


                                                                    


class MVCResidualController:
    'g2 MVC'

    def __init__(
        self,
        *,
        train_cfg: TrainConfig | None = None,
        online_cfg: OnlineLearnConfig | None = None,
        overlap_pairs: set[tuple[str, str]] | None = None,
        camera_translation_corrections: CameraTranslationCorrections | None = None,
        checkpoint_path: Path | None = None,
        save_checkpoint_path: Path | None = None,
    ):
        _require_torch()
        self.train_cfg = train_cfg or TrainConfig()
        self.online_cfg = online_cfg or OnlineLearnConfig()
        self.overlap_pairs = overlap_pairs or set()
        self.checkpoint_path = checkpoint_path
        # Loading and saving are intentionally separated: online adaptation must
        # never overwrite the validated offline checkpoint unless an explicit
        # output path is provided by the caller.
        self.save_checkpoint_path = save_checkpoint_path
        self.view = GeometryDataView()
        explicit_train_cfg = train_cfg is not None
        loaded_meta: dict[str, Any] = {}
        self.camera_translation_corrections = CameraTranslationCorrections()
        self.training_camera_translation_corrections = CameraTranslationCorrections()

        if checkpoint_path is not None and checkpoint_path.is_file():
            mlp, meta = load_checkpoint(checkpoint_path)
            loaded_meta = dict(meta)
            if not explicit_train_cfg and isinstance(meta.get("train_config"), dict):
                saved_cfg = meta["train_config"]
                valid_fields = {f.name for f in fields(TrainConfig)}
                for key, value in saved_cfg.items():
                    if key in valid_fields:
                        setattr(self.train_cfg, key, value)
            if not explicit_train_cfg and "loss_mode" in meta:
                self.train_cfg.loss_mode = str(meta["loss_mode"])
                                                      
                                                                
                                                    
                                                                        
                                                       
                                                                  
            self.train_cfg.risk_scalar_mode = str(
                meta.get("risk_scalar_mode", self.train_cfg.risk_scalar_mode)
            )
            self.train_cfg.anchor_mode = str(
                meta.get("anchor_mode", self.train_cfg.anchor_mode)
            )
            self.train_cfg.max_residual_m = float(
                meta.get("max_residual_m", self.train_cfg.max_residual_m)
            )
            if "max_pair_dist_m" in meta:
                _v = meta["max_pair_dist_m"]
                self.train_cfg.max_pair_dist_m = None if _v is None else float(_v)
            self.train_cfg.s0_m = float(meta.get("s0_m", self.train_cfg.s0_m))
            self.train_cfg.feature_l0_m = float(
                meta.get("feature_l0_m", self.train_cfg.feature_l0_m)
            )
            self.train_cfg.anchor_shape_eta = float(
                meta.get("anchor_shape_eta", self.train_cfg.anchor_shape_eta)
            )
            self.train_cfg.anchor_aspect_ratio_min = float(
                meta.get("anchor_aspect_ratio_min", self.train_cfg.anchor_aspect_ratio_min)
            )
            self.train_cfg.anchor_aspect_ratio_max = float(
                meta.get("anchor_aspect_ratio_max", self.train_cfg.anchor_aspect_ratio_max)
            )
            self.train_cfg.pixel_grazing_d0 = float(
                meta.get("pixel_grazing_d0", self.train_cfg.pixel_grazing_d0)
            )
                                                                      
                                                      
                                      
            if "sensitivity_clip_m" in meta:
                _v = meta["sensitivity_clip_m"]
                self.train_cfg.sensitivity_clip_m = None if _v is None else float(_v)
            if "extent_clip_m" in meta:
                _v = meta["extent_clip_m"]
                self.train_cfg.extent_clip_m = None if _v is None else float(_v)
        else:
            mlp = GeometryGuidedResidualMLP(max_residual_m=self.train_cfg.max_residual_m)

        self.training_camera_translation_corrections = coerce_camera_translation_corrections(
            loaded_meta.get(TRAINING_CAMERA_TRANSLATION_META_KEY)
        )
        self.camera_translation_corrections = coerce_camera_translation_corrections(
            camera_translation_corrections
        )
        # Runtime T_cam is an explicit CLI/runtime choice, never inherited from
        # checkpoint metadata. Drop old/runtime hint keys before rebuilding meta.
        loaded_meta.pop("camera_translation_corrections", None)
        loaded_meta.pop("runtime_camera_translation_corrections", None)
        loaded_meta.pop("runtime_camera_translation_corrections_hint", None)
        self.model = ResidualMLPModel(mlp, self.train_cfg)
        self._runtime_buffer: deque[AnchorFeatures] = deque(
            maxlen=self.online_cfg.max_buffer_obs
        )
        self._frames_since_learn = 0
        self._learn_count = 0
        # Track scenes for which a missing-correction warning has already been emitted,
        # so the message appears only once per (scene_id, camera_name) instead of per frame.
        self._warned_missing_corrections: set[tuple[str, str]] = set()
        # Residual invoke: default is training-overlap coverage from checkpoint
        # meta. configure_residual_invoke() may load a val JSON or force-all.
        self.force_all_residual = False
        self.residual_cameras: set[str] | None = None
        self.residual_invoke_source = "unset"
        self._meta: dict[str, Any] = {
            **loaded_meta,
            "risk_scalar_mode": self.train_cfg.risk_scalar_mode,
            "anchor_mode": self.train_cfg.anchor_mode,
            "max_residual_m": self.train_cfg.max_residual_m,
            "max_pair_dist_m": self.train_cfg.max_pair_dist_m,
            "s0_m": self.train_cfg.s0_m,
            "feature_l0_m": self.train_cfg.feature_l0_m,
            "anchor_shape_eta": self.train_cfg.anchor_shape_eta,
            "anchor_aspect_ratio_min": self.train_cfg.anchor_aspect_ratio_min,
            "anchor_aspect_ratio_max": self.train_cfg.anchor_aspect_ratio_max,
            "pixel_grazing_d0": self.train_cfg.pixel_grazing_d0,
            "sensitivity_clip_m": self.train_cfg.sensitivity_clip_m,
            "extent_clip_m": self.train_cfg.extent_clip_m,
            "lambda_mean_delta": self.train_cfg.lambda_mean_delta,
            "lambda_pair_center": self.train_cfg.lambda_pair_center,
            "loss_mode": self.train_cfg.loss_mode,
        }
        if self.training_camera_translation_corrections:
            self._meta[TRAINING_CAMERA_TRANSLATION_META_KEY] = serialize_camera_translation_corrections(
                self.training_camera_translation_corrections
            )
        if self.camera_translation_corrections:
            self._meta["runtime_camera_translation_corrections_hint"] = serialize_camera_translation_corrections(
                self.camera_translation_corrections
            )
        self.configure_residual_invoke()

    def train_overlap_cameras(self) -> set[str]:
        from pipeline.residual_invoke import cameras_on_overlap, overlap_from_meta

        meta_ov = overlap_from_meta(self._meta)
        if meta_ov:
            return cameras_on_overlap(meta_ov)
        return cameras_on_overlap(self.overlap_pairs)

    def configure_residual_invoke(
        self,
        *,
        acceptor_json: str | Path | None = None,
        force_all: bool = False,
    ) -> None:
        """Select cameras that receive ΔP. T_c is not gated here.

        force_all: shared head on every box (negative-transfer diagnostic).
        acceptor_json: val-fitted residual_cameras list.
        otherwise: cameras on the training overlap graph in checkpoint meta.
        """
        from pipeline.residual_invoke import load_residual_acceptor_json

        if force_all and acceptor_json:
            raise ValueError("pass only one of acceptor_json / force_all residual")
        if force_all:
            self.force_all_residual = True
            self.residual_cameras = None
            self.residual_invoke_source = "force_all"
            return
        self.force_all_residual = False
        if acceptor_json:
            blob = load_residual_acceptor_json(acceptor_json)
            self.residual_cameras = set(blob["_residual_cameras"])
            self.residual_invoke_source = f"json:{acceptor_json}"
            return
        cams = self.train_overlap_cameras()
        if cams:
            self.residual_cameras = cams
            self.residual_invoke_source = "train_overlap"
            return
        # Checkpoints without an overlap graph in meta keep the old apply-all
        # behavior; coverage cannot be inferred.
        self.residual_cameras = None
        self.residual_invoke_source = "no_train_overlap_meta"

    def residual_delta_allowed(self, camera_name: str | None) -> bool:
        if self.force_all_residual:
            return True
        if not camera_name:
            return False
        allow = self.residual_cameras
        if allow is None:
            return True
        return camera_name in allow

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        online_cfg: OnlineLearnConfig | None = None,
        overlap_pairs: set[tuple[str, str]] | None = None,
        camera_translation_corrections: CameraTranslationCorrections | None = None,
        save_checkpoint_path: Path | None = None,
    ) -> MVCResidualController:
        return cls(
            checkpoint_path=path,
            save_checkpoint_path=save_checkpoint_path,
            online_cfg=online_cfg,
            overlap_pairs=overlap_pairs,
            camera_translation_corrections=camera_translation_corrections,
        )

    def camera_translation(
        self,
        camera_name: str | None,
        scene_id: str | None = None,
    ) -> tuple[float, float]:
        found, result = camera_translation_lookup(
            self.camera_translation_corrections,
            camera_name,
            scene_id,
        )
        # Warn once when runtime corrections are loaded but do not cover this scene/camera,
        # so the caller is aware that T_cam falls back to (0,0) silently.
        if (
            self.camera_translation_corrections
            and camera_name
            and not found
        ):
            key = (str(scene_id or ""), str(camera_name))
            if key not in self._warned_missing_corrections:
                self._warned_missing_corrections.add(key)
                scene_label = f"scene={scene_id!r}" if scene_id else "scene=<unset>"
                print(
                    f"  [g2][] runtime fixed-translation projection JSON loaded but "
                    f"{scene_label} camera={camera_name!r} not found in the map — "
                    f"T_cam falls back to (0, 0) for this combination.  "
                    f"Check that --g2-camera-translation-json covers the active scene."
                )
        return result

    def _missing_runtime_translation_keys(
        self,
        pairs: list[MVCPair],
    ) -> list[tuple[str, str]]:
        missing: set[tuple[str, str]] = set()
        for pair in pairs:
            for obs in (pair.obs_a, pair.obs_b):
                trained_with_translation, _trained_delta = camera_translation_lookup(
                    self.training_camera_translation_corrections,
                    obs.camera_name,
                    obs.scene_id,
                )
                if not trained_with_translation:
                    continue
                found, _delta = camera_translation_lookup(
                    self.camera_translation_corrections,
                    obs.camera_name,
                    obs.scene_id,
                )
                if not found:
                    missing.add((str(obs.scene_id or ""), str(obs.camera_name)))
        return sorted(missing)

    def predict_residual(
        self,
        xyxy: np.ndarray,
        homography: np.ndarray,
        scale: float,
        frame_wh: tuple[int, int],
        *,
        camera_name: str | None = None,
        scene_id: str | None = None,
        world_geo_override: tuple[float, float] | None = None,
    ) -> ResidualPrediction:
        """P_final = P_geo + explicit runtime T_cam + tanh(MLP(F))*r_max."""
        anchor_xy, world_geo, feat, _risk = compute_geometry_base(
            xyxy,
            homography,
            scale,
            frame_wh,
            risk_scalar_mode=self.train_cfg.risk_scalar_mode,
            s0_m=self.train_cfg.s0_m,
            feature_l0_m=self.train_cfg.feature_l0_m,
            pixel_grazing_d0=self.train_cfg.pixel_grazing_d0,
            anchor_mode=self.train_cfg.anchor_mode,
            anchor_shape_eta=self.train_cfg.anchor_shape_eta,
            anchor_aspect_ratio_min=self.train_cfg.anchor_aspect_ratio_min,
            anchor_aspect_ratio_max=self.train_cfg.anchor_aspect_ratio_max,
            sensitivity_clip_m=self.train_cfg.sensitivity_clip_m,
            extent_clip_m=self.train_cfg.extent_clip_m,
        )
        delta_m = (0.0, 0.0)
        if self.residual_delta_allowed(camera_name):
            delta = self.model.predict_delta(feat)[0]
            delta_m = (float(delta[0]), float(delta[1]))
        base_geo = world_geo_override or world_geo
        tx, ty = self.camera_translation(camera_name, scene_id)
        corrected_geo = (float(base_geo[0] + tx), float(base_geo[1] + ty))
        world_final = (corrected_geo[0] + delta_m[0], corrected_geo[1] + delta_m[1])
        return ResidualPrediction(
            image_anchor_geo=anchor_xy,
            world_geo=world_geo,
            delta_m=delta_m,
            world_final=world_final,
            camera_translation_m=(tx, ty),
            world_corrected_geo=corrected_geo,
        )

    def record_runtime_observation(self, obs: RuntimeObservation) -> None:
        'clipped'
        if not self.online_cfg.enabled or obs.clipped:
            return
        self._runtime_buffer.append(
            self.view.observation_to_features(
                obs,
                risk_scalar_mode=self.train_cfg.risk_scalar_mode,
                s0_m=self.train_cfg.s0_m,
                feature_l0_m=self.train_cfg.feature_l0_m,
                pixel_grazing_d0=self.train_cfg.pixel_grazing_d0,
                anchor_mode=self.train_cfg.anchor_mode,
                anchor_shape_eta=self.train_cfg.anchor_shape_eta,
                anchor_aspect_ratio_min=self.train_cfg.anchor_aspect_ratio_min,
                anchor_aspect_ratio_max=self.train_cfg.anchor_aspect_ratio_max,
                sensitivity_clip_m=self.train_cfg.sensitivity_clip_m,
                extent_clip_m=self.train_cfg.extent_clip_m,
            )
        )
        # Keep runtime buffer in raw P_geo. Fixed translations are applied exactly
        # once after MVC mining, so nearest/off online modes cannot double-apply T_cam.

    def _mine_buffer_pairs(self) -> tuple[list[MVCPair], list[TemporalPair]]:
        obs_list = list(self._runtime_buffer)
        mvc = self.view.mine_mvc_pairs(
            obs_list,
            self.overlap_pairs,
            max_time_gap_sec=self.online_cfg.max_time_gap_sec,
            motion_compensation=self.train_cfg.motion_compensation,
            time_offsets_sec=_parse_time_offsets(self.train_cfg.time_offsets_sec),
            risk_scalar_mode=self.train_cfg.risk_scalar_mode,
            s0_m=self.train_cfg.s0_m,
            feature_l0_m=self.train_cfg.feature_l0_m,
            pixel_grazing_d0=self.train_cfg.pixel_grazing_d0,
            anchor_mode=self.train_cfg.anchor_mode,
            anchor_shape_eta=self.train_cfg.anchor_shape_eta,
            anchor_aspect_ratio_min=self.train_cfg.anchor_aspect_ratio_min,
            anchor_aspect_ratio_max=self.train_cfg.anchor_aspect_ratio_max,
            sensitivity_clip_m=self.train_cfg.sensitivity_clip_m,
            extent_clip_m=self.train_cfg.extent_clip_m,
        )
        mvc = apply_camera_translation_to_pairs(mvc, self.camera_translation_corrections)
        mvc = _filter_mvc_pairs_by_world_distance(mvc, self.train_cfg.max_pair_dist_m)
        temporal = self.view.mine_temporal_pairs(obs_list)
        return mvc, temporal

    def maybe_incremental_learn(self, *, force: bool = False) -> bool:
        'True'
        if not self.online_cfg.enabled:
            return False
        self._frames_since_learn += 1
        if not force and self._frames_since_learn < self.online_cfg.every_frames:
            return False
        self._frames_since_learn = 0

        mvc_pairs, temporal_pairs = self._mine_buffer_pairs()
        raw_pair_count = len(mvc_pairs)
        mvc_pairs = self.model.select_mvc_pairs(
            mvc_pairs,
            max_pairs=self.online_cfg.max_pairs_per_learn,
        )
        if len(mvc_pairs) < self.online_cfg.min_pairs:
            print(
                f"  [g2 online] skip: qualified_pairs={len(mvc_pairs)}/"
                f"{raw_pair_count} < min_pairs={self.online_cfg.min_pairs}; "
                f"teacher_gap>={self.train_cfg.min_teacher_confidence_gap:.3f}"
            )
            return False

        # Guard: if the MLP was trained after applying an accepted fixed-translation
        # projection but active online pairs are not covered by the same runtime
        # projection map, online supervision runs on raw/partially projected P_geo.
        # That is inconsistent with the offline pairs the model learned from.
        missing_runtime_keys = (
            self._missing_runtime_translation_keys(mvc_pairs)
            if self.training_camera_translation_corrections
            else []
        )
        if missing_runtime_keys:
            preview = ", ".join(
                f"{scene or '<unset>'}:{camera}"
                for scene, camera in missing_runtime_keys[:6]
            )
            if len(missing_runtime_keys) > 6:
                preview += f", ... (+{len(missing_runtime_keys) - 6})"
            print(
                "  [g2 online] skip: checkpoint was trained with an accepted "
                "fixed-translation projection for some active scene/camera keys, but runtime "
                "--g2-camera-translation-json does not cover them "
                f"({preview}). Online supervision would run "
                "on raw or partially projected P_geo, inconsistent with the "
                "offline training. Pass a scene-aware projection JSON covering these "
                "scene/camera keys to enable online learning."
            )
            return False

        preflight = diagnose_low_order_bias(
            mvc_pairs,
            pair_source="online_selected_mvc_pairs",
            max_residual_m=self.train_cfg.max_residual_m,
        )
        if preflight.get("status") == "warning":
            worst = max(
                preflight.get("scene_camera_pairs", []) or preflight.get("camera_pairs", []),
                key=lambda item: float(item.get("mean_bias_norm_m", 0.0)),
                default=None,
            )
            if worst:
                label = (
                    f"{worst.get('scene_id', '')} {worst.get('camera_a')}:{worst.get('camera_b')}"
                ).strip()
                detail = (
                    f"{label} |mean|={float(worst.get('mean_bias_norm_m', 0.0)):.3f}m "
                    f"translation_explained="
                    f"{100.0 * float(worst.get('translation_explained_frac', 0.0)):.1f}%"
                )
            else:
                detail = "fixed-translation projection candidate detected"
            print(
                "  [g2 online] skip: label-free preflight found a fixed-translation "
                "projection candidate "
                f"({detail}). Run pipeline/g2_low_order_correction.py fit on logged data "
                "before online learning. Retrain with --camera-translation-json if the MLP "
                "should learn residuals after applying the accepted projection; pass "
                "--g2-camera-translation-json only if runtime world coordinates should "
                "include that accepted fixed-translation projection."
            )
            return False
        if preflight.get("status") == "caution":
            print(
                '[g2 online][] preflight=caution; continuing online learn,'
                "but inspect logged data if this repeats."
            )

        lr = self.train_cfg.lr * self.online_cfg.lr_scale
        t0 = time.perf_counter()
        stats = self.model.train_offline(
            mvc_pairs,
            temporal_pairs,
            epochs=self.online_cfg.epochs,
            lr=lr,
        )
        self._learn_count += 1
        elapsed = time.perf_counter() - t0
        print(
            f"  [g2 online] #{self._learn_count}  pairs={len(mvc_pairs)}  "
            f"epochs={self.online_cfg.epochs}  loss={stats.get('final_loss', 0):.4f}  "
            f"elapsed={elapsed:.2f}s"
        )
        if self.save_checkpoint_path is not None:
            self.save_checkpoint(extra={"online_learn_count": self._learn_count})
        return True

    def save_checkpoint(self, extra: dict[str, Any] | None = None) -> bool:
        if not self.online_cfg.enabled or self.save_checkpoint_path is None:
            return False
        meta = {
            **self._meta,
            "train_config": self.train_cfg.__dict__,
            "online_config": self.online_cfg.__dict__,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if extra:
            meta.update(extra)
        save_checkpoint(self.save_checkpoint_path, self.model.mlp, meta)
        return True

    def flush_learn(self) -> bool:
        'See the HSG-SRL reproduction notes.'
        if not self.online_cfg.enabled:
            return False
        return self.maybe_incremental_learn(force=True)


        
ResidualAnchorCorrector = MVCResidualController


# ── CLI ─────────────────────────────────────────────────────────────────────


def _parse_overlap_pairs(raw: str) -> set[tuple[str, str]]:
    '``A:B,B:C`` JSON ``[["A", "B"], ["B", "C"]]``'
    raw = raw.strip()
    if not raw:
        return set()
    if raw.startswith("["):
        data = json.loads(raw)
        return {tuple(sorted((str(a), str(b)))) for a, b in data}

    out: set[tuple[str, str]] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        sep = ":" if ":" in item else "-"
        parts = [p.strip() for p in item.split(sep) if p.strip()]
        if len(parts) != 2:
            raise ValueError(f" overlap pair: {item!r}")
        out.add(tuple(sorted((parts[0], parts[1]))))
    return out


def _parse_time_offsets(raw: dict[str, float] | None) -> dict[tuple[str, str], float]:
    '``{"camA:camB": offset_sec}`` ``mine_mvc_pairs`` tuple-key \n\n ``offset`` camB camA ``overlap_camera_pairs``\n ``mine_mvc_pairs``'
    if not raw:
        return {}
    out: dict[tuple[str, str], float] = {}
    for key, value in raw.items():
        sep = ":" if ":" in key else "-"
        parts = [p.strip() for p in key.split(sep) if p.strip()]
        if len(parts) != 2:
            raise ValueError(f" time-offsets key: {key!r} 'camA:camB'")
        out[(parts[0], parts[1])] = float(value)
    return out


def _infer_and_calib_wh_from_db(
    conn: sqlite3.Connection,
    scene_id: str,
    batch_id: str,
    camera_names: list[str],
) -> tuple[dict[str, tuple[int, int] | None], dict[str, tuple[int, int] | None]]:
    'trajectory_camera_config \n\n (infer_wh_by_cam, calib_wh_by_cam) None batch \n CLI schema'
    infer: dict[str, tuple[int, int] | None] = {cam: None for cam in camera_names}
    calib: dict[str, tuple[int, int] | None] = {cam: None for cam in camera_names}
    rows = conn.execute(
        "SELECT camera_name, infer_w, infer_h, calib_w, calib_h "
        "FROM trajectory_camera_config WHERE scene_id=? AND batch_id=?",
        (scene_id, batch_id),
    ).fetchall()
    for name, iw, ih, cw, ch in rows:
        if name in infer:
            if iw and ih and iw > 0 and ih > 0:
                infer[name] = (int(iw), int(ih))
            if cw and ch and cw > 0 and ch > 0:
                calib[name] = (int(cw), int(ch))
    return infer, calib


def _frame_wh_by_camera_from_db(
    conn: sqlite3.Connection,
    scene_id: str,
    batch_id: str,
    camera_names: list[str],
    default_wh: tuple[int, int] | None = None,
) -> dict[str, tuple[int, int] | None]:
    'scene/batch trajectory_camera_config'
    infer, _ = _infer_and_calib_wh_from_db(conn, scene_id, batch_id, camera_names)
    return {cam: infer.get(cam) or default_wh for cam in camera_names}



def _filter_mvc_pairs_by_world_distance(
    pairs: list[MVCPair],
    max_dist_m: float | None,
) -> list[MVCPair]:
    if max_dist_m is None:
        return pairs
    out: list[MVCPair] = []
    for pair in pairs:
        ax, ay = pair.obs_a.world_geo
        bx, by = pair.obs_b.world_geo
        if float(np.hypot(ax - bx, ay - by)) <= max_dist_m:
            out.append(pair)
    return out


@dataclass(frozen=True)
class TrainingSplitFilter:
    path: str
    split_names: tuple[str, ...]
    source_ids_by_scene: dict[str, set[str]]
    gt_ids_by_source: dict[str, str]
    row_count: int


def _csv_row_source_detection_id(row: dict[str, str], batch_id: str) -> str:
    sid = str(row.get("source_detection_id", "")).strip()
    if sid:
        return sid
    scene = str(row.get("scene_id", "")).strip()
    camera = str(row.get("camera_id", row.get("camera_name", row.get("camera", "")))).strip()
    local = str(row.get("local_track_id", "")).strip()
    frame = str(row.get("frame_index", "")).strip()
    row_batch = str(row.get("batch_id", "")).strip() or batch_id
    if not (scene and camera and local and frame and row_batch):
        return ""
    return f"{scene}|{row_batch}|{camera}|{local}|{frame}"


def _csv_row_source_detection_ids(row: dict[str, str], batch_id: str) -> list[str]:
    ids: list[str] = []
    sid = _csv_row_source_detection_id(row, batch_id)
    if sid:
        ids.append(sid)
    scene = str(row.get("scene_id", "")).strip()
    camera = str(row.get("camera_id", row.get("camera_name", row.get("camera", "")))).strip()
    local = str(row.get("local_track_id", "")).strip()
    frame = str(row.get("frame_index", "")).strip()
    if scene and camera and local and frame and batch_id:
        normalized = f"{scene}|{batch_id}|{camera}|{local}|{frame}"
        if normalized not in ids:
            ids.append(normalized)
    return ids


def _observation_source_detection_id(obs: AnchorFeatures, scene_id: str, batch_id: str) -> str:
    if obs.frame_index is None:
        return ""
    scene = obs.scene_id or scene_id
    return f"{scene}|{batch_id}|{obs.camera_name}|{obs.local_track_id}|{obs.frame_index}"


def _load_training_split_filter(
    manifest_csv: str | Path | None,
    split_names_raw: str | None,
    *,
    scene_ids: list[str],
    batch_id: str,
) -> TrainingSplitFilter | None:
    if not manifest_csv:
        return None
    path = resolve_project_path(manifest_csv)
    split_names = tuple(s.strip() for s in str(split_names_raw or "train").split(",") if s.strip()) or ("train",)
    wanted_scenes = set(scene_ids)
    source_ids_by_scene: dict[str, set[str]] = defaultdict(set)
    gt_ids_by_source: dict[str, str] = {}
    row_count = 0
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scene = str(row.get("scene_id", "")).strip()
            if wanted_scenes and scene not in wanted_scenes:
                continue
            split = str(row.get("split", "")).strip()
            if split_names and split not in split_names:
                continue
            sids = _csv_row_source_detection_ids(row, batch_id)
            if not sids:
                continue
            gt = str(row.get("gt_global_id", row.get("matched_gt_id", ""))).strip()
            for sid in sids:
                source_ids_by_scene[scene].add(sid)
                if gt:
                    gt_ids_by_source[sid] = gt
            row_count += 1
    return TrainingSplitFilter(
        path=str(path),
        split_names=split_names,
        source_ids_by_scene=dict(source_ids_by_scene),
        gt_ids_by_source=gt_ids_by_source,
        row_count=row_count,
    )


def _filter_observations_for_training_protocol(
    observations: list[AnchorFeatures],
    *,
    scene_id: str,
    batch_id: str,
    split_filter: TrainingSplitFilter | None,
    time_start: float | None,
    time_end: float | None,
) -> tuple[dict[str, int], list[AnchorFeatures]]:
    before = len(observations)
    out: list[AnchorFeatures] = []
    for obs in observations:
        if time_start is not None and obs.timestamp_sec < time_start:
            continue
        if time_end is not None and obs.timestamp_sec >= time_end:
            continue
        out.append(obs)
    after_time = len(out)
    if split_filter is not None:
        allowed = split_filter.source_ids_by_scene.get(scene_id, set())
        out = [
            obs
            for obs in out
            if _observation_source_detection_id(obs, scene_id, batch_id) in allowed
        ]
    return {
        "observations_before_protocol_filter": before,
        "observations_after_time_filter": after_time,
        "observations_after_split_filter": len(out),
    }, out


def _gt_identity_leakage_report(
    observations: list[AnchorFeatures],
    *,
    scene_id: str,
    batch_id: str,
    split_filter: TrainingSplitFilter | None,
) -> dict[str, float]:
    if split_filter is None:
        return {"compared": 0.0, "matches": 0.0, "match_fraction": 0.0}
    compared = 0
    matches = 0
    for obs in observations:
        sid = _observation_source_detection_id(obs, scene_id, batch_id)
        gt = split_filter.gt_ids_by_source.get(sid)
        if not gt:
            continue
        compared += 1
        if str(obs.global_id) == str(gt):
            matches += 1
    frac = (matches / compared) if compared else 0.0
    return {"compared": float(compared), "matches": float(matches), "match_fraction": float(frac)}


TRAINING_CAMERA_TRANSLATION_META_KEY = "training_camera_translation_corrections"

# Ordered acceptance levels for --camera-translation-acceptance.
# Higher rank = stricter requirement; any = no filtering.
_LO_STATUS_RANK: dict[str, int] = {
    "not_recommended": 0,
    "caution": 1,
    "recommended": 2,
}
CAMERA_TRANSLATION_ACCEPTANCE_CHOICES = ["recommended", "caution", "any"]


@dataclass
class CameraTranslationCorrections:
    """Scene-aware fixed 2D translations applied before the residual MLP."""

    by_scene: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    source_path: str | None = None
    # Scenes present in the source JSON before acceptance filtering.
    source_scenes: frozenset[str] = field(default_factory=frozenset)

    def __bool__(self) -> bool:
        return bool(self.by_scene)

    def __len__(self) -> int:
        return self.camera_count()

    def camera_count(self) -> int:
        cameras: set[str] = set()
        for scene_map in self.by_scene.values():
            cameras.update(scene_map)
        return len(cameras)

    def scene_count(self) -> int:
        return len(self.by_scene)


def _translation_entry_to_pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        if "translation_m" in value:
            return _translation_entry_to_pair(value["translation_m"])
        dx = value.get("dx", value.get("x", value.get("dx_m")))
        dy = value.get("dy", value.get("y", value.get("dy_m")))
        if dx is None or dy is None:
            return None
        return (float(dx), float(dy))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    return None


def _parse_translation_map(data: Any, *, context: str) -> dict[str, tuple[float, float]]:
    if not data:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{context} must map camera -> [dx, dy]")
    out: dict[str, tuple[float, float]] = {}
    for camera, value in data.items():
        pair = _translation_entry_to_pair(value)
        if pair is None:
            raise ValueError(f"invalid translation entry for camera {camera!r}: {value!r}")
        dx, dy = pair
        if not (np.isfinite(dx) and np.isfinite(dy)):
            raise ValueError(f"non-finite translation entry for camera {camera!r}: {value!r}")
        out[str(camera)] = (float(dx), float(dy))
    return out


def _parse_scene_translation_map(data: Any) -> dict[str, dict[str, tuple[float, float]]]:
    if not data:
        return {}
    if not isinstance(data, dict):
        raise ValueError("corrections_m_by_scene must map scene -> camera translations")
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for scene_id, scene_map in data.items():
        parsed = _parse_translation_map(
            scene_map,
            context=f"corrections for scene {scene_id!r}",
        )
        if parsed:
            out[str(scene_id)] = parsed
    return out


def coerce_camera_translation_corrections(data: Any) -> CameraTranslationCorrections:
    """Parse scene-aware accepted fixed-translation projections."""
    if not data:
        return CameraTranslationCorrections()
    if isinstance(data, CameraTranslationCorrections):
        return data
    if isinstance(data, (str, Path)):
        return load_camera_translation_corrections(data)
    if not isinstance(data, dict):
        raise ValueError("fixed-translation projection JSON must be a JSON object")

    if TRAINING_CAMERA_TRANSLATION_META_KEY in data:
        return coerce_camera_translation_corrections(data[TRAINING_CAMERA_TRANSLATION_META_KEY])

    by_scene_raw = data.get("corrections_m_by_scene")
    if by_scene_raw is None:
        raise ValueError("fixed-translation projection JSON must contain corrections_m_by_scene")

    by_scene = _parse_scene_translation_map(by_scene_raw)
    return CameraTranslationCorrections(
        by_scene=by_scene,
        source_path=str(data.get("source_path")) if data.get("source_path") else None,
        source_scenes=frozenset(by_scene),
    )


def load_camera_translation_corrections(
    path: str | Path | None,
    acceptance: str = "recommended",
) -> CameraTranslationCorrections:
    """Load per-scene fixed-translation projection artifacts from *path*.

    *acceptance* controls which scenes are kept based on the per-scene
    ``diagnostics_by_scene[scene].decision.status`` field written by
    ``g2_low_order_correction.py fit``:

    * ``"recommended"`` – keep only scenes with status ``recommended``
    * ``"caution"``     – keep scenes with ``recommended`` or ``caution``
    * ``"any"``         – keep all scenes

    The default is ``"recommended"``.  Pass ``"any"`` when backward-compatible
    behaviour (apply everything in the JSON) is required.

    For ``"recommended"`` and ``"caution"``, scenes with missing or unknown
    diagnostic status are skipped (fail closed). Use ``"any"`` explicitly
    for legacy JSON files without ``diagnostics_by_scene``.
    """
    if not path:
        return CameraTranslationCorrections()
    if acceptance not in CAMERA_TRANSLATION_ACCEPTANCE_CHOICES:
        raise ValueError(
            f"invalid camera translation acceptance {acceptance!r}; "
            f"expected one of {CAMERA_TRANSLATION_ACCEPTANCE_CHOICES}"
        )
    p = resolve_project_path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    corrections = coerce_camera_translation_corrections(payload)
    corrections.source_path = str(p)

    if acceptance != "any" and corrections.by_scene:
        min_rank = _LO_STATUS_RANK[acceptance]
        raw_diagnostics = payload.get("diagnostics_by_scene")
        diagnostics: dict[str, Any] = (
            raw_diagnostics if isinstance(raw_diagnostics, dict) else {}
        )
        kept: list[str] = []
        skipped: list[str] = []
        for scene_id in list(corrections.by_scene):
            scene_diag = diagnostics.get(scene_id)
            decision = scene_diag.get("decision", {}) if isinstance(scene_diag, dict) else {}
            status = decision.get("status", "") if isinstance(decision, dict) else ""
            rank = _LO_STATUS_RANK.get(status, -1)
            if rank >= min_rank:
                kept.append(scene_id)
            else:
                del corrections.by_scene[scene_id]
                skipped.append(f"{scene_id}({status or 'missing_status'})")
        print(
            f"  [lo_correct] acceptance={acceptance!r}: "
            f"applied={len(kept)} skipped={len(skipped)}"
        )
        if kept:
            print(f"  [lo_correct] applied: {', '.join(kept)}")
        if skipped:
            print(f"  [lo_correct] skipped (T_cam=0): {', '.join(skipped)}")

    return corrections


def serialize_camera_translation_corrections(
    corrections: CameraTranslationCorrections,
) -> dict[str, Any]:
    corrections = coerce_camera_translation_corrections(corrections)
    if not corrections:
        return {}
    payload: dict[str, Any] = {
        "version": "g2_camera_translation_correction_v2",
        "unit": "meter",
    }
    if corrections.by_scene:
        payload["corrections_m_by_scene"] = {
            str(scene_id): {
                str(camera): {"dx": float(delta[0]), "dy": float(delta[1])}
                for camera, delta in sorted(scene_map.items())
            }
            for scene_id, scene_map in sorted(corrections.by_scene.items())
        }
    if corrections.source_path:
        payload["source_path"] = corrections.source_path
    return payload


def camera_translation_for(
    corrections: CameraTranslationCorrections,
    camera_name: str | None,
    scene_id: str | None = None,
) -> tuple[float, float]:
    return camera_translation_lookup(corrections, camera_name, scene_id)[1]


def camera_translation_lookup(
    corrections: CameraTranslationCorrections,
    camera_name: str | None,
    scene_id: str | None = None,
) -> tuple[bool, tuple[float, float]]:
    if not camera_name:
        return False, (0.0, 0.0)
    corrections = coerce_camera_translation_corrections(corrections)
    camera = str(camera_name)
    if scene_id:
        scene_map = corrections.by_scene.get(str(scene_id))
        if scene_map is not None and camera in scene_map:
            return True, scene_map[camera]
    return False, (0.0, 0.0)


def apply_typed_edge_policy(
    pairs: list[MVCPair],
    policy: dict[str, Any] | None,
    *,
    cap_ratio: float = 0.0,
) -> list[MVCPair]:
    """Apply per-edge loss mode / sampling weight. Weight 0 drops the pair.

    Teacher/edge weights stay in loss and sampling. They are not MLP inputs.
    """
    if not policy:
        return pairs
    out: list[MVCPair] = []
    for pair in pairs:
        key = ":".join(sorted((pair.obs_a.camera_name, pair.obs_b.camera_name)))
        rec = policy.get(key) if isinstance(policy, dict) else None
        if rec is None or not isinstance(rec, dict):
            out.append(pair)
            continue
        weight = float(rec.get("weight", 1.0))
        if weight <= 0.0:
            continue
        gap = rec.get("require_kappa_gap")
        out.append(
            replace(
                pair,
                sample_weight=weight,
                loss_mode_override=rec.get("loss_mode"),
                require_kappa_gap=None if gap is None else bool(gap),
            )
        )
    if cap_ratio and cap_ratio > 0.0 and out:
        sums: dict[str, float] = defaultdict(float)
        for pair in out:
            key = ":".join(sorted((pair.obs_a.camera_name, pair.obs_b.camera_name)))
            sums[key] += float(pair.sample_weight)
        med = float(np.median(list(sums.values()))) if sums else 0.0
        cap = float(cap_ratio) * max(med, 1e-6)
        scale = {
            key: (min(1.0, cap / val) if val > cap else 1.0)
            for key, val in sums.items()
        }
        out = [
            replace(
                pair,
                sample_weight=float(pair.sample_weight) * scale[
                    ":".join(sorted((pair.obs_a.camera_name, pair.obs_b.camera_name)))
                ],
            )
            for pair in out
        ]
    return out


def apply_camera_translation_to_anchor(
    obs: AnchorFeatures,
    corrections: CameraTranslationCorrections,
) -> AnchorFeatures:
    dx, dy = camera_translation_for(corrections, obs.camera_name, obs.scene_id)
    if dx == 0.0 and dy == 0.0:
        return obs
    x, y = obs.world_geo
    return replace(obs, world_geo=(float(x + dx), float(y + dy)))


def apply_camera_translation_to_pairs(
    pairs: list[MVCPair],
    corrections: CameraTranslationCorrections,
) -> list[MVCPair]:
    if not corrections:
        return pairs
    return [
        replace(
            pair,
            obs_a=apply_camera_translation_to_anchor(pair.obs_a, corrections),
            obs_b=apply_camera_translation_to_anchor(pair.obs_b, corrections),
        )
        for pair in pairs
    ]


# Published numeric configuration. scripts/train_recording.py does not use
# this command line. cmd_train rejects a run that changes these values.
_PAPER_TRAIN = {
    "max_residual_m": 1.0,
    "max_pair_dist_m": 2.0,
    "lambda_reg": 0.10,
    "lambda_smooth": 0.02,
    "lambda_mean_delta": 1.0,
    "lambda_pair_center": 0.50,
    "min_mvc_pairs": 32,
}


def _require_paper_train_flags(args: argparse.Namespace) -> None:
    current = {key: getattr(args, key) for key in _PAPER_TRAIN}
    if current != _PAPER_TRAIN:
        differ = [
            f"{key}={current[key]} (published {_PAPER_TRAIN[key]})"
            for key in _PAPER_TRAIN
            if current[key] != _PAPER_TRAIN[key]
        ]
        raise SystemExit(
            "This train command cannot replace scripts/train_recording.py.\n"
            "The published trainer fits the frozen pairs and does not call this "
            "command line. This database trainer still requires --split-manifest.\n"
            "Differing values: " + ", ".join(differ)
        )


def cmd_train(args: argparse.Namespace) -> None:
    _require_paper_train_flags(args)
    _require_torch()
    if getattr(args, "seed", None) is not None:
        seed = int(args.seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if args.frame_width < 1 or args.frame_height < 1:
        raise SystemExit('--frame-width --frame-height >= 1')
    if not (0 < args.pixel_grazing_d0 <= 1.0):
        raise SystemExit(
            f"--pixel-grazing-d0  (0, 1]  {args.pixel_grazing_d0}"
        )
    if args.risk_scalar not in RISK_SCALAR_MODES:
        raise SystemExit(f"--risk-scalar  {RISK_SCALAR_MODES} ")
    if args.s0_m <= 0:
        raise SystemExit('--s0-m > 0')
    if args.feature_l0_m <= 0:
        raise SystemExit('--feature-l0-m > 0')
    if args.anchor_shape_eta < 0 or args.anchor_shape_eta > 1:
        raise SystemExit('--anchor-shape-eta [0, 1]')
    if args.anchor_aspect_ratio_min <= 0 or args.anchor_aspect_ratio_max <= 0:
        raise SystemExit('--anchor-aspect-ratio-min/max > 0')
    if args.anchor_aspect_ratio_min > args.anchor_aspect_ratio_max:
        raise SystemExit('--anchor-aspect-ratio-min <= --anchor-aspect-ratio-max')
    if args.clip_margin < 0:
        raise SystemExit('--clip-margin >= 0')
    if not 0 <= args.clip_margin_ratio < 0.5:
        raise SystemExit('--clip-margin-ratio [0, 0.5)')
    if args.max_pair_dist_m is not None and args.max_pair_dist_m <= 0:
        raise SystemExit('--max-pair-dist-m > 0')
    if args.min_mvc_pairs < 1:
        raise SystemExit('--min-mvc-pairs >= 1')

    if args.max_pairs is not None and args.max_pairs < 1:
        raise SystemExit("--max-pairs must be >= 1 when provided")
    if not args.split_manifest and not args.allow_unsplit_training:
        raise SystemExit(
            "This train command cannot replace scripts/train_recording.py. "
            "The published trainer fits the frozen pairs and does not call this "
            "command line. This database trainer still requires --split-manifest "
            "so split filtering and the ground-truth identity guard are active. "
            "Pass --allow-unsplit-training only for a legacy diagnostic."
        )
    if args.time_start is not None and args.time_end is not None and args.time_start >= args.time_end:
        raise SystemExit("--time-start must be smaller than --time-end")

    scene_ids = [s.strip() for s in args.scene.split(",") if s.strip()]
    if not scene_ids:
        raise SystemExit('--scene')

    split_filter = _load_training_split_filter(
        args.split_manifest,
        args.train_splits,
        scene_ids=scene_ids,
        batch_id=args.batch,
    )
    if split_filter is not None:
        if split_filter.row_count == 0:
            raise SystemExit(
                "FAB2 split manifest contains no rows for "
                f"scene={args.scene} batch={args.batch} split={','.join(split_filter.split_names)}"
            )
        print(
            "  [FAB2] training split filter loaded: "
            f"{split_filter.path} split={','.join(split_filter.split_names)} rows={split_filter.row_count}"
        )

    time_offsets_raw: dict[str, float] = (
        json.loads(args.time_offsets_json) if args.time_offsets_json else {}
    )
    time_offsets = _parse_time_offsets(time_offsets_raw)

    db_path = resolve_project_path(args.db)
    conn = sqlite3.connect(str(db_path))
    require_trajectory_batch_schema(conn)
    stage_run_id = start_batch_stage(
        conn,
        scene_ids,
        args.batch,
        "g2_train",
        {
            "out": str(resolve_project_path(args.out)),
            "camera_translation_json": args.camera_translation_json,
            "split_manifest": split_filter.path if split_filter else "",
            "train_splits": list(split_filter.split_names) if split_filter else [],
            "time_start": args.time_start,
            "time_end": args.time_end,
        },
    )

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        lambda_reg=args.lambda_reg,
        lambda_smooth=args.lambda_smooth,
        lambda_mean_delta=args.lambda_mean_delta,
        lambda_pair_center=args.lambda_pair_center,
        max_residual_m=args.max_residual_m,
        max_pair_dist_m=args.max_pair_dist_m,
        risk_scalar_mode=args.risk_scalar,
        anchor_mode=args.anchor_mode,
        s0_m=args.s0_m,
        feature_l0_m=args.feature_l0_m,
        anchor_shape_eta=args.anchor_shape_eta,
        anchor_aspect_ratio_min=args.anchor_aspect_ratio_min,
        anchor_aspect_ratio_max=args.anchor_aspect_ratio_max,
        pixel_grazing_d0=args.pixel_grazing_d0,
        sensitivity_clip_m=args.sensitivity_clip_m,
        extent_clip_m=args.extent_clip_m,
        motion_compensation=args.motion_compensation,
        time_offsets_sec=time_offsets_raw,
        loss_mode=args.loss_mode,
        min_teacher_confidence=args.min_teacher_confidence,
        min_teacher_confidence_gap=args.min_teacher_confidence_gap,
        teacher_confidence_power=args.teacher_confidence_power,
        student_weight_floor=args.student_weight_floor,
        reg_confidence_floor=args.reg_confidence_floor,
    )
    if cfg.max_pair_dist_m is not None:
        pair_close_bound = 2.0 * float(np.sqrt(2.0)) * cfg.max_residual_m
        if cfg.max_pair_dist_m > pair_close_bound * 1.5:
            print(
                '[] --max-pair-dist-m'
                f"{cfg.max_pair_dist_m:.2f}m vs 2*sqrt(2)*r_max ~= {pair_close_bound:.2f}m"
                'pair MLP tanh'
            )
    frame_wh = (int(args.frame_width), int(args.frame_height))
    camera_translation_corrections = load_camera_translation_corrections(
        args.camera_translation_json,
        acceptance=getattr(args, "camera_translation_acceptance", "recommended"),
    )
    if camera_translation_corrections or camera_translation_corrections.source_scenes:
        print(
            "  [g2] accepted fixed-translation projections loaded: "
            f"{camera_translation_corrections.camera_count()} camera(s), "
            f"{camera_translation_corrections.scene_count()} scene map(s)"
        )
        # Acceptance filtering already prints applied/skipped; only tip about
        # training scenes that were never present in the source JSON at all.
        absent = [
            s for s in scene_ids
            if s not in camera_translation_corrections.source_scenes
        ]
        if absent:
            print(
                '[g2][] camera translation JSON scene'
                f" T_cam=0: {','.join(absent)}"
            )
    view = GeometryDataView()

                                          
    global_camera_names = [s.strip() for s in args.cameras.split(",") if s.strip()]
                                                                    
    global_overlap: set[tuple[str, str]] | None = (
        _parse_overlap_pairs(args.overlap_pairs) if args.overlap_pairs else None
    )

    all_mvc_pairs: list[MVCPair] = []
    all_temporal_pairs: list[TemporalPair] = []
    per_scene_meta: list[dict[str, Any]] = []

    for scene_id in scene_ids:
        print(f"\n─── scene={scene_id} ───")

        camera_names = global_camera_names or [
            r[0]
            for r in conn.execute(
                """
                SELECT DISTINCT camera_name FROM trajectory_observations
                 WHERE scene_id = ? AND batch_id = ? ORDER BY camera_name
                """,
                (scene_id, args.batch),
            ).fetchall()
        ]
        if len(camera_names) < 2:
            print(f"  [] scene={scene_id}  {len(camera_names)} < 2 MVC ")
            continue

        try:
            calibrations = view.load_camera_calibrations(conn, scene_id, camera_names)
        except RuntimeError as exc:
            print(f"  [] scene={scene_id} {exc}")
            continue

                                                                 
        db_infer_wh, db_calib_wh = _infer_and_calib_wh_from_db(
            conn, scene_id, args.batch, camera_names
        )
        frame_wh_by_camera = {
            cam: db_infer_wh.get(cam) or frame_wh
            for cam in camera_names
        }
                                         
                                                       
        adapted_calibrations: dict[str, tuple[np.ndarray, float]] = {}
        for cam, (H_cal, sc) in calibrations.items():
            infer_wh_cam = frame_wh_by_camera.get(cam)
            calib_wh_cam = db_calib_wh.get(cam)
            if infer_wh_cam and calib_wh_cam and calib_wh_cam != infer_wh_cam:
                H_cal = adapt_homography_to_resolution(H_cal, calib_wh_cam, infer_wh_cam)
            adapted_calibrations[cam] = (H_cal, sc)
        observations = list(
            view.iter_observations_from_db(
                conn,
                scene_id,
                args.batch,
                camera_names,
                adapted_calibrations,
                frame_wh_by_camera=frame_wh_by_camera,
                risk_scalar_mode=cfg.risk_scalar_mode,
                s0_m=cfg.s0_m,
                feature_l0_m=cfg.feature_l0_m,
                pixel_grazing_d0=cfg.pixel_grazing_d0,
                anchor_mode=cfg.anchor_mode,
                anchor_shape_eta=cfg.anchor_shape_eta,
                anchor_aspect_ratio_min=cfg.anchor_aspect_ratio_min,
                anchor_aspect_ratio_max=cfg.anchor_aspect_ratio_max,
                sensitivity_clip_m=cfg.sensitivity_clip_m,
                extent_clip_m=cfg.extent_clip_m,
                clip_margin_px=args.clip_margin,
                clip_margin_ratio=args.clip_margin_ratio,
            )
        )
        total_db_obs = int(
            conn.execute(
                f"""
                SELECT COUNT(*) FROM trajectory_observations
                 WHERE scene_id=? AND batch_id=? AND camera_name IN ({','.join('?' for _ in camera_names)})
                """,
                (scene_id, args.batch, *camera_names),
            ).fetchone()[0]
        )
        protocol_stats, observations = _filter_observations_for_training_protocol(
            observations,
            scene_id=scene_id,
            batch_id=args.batch,
            split_filter=split_filter,
            time_start=args.time_start,
            time_end=args.time_end,
        )
        leak_report = _gt_identity_leakage_report(
            observations,
            scene_id=scene_id,
            batch_id=args.batch,
            split_filter=split_filter,
        )
        if (
            split_filter is not None
            and not args.allow_gt_identity_mining
            and leak_report["compared"] >= max(1, min(32, len(observations)))
            and leak_report["match_fraction"] >= 0.95
        ):
            raise SystemExit(
                "FAB2 protocol guard: MVC mining appears to be using GT identities "
                f"for scene={scene_id} ({int(leak_report['matches'])}/"
                f"{int(leak_report['compared'])} observations have global_id == gt_global_id). "
                "Run importer/manifest first, then re-merge with pseudo identities before "
                "training, or pass --allow-gt-identity-mining only for supervised smoke tests."
            )
        print(
            f"  observations={len(observations)}/{total_db_obs}  "
            f"clip_filter={total_db_obs - protocol_stats['observations_before_protocol_filter']}  "
            f"protocol_filter={protocol_stats['observations_before_protocol_filter'] - len(observations)}"
        )

        if global_overlap is not None:
            overlap = global_overlap
            overlap_source = "--overlap-pairs"
        else:
            overlap = load_scene_overlap_pairs(conn, scene_id, camera_names)
            overlap_source = "db:scene_overlap_pairs"
        if not overlap:
            raise RuntimeError(
                f"scene={scene_id}: DB  scene_overlap_pairs"
                'import / tracking --overlap-pairs'
            )

        mvc = view.mine_mvc_pairs(
            observations,
            overlap,
            max_time_gap_sec=args.max_time_gap,
            motion_compensation=cfg.motion_compensation,
            time_offsets_sec=time_offsets,
            risk_scalar_mode=cfg.risk_scalar_mode,
            s0_m=cfg.s0_m,
            feature_l0_m=cfg.feature_l0_m,
            pixel_grazing_d0=cfg.pixel_grazing_d0,
            anchor_mode=cfg.anchor_mode,
            anchor_shape_eta=cfg.anchor_shape_eta,
            anchor_aspect_ratio_min=cfg.anchor_aspect_ratio_min,
            anchor_aspect_ratio_max=cfg.anchor_aspect_ratio_max,
            sensitivity_clip_m=cfg.sensitivity_clip_m,
            extent_clip_m=cfg.extent_clip_m,
        )
        raw_count = len(mvc)
        mvc = apply_camera_translation_to_pairs(mvc, camera_translation_corrections)
        mvc = _filter_mvc_pairs_by_world_distance(mvc, cfg.max_pair_dist_m)
        dist_count = len(mvc)
        temporal = view.mine_temporal_pairs(observations)

        dist_info = f"/{cfg.max_pair_dist_m:g}m" if cfg.max_pair_dist_m is not None else ""
        print(
            f"  overlap_pairs={len(overlap)}  mvc_raw={raw_count}  "
            f"mvc_dist{dist_info}={dist_count}  temporal={len(temporal)}  "
            f"overlap_source={overlap_source}"
        )

        all_mvc_pairs.extend(mvc)
        all_temporal_pairs.extend(temporal)
        per_scene_meta.append({
            "scene_id": scene_id,
            "batch_id": args.batch,
            "camera_names": camera_names,
            "overlap_pairs": [list(p) for p in sorted(overlap)],
            "overlap_source": overlap_source,
            "raw_mvc_pair_count": raw_count,
            "distance_qualified_mvc_pair_count": dist_count,
            "protocol_stats": protocol_stats,
            "gt_identity_guard": leak_report,
            "frame_wh_by_camera": {cam: list(wh) for cam, wh in frame_wh_by_camera.items()},
        })

    if not per_scene_meta:
        raise SystemExit('scene')

    print(f"\n───  ───")
                                         
    model = ResidualMLPModel.create(cfg)
    raw_total = len(all_mvc_pairs)
    edge_policy: dict[str, Any] = {}
    edge_json = str(getattr(args, "edge_weight_json", "") or "")
    if edge_json:
        policy_path = resolve_project_path(edge_json)
        loaded = json.loads(Path(policy_path).read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            if scene_ids and scene_ids[0] in loaded and isinstance(loaded[scene_ids[0]], dict):
                nested = loaded[scene_ids[0]]
                if any(":" in str(k) for k in nested):
                    loaded = nested
            edge_policy = loaded
        cap_ratio = float(getattr(args, "edge_weight_cap_ratio", 0.0) or 0.0)
        before_policy = len(all_mvc_pairs)
        all_mvc_pairs = apply_typed_edge_policy(
            all_mvc_pairs, edge_policy, cap_ratio=cap_ratio
        )
        print(
            f"  [g2] typed edge policy {policy_path.name}: "
            f"{before_policy}->{len(all_mvc_pairs)} pairs "
            f"cap_ratio={cap_ratio:g}"
        )
    all_mvc_pairs = model.select_mvc_pairs(all_mvc_pairs, max_pairs=args.max_pairs)
    confidence_count = len(all_mvc_pairs)
    if confidence_count < args.min_mvc_pairs:
        raise SystemExit(
            f"MVC {confidence_count} < --min-mvc-pairs {args.min_mvc_pairs}"
            f"raw_total={raw_total}"
            f"teacher_conf>={cfg.min_teacher_confidence:.3f}"
            f"|κA-κB|>={cfg.min_teacher_confidence_gap:.3f}"
            'near/far'
            '/'
        )
    print(
        f"scenes={len(per_scene_meta)}  "
        f"mvc_pairs={confidence_count}/{raw_total} confidence/raw  "
        f"temporal={len(all_temporal_pairs)}  p_geo_source=DB world_x/y"
    )

    preflight_report: dict[str, Any] | None = None
    if not args.skip_preflight_diagnostic:
        preflight_report = diagnose_low_order_bias(
            all_mvc_pairs,
            pair_source="selected_mvc_pairs_after_distance_and_teacher_filters",
            max_residual_m=cfg.max_residual_m,
        )
        for line in format_low_order_preflight_report(
            preflight_report,
            corrections_applied=bool(camera_translation_corrections),
        ):
            print(line)

    model.train_offline(all_mvc_pairs, all_temporal_pairs)
    meta: dict[str, Any] = {
        "scenes": per_scene_meta,
        "scene_id": args.scene,                
        "batch_id": args.batch,
        "seed": args.seed,
        "risk_scalar_mode": cfg.risk_scalar_mode,
        "s0_m": cfg.s0_m,
        "feature_l0_m": cfg.feature_l0_m,
        "anchor_shape_eta": cfg.anchor_shape_eta,
        "anchor_aspect_ratio_min": cfg.anchor_aspect_ratio_min,
        "anchor_aspect_ratio_max": cfg.anchor_aspect_ratio_max,
        "pixel_grazing_d0": cfg.pixel_grazing_d0,
        "sensitivity_clip_m": cfg.sensitivity_clip_m,
        "extent_clip_m": cfg.extent_clip_m,
        "motion_compensation": cfg.motion_compensation,
        "time_offsets_sec": cfg.time_offsets_sec,
        "anchor_mode": cfg.anchor_mode,
        "max_residual_m": args.max_residual_m,
        "lambda_mean_delta": cfg.lambda_mean_delta,
        "lambda_pair_center": cfg.lambda_pair_center,
        "loss_mode": cfg.loss_mode,
        "mvc_pair_count": confidence_count,
        "raw_mvc_pair_count": raw_total,
        "max_pairs": args.max_pairs,
        "split_manifest": split_filter.path if split_filter else "",
        "train_splits": list(split_filter.split_names) if split_filter else [],
        "time_start": args.time_start,
        "time_end": args.time_end,
        "allow_gt_identity_mining": bool(args.allow_gt_identity_mining),
        "temporal_pair_count": len(all_temporal_pairs),
        "min_teacher_confidence_gap": cfg.min_teacher_confidence_gap,
        "max_pair_dist_m": cfg.max_pair_dist_m,
        TRAINING_CAMERA_TRANSLATION_META_KEY: serialize_camera_translation_corrections(
            camera_translation_corrections
        ),
        "frame_wh": frame_wh,
        "clip_margin_px": args.clip_margin,
        "clip_margin_ratio": args.clip_margin_ratio,
        "train_config": cfg.__dict__,
    }
    if preflight_report is not None:
        meta["label_free_preflight"] = preflight_report
    save_checkpoint(resolve_project_path(args.out), model.mlp, meta)
    finish_batch_stage(
        conn,
        stage_run_id,
        "completed",
        {
            "mvc_pair_count": confidence_count,
            "raw_mvc_pair_count": raw_total,
            "temporal_pair_count": len(all_temporal_pairs),
            "checkpoint": str(resolve_project_path(args.out)),
        },
    )
    conn.close()
    print(f"saved -> {args.out}  feature_version={FEATURE_VERSION}")


def cmd_infer(args: argparse.Namespace) -> None:
    runtime_corrections = load_camera_translation_corrections(
        args.camera_translation_json,
        acceptance=getattr(args, "camera_translation_acceptance", "recommended"),
    )
    controller = MVCResidualController.from_checkpoint(
        resolve_project_path(args.checkpoint),
        camera_translation_corrections=runtime_corrections,
    )
    controller.configure_residual_invoke(
        acceptor_json=getattr(args, "acceptor_json", None),
        force_all=bool(getattr(args, "force_all_residual", False)),
    )
    if runtime_corrections:
        print(
            "[g2 infer] runtime camera translations loaded: "
            f"{runtime_corrections.camera_count()} camera(s), "
            f"{runtime_corrections.scene_count()} scene map(s)"
        )
    elif controller.training_camera_translation_corrections:
        print(
            '[g2 infer][] checkpoint records training-time camera translations,'
            "but runtime T_cam is disabled unless --camera-translation-json is provided."
        )
    conn = sqlite3.connect(str(resolve_project_path(args.db)))
    cal = load_calibration(conn, args.camera, args.scene)
    if cal is None:
        raise SystemExit(f": {args.camera}")
    H, scale = cal
    xyxy = np.array([800.0, 400.0, 1000.0, 550.0])
    wh = (1920, 1080)
    pred = controller.predict_residual(
        xyxy,
        H,
        scale,
        wh,
        camera_name=args.camera,
        scene_id=args.scene,
    )
    print(
        json.dumps(
            {
                "camera": args.camera,
                "world_geo_m": pred.world_geo,
                "camera_translation_m": pred.camera_translation_m,
                "world_corrected_geo_m": pred.world_corrected_geo,
                "delta_m": list(pred.delta_m),
                "world_final_m": pred.world_final,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Geometry-guided residual MLP. Published fitting is "
            "scripts/train_recording.py. This train subcommand cannot replace "
            "it: it is a database trainer and still requires --split-manifest."
        )
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser(
        "train",
        help=(
            "Database trainer. Not a substitute for scripts/train_recording.py. "
            "Requires --split-manifest. Numeric defaults match the published loss."
        ),
    )
    p_train.add_argument("--db", default="cals.db")
    p_train.add_argument(
        "--batch",
        required=True,
        help='ID trajectory_observations',
    )
    p_train.add_argument(
        "--scene",
        default="parking",
        help='ID scene xizi_01,xizi_02,xizi_03'
             'scene',
    )
    p_train.add_argument("--cameras", default="", help='scene')
    p_train.add_argument("--out", default="nn_models/checkpoints/residual_mlp.pt")
    p_train.add_argument(
        "--camera-translation-json",
        default=None,
        help=(
            "Optional accepted fixed-translation projection JSON. Expected forms include "
            "{'corrections_m_by_scene': {'scene': {'cam': {'dx': ..., 'dy': ...}}}}; "
            "normally produced by pipeline/g2_low_order_correction.py fit, applied before "
            "MVC filtering/training, and stored as training metadata. It is not auto-applied "
            "at runtime."
        ),
    )
    p_train.add_argument(
        "--camera-translation-acceptance",
        dest="camera_translation_acceptance",
        choices=CAMERA_TRANSLATION_ACCEPTANCE_CHOICES,
        default="recommended",
        help=(
            "Acceptance level for per-scene fixed-translation projections from "
            "--camera-translation-json (default: 'recommended'). "
            "'recommended' keeps only scenes with decision.status=recommended; "
            "'caution' keeps recommended + caution; "
            "'any' keeps everything. "
            "Scenes with missing/unknown status are skipped; use 'any' explicitly for legacy JSON."
        ),
    )
    p_train.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for numpy/torch; stored in checkpoint metadata.",
    )
    p_train.add_argument(
        "--split-manifest",
        default=None,
        help="FAB2 gt_index.csv or dataset_split_manifest.csv; limits training to --train-splits.",
    )
    p_train.add_argument(
        "--allow-unsplit-training",
        action="store_true",
        help="Legacy diagnostic only: permit training without --split-manifest.",
    )
    p_train.add_argument(
        "--train-splits",
        default="train",
        help="Comma-separated split names to use from --split-manifest (default: train).",
    )
    p_train.add_argument(
        "--time-start",
        type=float,
        default=None,
        help="Optional inclusive lower timestamp bound for training observations.",
    )
    p_train.add_argument(
        "--time-end",
        type=float,
        default=None,
        help="Optional exclusive upper timestamp bound for training observations.",
    )
    p_train.add_argument(
        "--allow-gt-identity-mining",
        action="store_true",
        help="Bypass FAB2 guard that rejects MVC mining on GT global_id values; use only for smoke tests.",
    )
    p_train.add_argument("--epochs", type=int, default=80)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument(
        "--lambda-reg",
        type=float,
        default=0.05,
        help="Published magnitude weight, 0.10.",
    )
    p_train.add_argument(
        "--lambda-smooth",
        type=float,
        default=0.02,
        help="Temporal weight. The published value is 0.02.",
    )
    p_train.add_argument(
        "--lambda-mean-delta",
        dest="lambda_mean_delta",
        type=float,
        default=1.0,
        help="Published mean-residual weight, 1. The term is ||mean residual||^2.",
    )
    p_train.add_argument(
        "--lambda-pair-center",
        dest="lambda_pair_center",
        type=float,
        default=0.50,
        help=(
            "Published pair-center weight, 0.50. The term weights "
            "||(residual_a + residual_b) / 2||^2 by the lower sensitivity."
        ),
    )
    p_train.add_argument(
        "--max-residual-m",
        type=float,
        default=1.0,
        help="Published per-coordinate residual bound, 1 m.",
    )
    p_train.add_argument("--max-time-gap", type=float, default=0.35)
    p_train.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional diagnostic cap on retained MVC pairs after confidence/gap filtering.",
    )
    p_train.add_argument(
        "--risk-scalar",
        dest="risk_scalar",
        choices=RISK_SCALAR_MODES,
        default="metric_jacobian",
        help=(
            'fab1 metric_jacobian s=||J_H||_F'
            'pixel_orthogonal/pixel_vertical grazing_angle'
        ),
    )
    p_train.add_argument(
        "--anchor-mode",
        dest="anchor_mode",
        choices=("multiplicative_gating_offsetted", "bottom_center", "center"),
        default="multiplicative_gating_offsetted",
        help=(
            "Published HSG anchor is multiplicative_gating_offsetted. "
            "center and bottom_center are fixed-anchor controls, not a second method."
        ),
    )
    p_train.add_argument(
        "--s0-m",
        dest="s0_m",
        type=float,
        default=60.0,
        metavar="METERS",
        help='s0α=s/(s+s0) --risk-scalar=metric_jacobian 60',
    )
    p_train.add_argument(
        "--feature-l0-m",
        dest="feature_l0_m",
        type=float,
        default=10.0,
        metavar="METERS",
        help='L0F[2:4]=w_m/L0,h_m/L0 10',
    )
    p_train.add_argument(
        "--anchor-shape-eta",
        dest="anchor_shape_eta",
        type=float,
        default=DEFAULT_ANCHOR_SHAPE_ETA,
        metavar="ETA",
        help=(
            'η'
            'beta=alpha+η*gate*(shape-alpha) 0.75'
        ),
    )
    p_train.add_argument(
        "--anchor-aspect-ratio-min",
        dest="anchor_aspect_ratio_min",
        type=float,
        default=DEFAULT_ANCHOR_ASPECT_RATIO_MIN,
        metavar="RHO_MIN",
        help='rho=h/w 0.5',
    )
    p_train.add_argument(
        "--anchor-aspect-ratio-max",
        dest="anchor_aspect_ratio_max",
        type=float,
        default=DEFAULT_ANCHOR_ASPECT_RATIO_MAX,
        metavar="RHO_MAX",
        help='rho=h/w 3.0',
    )
    p_train.add_argument(
        "--pixel-grazing-d0",
        dest="pixel_grazing_d0",
        type=float,
        default=50.0 / 1080.0,
        metavar="FRAC",
        help=(
            'd0_px = FRAC × frame_h'
            '--risk-scalar=pixel_orthogonal/pixel_vertical 50/1080≈0.046'
        ),
    )
    p_train.add_argument(
        "--metric-sensitivity-clip-m",
        dest="sensitivity_clip_m",
        type=float,
        default=None,
        metavar="METERS",
        help=(
            's α/κ checkpoint meta'
            '//remerge/eval'
        ),
    )
    p_train.add_argument(
        "--extent-clip-m",
        dest="extent_clip_m",
        type=float,
        default=None,
        metavar="METERS",
        help='w_m/h_m α/κ',
    )
    p_train.add_argument(
        "--motion-compensation",
        dest="motion_compensation",
        choices=("off", "nearest", "linear_interpolate"),
        default="linear_interpolate",
        help=(
            'MVC pair linear_interpolatefab1 t*'
            'nearest/offcla1 (ix)'
        ),
    )
    p_train.add_argument(
        "--time-offsets-json",
        dest="time_offsets_json",
        default=None,
        metavar="JSON",
        help='JSON \'{"c003:c004": 0.032}\'camB camA',
    )
    p_train.add_argument(
        "--loss-mode",
        choices=("asymmetric_teacher", "symmetric_weighted"),
        default="asymmetric_teacher",
        help='MVC stop-gradient teachersymmetric_weighted',
    )
    p_train.add_argument(
        "--edge-weight-json",
        default="",
        help=(
            "Optional per-edge teacher policy JSON: "
            '{"C05:C06": {"type": "NEAR_FAR", "loss_mode": "asymmetric_teacher", '
            '"weight": 1.0, "require_kappa_gap": true}}. '
            "Weights stay in loss/sampling; they are not MLP inputs."
        ),
    )
    p_train.add_argument(
        "--edge-weight-cap-ratio",
        type=float,
        default=0.0,
        help="If >0, cap each edge's total sample weight at this times the median edge weight",
    )
    p_train.add_argument(
        "--min-teacher-confidence",
        type=float,
        default=0.20,
        help='ω 0.20',
    )
    p_train.add_argument(
        "--min-teacher-confidence-gap",
        type=float,
        default=0.05,
        help=(
            'MVC pair |κA-κB|'
            'near/far 0.05'
        ),
    )
    p_train.add_argument(
        "--teacher-confidence-power",
        type=float,
        default=1.0,
        help='teacher >1 1.0',
    )
    p_train.add_argument(
        "--student-weight-floor",
        type=float,
        default=0.05,
        help='student 0.05',
    )
    p_train.add_argument(
        "--reg-confidence-floor",
        type=float,
        default=0.25,
        help='0.25',
    )
    p_train.add_argument(
        "--frame-width",
        type=int,
        default=1920,
        help='bbox 1920',
    )
    p_train.add_argument(
        "--frame-height",
        type=int,
        default=1080,
        help='bbox 1080',
    )
    p_train.add_argument(
        "--clip-margin",
        type=int,
        default=3,
        help='DB bbox 3',
    )
    p_train.add_argument(
        "--clip-margin-ratio",
        type=float,
        default=0.01,
        help='DB bbox 0.01 1%%',
    )
    p_train.add_argument(
        "--overlap-pairs",
        default="",
        help='c003:c004,c003:c005 DB scene_overlap_pairs',
    )
    p_train.add_argument(
        "--max-pair-dist-m",
        type=float,
        default=2.0,
        help="Published geometric pair gate, 2 m.",
    )
    p_train.add_argument(
        "--min-mvc-pairs",
        type=int,
        default=32,
        help="Published minimum training support, 32 retained pairs.",
    )
    p_train.add_argument(
        "--skip-preflight-diagnostic",
        action="store_true",
        help='label-free',
    )
    p_train.set_defaults(func=cmd_train)

    p_infer = sub.add_parser("infer", help='See the HSG-SRL reproduction notes.')
    p_infer.add_argument("--checkpoint", required=True)
    p_infer.add_argument("--db", default="cals.db")
    p_infer.add_argument("--scene", default="parking")
    p_infer.add_argument("--camera", default="A")
    p_infer.add_argument(
        "--camera-translation-json",
        default=None,
        help=(
            "Explicit runtime per-camera translation JSON. Checkpoint training metadata is "
            "audit-only and is not applied during inference unless this is provided."
        ),
    )
    p_infer.add_argument(
        "--camera-translation-acceptance",
        dest="camera_translation_acceptance",
        choices=CAMERA_TRANSLATION_ACCEPTANCE_CHOICES,
        default="recommended",
        help=(
            "Filter runtime corrections by per-scene decision.status (default: "
            "'recommended'). 'caution' keeps recommended+caution; 'any' disables "
            "filtering for legacy JSON. Missing/unknown status is skipped."
        ),
    )
    p_infer.add_argument(
        "--acceptor-json",
        default=None,
        help="Val-fitted residual invoke JSON; cameras not listed get dP=0.",
    )
    p_infer.add_argument(
        "--force-all-residual",
        action="store_true",
        help="Apply dP on every camera (negative-transfer diagnostic).",
    )
    p_infer.set_defaults(func=cmd_infer)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
