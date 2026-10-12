#!/usr/bin/env python3
"""Label-free diagnostics for mined cross-view pairs.

The filename keeps the internal tag g2. The published pair miner is
scripts/freeze_pairs.py. This diagnostic reads projected pair coordinates
and does not use ground truth as a training target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LowOrderPreflightThresholds:
    min_pairs: int = 32
    warn_bias_m: float = 0.50
    warn_translation_explained_frac: float = 0.50
    caution_bias_m: float = 0.30
    caution_translation_explained_frac: float = 0.35


def _camera_name(obs: Any) -> str:
    return str(getattr(obs, "camera_name", ""))


def _scene_id(obs: Any) -> str:
    return str(getattr(obs, "scene_id", "") or "")


def _world_geo(obs: Any) -> np.ndarray:
    return np.asarray(getattr(obs, "world_geo"), dtype=np.float64).reshape(2)


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def _norm_stats(vectors: np.ndarray) -> dict[str, float]:
    if vectors.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "mean": float(np.mean(norms)),
        "p50": _percentile(norms, 50),
        "p90": _percentile(norms, 90),
        "max": float(np.max(norms)),
    }


def _explained_fraction(total_rss: float, residual_rss: float) -> float:
    if total_rss <= 1e-12:
        return 0.0
    return float(np.clip(1.0 - residual_rss / total_rss, 0.0, 1.0))


def _affine_residual(
    diffs: np.ndarray,
    centers: np.ndarray,
) -> tuple[float, float | None]:
    if len(diffs) < 6:
        return 0.0, None
    x_mat = np.column_stack(
        [
            np.ones(len(centers), dtype=np.float64),
            centers[:, 0],
            centers[:, 1],
        ]
    )
    if np.linalg.matrix_rank(x_mat) < 3:
        return 0.0, None
    coef, *_ = np.linalg.lstsq(x_mat, diffs, rcond=None)
    residual = diffs - x_mat @ coef
    residual_rss = float(np.sum(residual * residual))
    return residual_rss, float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def _summarize_vectors(
    diffs: np.ndarray,
    centers: np.ndarray,
    *,
    thresholds: LowOrderPreflightThresholds,
) -> dict[str, Any]:
    diffs = np.asarray(diffs, dtype=np.float64).reshape(-1, 2)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    if len(diffs) == 0:
        return {
            "n": 0,
            "status": "insufficient",
            "mean_xy_m": [0.0, 0.0],
            "median_xy_m": [0.0, 0.0],
            "mean_bias_norm_m": 0.0,
            "norm_m": _norm_stats(diffs),
            "translation_residual_rms_m": 0.0,
            "translation_explained_frac": 0.0,
            "affine_residual_rms_m": None,
            "affine_explained_frac": None,
        }

    mean_xy = np.mean(diffs, axis=0)
    centered = diffs - mean_xy
    total_rss = float(np.sum(diffs * diffs))
    translation_rss = float(np.sum(centered * centered))
    affine_rss, affine_rms = _affine_residual(diffs, centers)
    affine_explained = (
        None if affine_rms is None else _explained_fraction(total_rss, affine_rss)
    )
    mean_bias_norm = float(np.linalg.norm(mean_xy))
    translation_explained = _explained_fraction(total_rss, translation_rss)

    if len(diffs) < thresholds.min_pairs:
        status = "insufficient"
    elif (
        mean_bias_norm >= thresholds.warn_bias_m
        and translation_explained >= thresholds.warn_translation_explained_frac
    ):
        status = "warning"
    elif (
        mean_bias_norm >= thresholds.caution_bias_m
        and translation_explained >= thresholds.caution_translation_explained_frac
    ):
        status = "caution"
    else:
        status = "ok"

    return {
        "n": int(len(diffs)),
        "status": status,
        "mean_xy_m": [float(mean_xy[0]), float(mean_xy[1])],
        "median_xy_m": [float(v) for v in np.median(diffs, axis=0)],
        "mean_bias_norm_m": mean_bias_norm,
        "norm_m": _norm_stats(diffs),
        "translation_residual_rms_m": float(
            np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
        ),
        "translation_explained_frac": translation_explained,
        "affine_residual_rms_m": affine_rms,
        "affine_explained_frac": affine_explained,
    }


def diagnose_low_order_bias(
    pairs: list[Any],
    *,
    pair_source: str,
    max_residual_m: float,
    thresholds: LowOrderPreflightThresholds | None = None,
) -> dict[str, Any]:
    """Summarize fixed-translation projection candidates in MVC P_geo pair disagreements."""
    thresholds = thresholds or LowOrderPreflightThresholds()
    diffs: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    grouped: dict[tuple[str, str], tuple[list[np.ndarray], list[np.ndarray]]] = {}
    scene_grouped: dict[tuple[str, str, str], tuple[list[np.ndarray], list[np.ndarray]]] = {}

    for pair in pairs:
        obs_a = getattr(pair, "obs_a")
        obs_b = getattr(pair, "obs_b")
        geo_a = _world_geo(obs_a)
        geo_b = _world_geo(obs_b)
        diff = geo_a - geo_b
        center = 0.5 * (geo_a + geo_b)
        scene_id = _scene_id(obs_a) or _scene_id(obs_b)
        diffs.append(diff)
        centers.append(center)
        key = (_camera_name(obs_a), _camera_name(obs_b))
        group_diffs, group_centers = grouped.setdefault(key, ([], []))
        group_diffs.append(diff)
        group_centers.append(center)
        scene_key = (scene_id, _camera_name(obs_a), _camera_name(obs_b))
        scene_diffs, scene_centers = scene_grouped.setdefault(scene_key, ([], []))
        scene_diffs.append(diff)
        scene_centers.append(center)

    all_diffs = np.asarray(diffs, dtype=np.float64).reshape(-1, 2) if diffs else np.zeros((0, 2))
    all_centers = (
        np.asarray(centers, dtype=np.float64).reshape(-1, 2) if centers else np.zeros((0, 2))
    )
    global_summary = _summarize_vectors(
        all_diffs,
        all_centers,
        thresholds=thresholds,
    )

    pair_summaries: list[dict[str, Any]] = []
    for (cam_a, cam_b), (group_diffs, group_centers) in sorted(grouped.items()):
        summary = _summarize_vectors(
            np.asarray(group_diffs, dtype=np.float64),
            np.asarray(group_centers, dtype=np.float64),
            thresholds=thresholds,
        )
        mean_xy = np.asarray(summary["mean_xy_m"], dtype=np.float64)
        summary.update(
            {
                "camera_a": cam_a,
                "camera_b": cam_b,
                "balanced_translation_hint_m": {
                    cam_a: [float(-0.5 * mean_xy[0]), float(-0.5 * mean_xy[1])],
                    cam_b: [float(+0.5 * mean_xy[0]), float(+0.5 * mean_xy[1])],
                },
            }
        )
        pair_summaries.append(summary)

    scene_pair_summaries: list[dict[str, Any]] = []
    for (scene_id, cam_a, cam_b), (group_diffs, group_centers) in sorted(scene_grouped.items()):
        summary = _summarize_vectors(
            np.asarray(group_diffs, dtype=np.float64),
            np.asarray(group_centers, dtype=np.float64),
            thresholds=thresholds,
        )
        mean_xy = np.asarray(summary["mean_xy_m"], dtype=np.float64)
        summary.update(
            {
                "scene_id": scene_id,
                "camera_a": cam_a,
                "camera_b": cam_b,
                "balanced_translation_hint_m": {
                    cam_a: [float(-0.5 * mean_xy[0]), float(-0.5 * mean_xy[1])],
                    cam_b: [float(+0.5 * mean_xy[0]), float(+0.5 * mean_xy[1])],
                },
            }
        )
        scene_pair_summaries.append(summary)

    status_rank = {"warning": 3, "caution": 2, "ok": 1, "insufficient": 0}
    status = global_summary["status"]
    for summary in pair_summaries + scene_pair_summaries:
        if status_rank[summary["status"]] > status_rank[status]:
            status = summary["status"]

    return {
        "version": "g2_label_free_preflight_v1",
        "pair_source": pair_source,
        "max_residual_m": float(max_residual_m),
        "thresholds": {
            "min_pairs": thresholds.min_pairs,
            "warn_bias_m": thresholds.warn_bias_m,
            "warn_translation_explained_frac": thresholds.warn_translation_explained_frac,
            "caution_bias_m": thresholds.caution_bias_m,
            "caution_translation_explained_frac": thresholds.caution_translation_explained_frac,
        },
        "status": status,
        "global": global_summary,
        "camera_pairs": pair_summaries,
        "scene_camera_pairs": scene_pair_summaries,
    }


def _fmt_xy(values: Any) -> str:
    xy = np.asarray(values, dtype=np.float64).reshape(2)
    return f"[{xy[0]:+.3f}, {xy[1]:+.3f}]"


def _fmt_frac(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _summary_line(prefix: str, summary: dict[str, Any]) -> str:
    norm = summary["norm_m"]
    affine_rms = summary.get("affine_residual_rms_m")
    affine_rms_text = "n/a" if affine_rms is None else f"{float(affine_rms):.3f}m"
    return (
        f"{prefix} n={summary['n']} mean={_fmt_xy(summary['mean_xy_m'])}m "
        f"|mean|={summary['mean_bias_norm_m']:.3f}m; "
        f"|diff| mean/p50/p90={norm['mean']:.3f}/{norm['p50']:.3f}/{norm['p90']:.3f}m; "
        f"translation_explained={_fmt_frac(summary['translation_explained_frac'])} "
        f"residual_rms={summary['translation_residual_rms_m']:.3f}m; "
        f"affine_explained={_fmt_frac(summary.get('affine_explained_frac'))} "
        f"affine_rms={affine_rms_text}"
    )


def format_low_order_preflight_report(
    report: dict[str, Any],
    *,
    corrections_applied: bool,
    top_k_pairs: int = 3,
) -> list[str]:
    """Format a concise train-time preflight report."""
    lines = [
        "  [g2 preflight] label-free low-order check "
        f"source={report['pair_source']} status={report['status']}"
    ]
    lines.append(_summary_line("  [g2 preflight] all", report["global"]))

    pair_summaries = sorted(
        report["camera_pairs"],
        key=lambda item: float(item.get("mean_bias_norm_m", 0.0)),
        reverse=True,
    )
    for item in pair_summaries[: max(0, top_k_pairs)]:
        prefix = f"  [g2 preflight] {item['camera_a']}:{item['camera_b']}"
        lines.append(_summary_line(prefix, item))

    scene_pair_summaries = sorted(
        report.get("scene_camera_pairs", []),
        key=lambda item: float(item.get("mean_bias_norm_m", 0.0)),
        reverse=True,
    )
    for item in scene_pair_summaries[: max(0, top_k_pairs)]:
        if not item.get("scene_id"):
            continue
        prefix = f"  [g2 preflight] {item['scene_id']} {item['camera_a']}:{item['camera_b']}"
        lines.append(_summary_line(prefix, item))

    flagged = [
        p
        for p in (scene_pair_summaries or pair_summaries)
        if p.get("status") in {"warning", "caution"}
    ]
    if flagged and not corrections_applied:
        worst = flagged[0]
        lines.append(
            '[g2 preflight][] MVC pairs show a fixed-translation projection candidate'
            "before "
            f"MLP training ({worst['camera_a']}:{worst['camera_b']} |mean|="
            f"{worst['mean_bias_norm_m']:.3f}m). Fit the projection artifact with "
            "pipeline/g2_low_order_correction.py and pass it with "
            "--camera-translation-json only under the acceptance gate."
        )
    elif flagged and corrections_applied:
        worst = flagged[0]
        lines.append(
            '[g2 preflight][] fixed-translation projections are already applied, but'
            f"{worst['camera_a']}:{worst['camera_b']} still has |mean|="
            f"{worst['mean_bias_norm_m']:.3f}m. The remaining field may need stricter "
            "pair filtering or a higher-order geometry correction."
        )
    else:
        lines.append(
            "  [g2 preflight] no deployable fixed-translation projection detected in the "
            "selected MVC supervision set."
        )
    return lines
