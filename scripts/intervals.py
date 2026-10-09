"""Paired bootstrap clustered by camera and local image track.

One thousand resamples, seed 20260921. The interval is the 2.5 and 97.5
percentiles of the paired difference. It is not a confidence interval for a
new intersection.
"""

from __future__ import annotations

import numpy as np

from protocol import BOOT_DRAWS, BOOT_SEED


def _stat(values: np.ndarray, name: str) -> float:
    if name == "median":
        return float(np.median(values))
    if name == "p90":
        return float(np.percentile(values, 90))
    return float(np.sqrt(np.mean(np.square(values))))


def paired_bootstrap(
    by_track: dict,
    err_geo: np.ndarray,
    err_refined: np.ndarray,
    seed: int = BOOT_SEED,
    rng: np.random.Generator | None = None,
) -> dict:
    tracks = []
    for indexes in by_track.values():
        block = np.column_stack([err_geo[indexes], err_refined[indexes]])
        ok = np.isfinite(block).all(axis=1)
        if ok.any():
            tracks.append(block[ok])
    if not tracks:
        return {}
    full = np.vstack(tracks)
    rng = rng or np.random.default_rng(seed)
    count = len(tracks)
    out = {}
    for name in ("median", "p90", "rmse"):
        point = _stat(full[:, 1], name) - _stat(full[:, 0], name)
        draws = np.empty(BOOT_DRAWS)
        for draw in range(BOOT_DRAWS):
            take = rng.integers(0, count, size=count)
            sample = np.vstack([tracks[index] for index in take])
            draws[draw] = _stat(sample[:, 1], name) - _stat(sample[:, 0], name)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out[name] = {"d": float(point), "lo": float(lo), "hi": float(hi)}
    return out
