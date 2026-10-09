# The HOTA modules under third_party/TrackEval are TrackEval commit
# 12c8791b303e0a0b50f753af204249e622d0281a (Luiten et al., IJCV 2021).
"""BEV-HOTA through the vendored TrackEval HOTA class.

Similarity is max(0, 1 - d / 2.5 m). One Hungarian match is shared across
thresholds. The four reported gates are the TrackEval alphas 0.80, 0.60,
0.40 and 0.20. This is not image-box IoU HOTA.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "float"):
    np.float = float  # NumPy 2 removed the alias used by this TrackEval commit.

_VENDOR = Path(__file__).resolve().parents[2] / "third_party" / "TrackEval"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from trackeval.metrics.hota import HOTA  # noqa: E402

D_MAX_M = 2.5
GATES = (0.5, 1.0, 1.5, 2.0)
ALPHAS = (0.80, 0.60, 0.40, 0.20)
_METRIC = HOTA()


def _similarity(g_xy, p_xy):
    if len(g_xy) == 0 or len(p_xy) == 0:
        return np.zeros((len(g_xy), len(p_xy)))
    d = np.hypot(g_xy[:, None, 0] - p_xy[None, :, 0], g_xy[:, None, 1] - p_xy[None, :, 1])
    return np.maximum(0.0, 1.0 - d / D_MAX_M)


def _pack(gt_bins, pred_bins):
    gt_ids = sorted({i for fr in gt_bins.values() for i in fr})
    pr_ids = sorted({i for fr in pred_bins.values() for i in fr})
    gt_index = {i: n for n, i in enumerate(gt_ids)}
    pr_index = {i: n for n, i in enumerate(pr_ids)}
    times = sorted(set(gt_bins) | set(pred_bins))
    gt_list, pr_list, sims = [], [], []
    n_gt = n_pr = 0
    for t in times:
        g = gt_bins.get(t, {})
        p = pred_bins.get(t, {})
        g_idx = np.array([gt_index[i] for i in g], dtype=np.int64)
        p_idx = np.array([pr_index[i] for i in p], dtype=np.int64)
        g_xy = np.array([g[i] for i in g], dtype=np.float64).reshape(-1, 2)
        p_xy = np.array([p[i] for i in p], dtype=np.float64).reshape(-1, 2)
        gt_list.append(g_idx)
        pr_list.append(p_idx)
        sims.append(_similarity(g_xy, p_xy))
        n_gt += len(g_idx)
        n_pr += len(p_idx)
    data = {
        "num_timesteps": len(times),
        "num_gt_ids": len(gt_ids),
        "num_tracker_ids": len(pr_ids),
        "num_gt_dets": n_gt,
        "num_tracker_dets": n_pr,
        "gt_ids": gt_list,
        "tracker_ids": pr_list,
        "similarity_scores": sims,
    }
    return data


def hota_trackeval(gt_bins, pred_bins) -> list[dict]:
    data = _pack(gt_bins, pred_bins)
    res = _METRIC.eval_sequence(data)
    labels = _METRIC.array_labels
    out = []
    for gate, alpha in zip(GATES, ALPHAS):
        a = int(np.argmin(np.abs(labels - alpha)))
        if abs(float(labels[a]) - alpha) > 1e-8:
            raise RuntimeError(f"alpha {alpha} is not on the TrackEval grid")
        tp = int(res["HOTA_TP"][a])
        fp = int(res["HOTA_FP"][a])
        fn = int(res["HOTA_FN"][a])
        out.append({
            "gate": gate,
            "alpha": float(labels[a]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "deta": float(res["DetA"][a]),
            "assa": float(res["AssA"][a]),
            "hota": float(res["HOTA"][a]),
            "loca": float(res["LocA"][a]),
        })
    return out


def _self_check() -> None:
    eps = np.finfo(float).eps
    perfect_gt = {0: {1: (0.0, 0.0)}, 1: {1: (1.0, 0.0)}}
    perfect_pr = {0: {7: (0.0, 0.0)}, 1: {7: (1.0, 0.0)}}
    perfect = hota_trackeval(perfect_gt, perfect_pr)
    for row in perfect:
        assert row["tp"] == 2 and row["fp"] == 0 and row["fn"] == 0
        assert abs(row["hota"] - 1.0) < 1e-9
    empty = hota_trackeval({0: {1: (0.0, 0.0)}}, {})
    for row in empty:
        assert row["tp"] == 0 and row["fn"] == 1 and row["fp"] == 0 and row["hota"] == 0.0
    extra = hota_trackeval({0: {1: (0.0, 0.0)}}, {0: {7: (0.0, 0.0), 8: (10.0, 0.0)}})
    assert extra[0]["tp"] == 1 and extra[0]["fp"] == 1 and extra[0]["fn"] == 0
    switch_gt = {0: {1: (0.0, 0.0)}, 1: {1: (0.0, 0.0)}}
    switch_pr = {0: {7: (0.0, 0.0)}, 1: {8: (0.0, 0.0)}}
    switched = hota_trackeval(switch_gt, switch_pr)
    assert switched[0]["tp"] == 2 and switched[0]["assa"] < 1.0 - 1e-9
    on_gate = {}
    off_gate = {}
    for gate, alpha in zip(GATES, ALPHAS):
        d_on = gate
        d_off = gate + 1e-6
        s_on = max(0.0, 1.0 - d_on / D_MAX_M)
        s_off = max(0.0, 1.0 - d_off / D_MAX_M)
        assert s_on + eps >= alpha
        assert s_off < alpha - eps
        on_gate[0] = {1: (0.0, 0.0)}
        kept = hota_trackeval(on_gate, {0: {7: (d_on, 0.0)}})
        dropped = hota_trackeval(on_gate, {0: {7: (d_off, 0.0)}})
        hit = [r for r in kept if r["gate"] == gate][0]
        miss = [r for r in dropped if r["gate"] == gate][0]
        assert hit["tp"] == 1, (gate, hit)
        assert miss["tp"] == 0, (gate, miss)
        off_gate.clear()


if __name__ == "__main__":
    _self_check()
    print("hota_trackeval self-check ok")
