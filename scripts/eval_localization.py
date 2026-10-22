#!/usr/bin/env python3
"""Score matched-detection localization and the paired track bootstrap.

Writes one JSON per recording with HSG, each residual seed, and the
1000-draw interval for the residual-minus-HSG median. The residual is added
only on cameras in that seed's support set whose stored HSG point lies in
the frozen overlap union. ``--no-spatial-gate`` keeps the camera support
and drops the overlap mask.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from bootstrap import ARTIFACT_ROOT
from db_access import deploy_file, matched_rows, positions
from intervals import paired_bootstrap
from protocol import SEEDS, SITES


def stats(values: np.ndarray) -> dict:
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
    }


def deploy_for(site: str, seed: int) -> dict:
    return json.loads(deploy_file(site, seed).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, choices=sorted(SITES))
    parser.add_argument("--seed", type=int, default=None, help="default: seeds 0, 1, and 2")
    parser.add_argument("--no-spatial-gate", action="store_true")
    args = parser.parse_args()
    seeds = (args.seed,) if args.seed is not None else SEEDS
    rows, by_track = matched_rows(args.site)
    gt = np.stack([row["gt"] for row in rows])
    geo = np.linalg.norm(np.stack([row["xy"] for row in rows]) - gt, axis=1)
    result = {"site": args.site, "geo": stats(geo), "seeds": {}}
    for seed in seeds:
        deploy = deploy_for(args.site, seed)
        refined = np.linalg.norm(
            positions(args.site, seed, rows, deploy, spatial=not args.no_spatial_gate) - gt,
            axis=1,
        )
        result["seeds"][str(seed)] = {
            "S": deploy.get("S") or [],
            "hsg_srl": stats(refined),
            "bootstrap_minus_hsg": paired_bootstrap(by_track, geo, refined),
        }
    dest = ARTIFACT_ROOT / "results" / "runs" / "localization" / f"{args.site}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[localization] -> {dest}")


if __name__ == "__main__":
    main()
