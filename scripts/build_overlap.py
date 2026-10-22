#!/usr/bin/env python3
"""Write a working overlap graph for a from-video installation.

LUMPI and Synthehicle keep an edge at 1000 m^2 or more. Ko-PER keeps an edge
at 500 m^2 or more, which retains its 579.35 m^2 SK1-SK4 intersection.
Published pair mining and tracking read configs/overlap and do not apply this
threshold again. This file is not a spanning tree. The geometry-database
remerge does not use it; pseudo-identities use configs/pseudo_identity_overlap.
"""

from __future__ import annotations

import json
import sys

from bootstrap import ARTIFACT_ROOT
from db_access import connect_ro
from protocol import AREA_MIN_M2, DET_BATCH, SITES, geometry_db, overlap_path


def main() -> None:
    sys.path.insert(0, str(ARTIFACT_ROOT / "src"))
    from geometry.fov_overlap import candidate_edges

    sites = sys.argv[1:] or list(SITES)
    for site in sites:
        cameras = list(SITES[site]["cameras"])
        minimum = AREA_MIN_M2[SITES[site]["kind"]]
        conn = connect_ro(geometry_db(site))
        edges = candidate_edges(conn, site, cameras, batch=DET_BATCH, a_min=minimum)
        conn.close()
        payload = {
            "site": site,
            "cameras": cameras,
            "edges": [
                {"a": a, "b": b, "edge": f"{a}:{b}", "inter_area_m2": area}
                for a, b, area in edges
            ],
            "rule": (
                f"working graph: projected FOV hull intersection at least {minimum:.0f} m^2; "
                "published scoring uses configs/overlap without a second area filter"
            ),
        }
        dest = overlap_path(site)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[overlap] {site} edges={len(edges)}", flush=True)


if __name__ == "__main__":
    main()
