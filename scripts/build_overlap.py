#!/usr/bin/env python3
"""Write the operational overlap graph used for pair mining and final tracking.

An edge is kept when the projected ground fields of view intersect by at least
1000 square meters. This is the full retained graph, not a spanning tree.
The geometry-database remerge does not use this file; it keeps the overlap
stored by the importer, which is what formed the pseudo-identities.
"""

from __future__ import annotations

import json
import sys

from bootstrap import ARTIFACT_ROOT
from db_access import connect_ro
from protocol import DET_BATCH, SITES, geometry_db, overlap_path


def main() -> None:
    sys.path.insert(0, str(ARTIFACT_ROOT / "src"))
    from geometry.fov_overlap import candidate_edges

    sites = sys.argv[1:] or list(SITES)
    for site in sites:
        cameras = list(SITES[site]["cameras"])
        conn = connect_ro(geometry_db(site))
        edges = candidate_edges(conn, site, cameras, batch=DET_BATCH)
        conn.close()
        payload = {
            "site": site,
            "cameras": cameras,
            "edges": [
                {"a": a, "b": b, "edge": f"{a}:{b}", "inter_area_m2": area}
                for a, b, area in edges
            ],
            "rule": "projected FOV hull intersection at least 1000 m^2; not an MST",
        }
        dest = overlap_path(site)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[overlap] {site} edges={len(edges)}", flush=True)


if __name__ == "__main__":
    main()
