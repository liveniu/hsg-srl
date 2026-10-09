#!/usr/bin/env python3
"""Relax mmdet3d's mmcv cap so 1.7.1 can import. Do not import mmdet3d first."""

from __future__ import annotations

import re
import site
import sys
from pathlib import Path


def _candidates() -> list[Path]:
    roots = []
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    roots.extend(sys.path)
    out = []
    seen = set()
    for root in roots:
        if not root:
            continue
        p = Path(root) / "mmdet3d" / "__init__.py"
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            out.append(p)
    return out


def main() -> None:
    paths = _candidates()
    if not paths:
        raise SystemExit("mmdet3d/__init__.py not found in site-packages")
    n_all = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        new, n = re.subn(
            r"mmcv_maximum_version\s*=\s*['\"][\d.]+['\"]",
            "mmcv_maximum_version = '1.8.0'",
            text,
            count=1,
        )
        if n == 0:
            print(f"no mmcv_maximum_version in {path}")
            continue
        path.write_text(new, encoding="utf-8")
        n_all += n
        print(f"relaxed mmcv_maximum_version -> 1.8.0 in {path}")
    if n_all == 0:
        raise SystemExit("found mmdet3d but could not patch mmcv_maximum_version")
    import mmdet3d  # noqa: F401

    print("import mmdet3d: ok")


if __name__ == "__main__":
    main()
