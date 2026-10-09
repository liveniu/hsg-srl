#!/usr/bin/env python3
"""Fail fast if this machine cannot run the 4090 BEVHeight stack."""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BEV_ROOT = Path(os.environ.get("BEVHEIGHT_ROOT", HERE / "vendor" / "BEVHeight-main"))
sys.path.insert(0, str(BEV_ROOT))
sys.path.insert(0, str(HERE / "tools"))

from hw_guard import apply_4090_runtime  # noqa: E402


def main() -> None:
    apply_4090_runtime(strict=True)
    import mmcv
    import mmdet
    import mmdet3d
    import pytorch_lightning as pl

    print("mmcv", mmcv.__version__)
    print("mmdet", mmdet.__version__)
    print("mmdet3d", mmdet3d.__version__)
    print("lightning", pl.__version__)
    import mmcv._ext  # noqa: F401

    print("mmcv._ext ok")
    if not (BEV_ROOT / "ops" / "voxel_pooling").is_dir():
        raise SystemExit(f"missing vendor ops: {BEV_ROOT}")
    from ops.voxel_pooling import voxel_pooling  # noqa: F401

    print("voxel_pooling ok")
    print("check_env: PASS")


if __name__ == "__main__":
    main()
