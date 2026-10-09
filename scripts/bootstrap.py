"""Resolve the artifact root and put src/ on the import path."""

from __future__ import annotations

import sys
from pathlib import Path

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ARTIFACT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(ARTIFACT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT / "scripts"))
