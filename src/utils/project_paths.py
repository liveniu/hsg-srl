'YAMLdata/ ``../`` \n ``py tools/print_ChArUco.py`` \n``<>/data/charuco/``'

from __future__ import annotations

from pathlib import Path

                                             
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # artifact root


def resolve_project_path(path: str | Path, *, base: Path | None = None) -> Path:
    'base'
    p = Path(path)
    if p.is_absolute():
        return p
    root = PROJECT_ROOT if base is None else base
    return (root / p).resolve()
