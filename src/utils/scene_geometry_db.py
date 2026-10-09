#!/usr/bin/env python3
"""Persist scene geometry metadata that used to live in ``*_import_meta.json``.

Stored in SQLite so training / low-order / evaluation stay coupled to the same DB
as observations and calibrations:

- ``scene_overlap_pairs``: MVC-eligible camera pairs (scene-scoped)
- ``trajectory_camera_config``: per-batch frame resolution (infer/calib wh)
- ``trajectory_batches.metadata_json``: optional import audit blob
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable


_SCENE_GEOMETRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS scene_overlap_pairs (
    scene_id   TEXT NOT NULL,
    camera_a   TEXT NOT NULL,
    camera_b   TEXT NOT NULL,
    PRIMARY KEY (scene_id, camera_a, camera_b)
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_scene_geometry_schema(conn: sqlite3.Connection) -> None:
    """Ensure geometry tables exist without changing transaction boundaries.

    Transaction ownership belongs to the caller.  In particular, importers
    must be able to roll back observations, calibrations and geometry metadata
    as one unit when any write fails.
    """
    conn.execute(_SCENE_GEOMETRY_SCHEMA)


def _norm_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))  # type: ignore[return-value]


def save_scene_overlap_pairs(
    conn: sqlite3.Connection,
    scene_id: str,
    pairs: Iterable[tuple[str, str] | list[str]],
    *,
    replace: bool = True,
) -> int:
    """Write overlap pairs for ``scene_id``. Returns number of pairs stored."""
    ensure_scene_geometry_schema(conn)
    if replace:
        conn.execute("DELETE FROM scene_overlap_pairs WHERE scene_id=?", (scene_id,))
    n = 0
    for pair in pairs:
        if len(pair) != 2:
            continue
        a, b = _norm_pair(pair[0], pair[1])
        if a == b:
            continue
        conn.execute(
            """
            INSERT OR IGNORE INTO scene_overlap_pairs (scene_id, camera_a, camera_b)
            VALUES (?, ?, ?)
            """,
            (scene_id, a, b),
        )
        n += 1
    return n


def load_scene_overlap_pairs(
    conn: sqlite3.Connection,
    scene_id: str,
    camera_names: Iterable[str] | None = None,
) -> set[tuple[str, str]]:
    ensure_scene_geometry_schema(conn)
    rows = conn.execute(
        "SELECT camera_a, camera_b FROM scene_overlap_pairs WHERE scene_id=?",
        (scene_id,),
    ).fetchall()
    camera_set = set(camera_names) if camera_names is not None else None
    out: set[tuple[str, str]] = set()
    for a, b in rows:
        pair = _norm_pair(a, b)
        if camera_set is not None and (pair[0] not in camera_set or pair[1] not in camera_set):
            continue
        out.add(pair)
    return out


def save_batch_camera_frame_wh(
    conn: sqlite3.Connection,
    scene_id: str,
    batch_id: str,
    frame_wh_by_camera: dict[str, tuple[int, int] | list[int]],
    *,
    calib_wh_by_camera: dict[str, tuple[int, int] | list[int]] | None = None,
) -> None:
    """Upsert infer/calib resolution into ``trajectory_camera_config``."""
    now = utc_now()
    calib_wh_by_camera = calib_wh_by_camera or {}
    for cam, wh in frame_wh_by_camera.items():
        if not isinstance(wh, (list, tuple)) or len(wh) != 2:
            continue
        iw, ih = int(wh[0]), int(wh[1])
        if iw <= 0 or ih <= 0:
            continue
        cwh = calib_wh_by_camera.get(cam, wh)
        cw = ch = None
        if isinstance(cwh, (list, tuple)) and len(cwh) == 2:
            cw, ch = int(cwh[0]), int(cwh[1])
        conn.execute(
            """
            INSERT INTO trajectory_camera_config
                (scene_id, batch_id, camera_name, infer_w, infer_h, calib_w, calib_h, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scene_id, batch_id, camera_name) DO UPDATE SET
                infer_w=excluded.infer_w,
                infer_h=excluded.infer_h,
                calib_w=excluded.calib_w,
                calib_h=excluded.calib_h,
                updated_at=excluded.updated_at
            """,
            (scene_id, batch_id, str(cam), iw, ih, cw, ch, now),
        )


def load_batch_camera_frame_wh(
    conn: sqlite3.Connection,
    scene_id: str,
    batch_id: str,
    camera_names: Iterable[str],
) -> dict[str, tuple[int, int] | None]:
    names = list(camera_names)
    out: dict[str, tuple[int, int] | None] = {cam: None for cam in names}
    if not names:
        return out
    rows = conn.execute(
        "SELECT camera_name, infer_w, infer_h "
        "FROM trajectory_camera_config WHERE scene_id=? AND batch_id=?",
        (scene_id, batch_id),
    ).fetchall()
    for name, iw, ih in rows:
        if name in out and iw and ih and int(iw) > 0 and int(ih) > 0:
            out[name] = (int(iw), int(ih))
    return out


def save_import_audit_metadata(
    conn: sqlite3.Connection,
    scene_id: str,
    batch_id: str,
    metadata: dict[str, Any],
) -> None:
    """Merge import audit fields into ``trajectory_batches.metadata_json``."""
    from utils.trajectory_batches import ensure_batch_record, validate_batch_id

    validate_batch_id(batch_id)
    ensure_batch_record(conn, scene_id, batch_id, metadata=None)
    row = conn.execute(
        "SELECT metadata_json FROM trajectory_batches WHERE scene_id=? AND batch_id=?",
        (scene_id, batch_id),
    ).fetchone()
    existing: dict[str, Any] = {}
    if row and row[0]:
        try:
            existing = json.loads(row[0])
        except json.JSONDecodeError:
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing.update(metadata)
    now = utc_now()
    conn.execute(
        """
        UPDATE trajectory_batches
           SET metadata_json=?, updated_at=?
         WHERE scene_id=? AND batch_id=?
        """,
        (json.dumps(existing, ensure_ascii=False, sort_keys=True), now, scene_id, batch_id),
    )


def load_import_audit_metadata(
    conn: sqlite3.Connection,
    scene_id: str,
    batch_id: str,
) -> dict[str, Any]:
    row = conn.execute(
        "SELECT metadata_json FROM trajectory_batches WHERE scene_id=? AND batch_id=?",
        (scene_id, batch_id),
    ).fetchone()
    if not row or not row[0]:
        return {}
    try:
        data = json.loads(row[0])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
