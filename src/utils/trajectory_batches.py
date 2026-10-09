#!/usr/bin/env python3
"""Shared batch isolation and audit metadata for trajectory DB tables.

``scene_id`` identifies a physical/calibration scene. ``batch_id`` identifies
one capture/experiment dataset within that scene. Every consumer of trajectory
rows must filter by both keys so repeated experiments can share one SQLite DB.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable


_BATCH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


_BATCH_AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectory_batches (
    scene_id       TEXT NOT NULL,
    batch_id       TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    metadata_json  TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (scene_id, batch_id)
);

CREATE TABLE IF NOT EXISTS trajectory_batch_stages (
    run_id         TEXT NOT NULL,
    scene_id       TEXT NOT NULL,
    batch_id       TEXT NOT NULL,
    stage          TEXT NOT NULL,
    status         TEXT NOT NULL,
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    metadata_json  TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, scene_id)
);
CREATE INDEX IF NOT EXISTS idx_traj_batch_stages_lookup
    ON trajectory_batch_stages(scene_id, batch_id, stage, started_at);
"""

_BATCH_INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_traj_obs_scene_batch_global_time
    ON trajectory_observations(scene_id, batch_id, global_id, timestamp_sec);
CREATE INDEX IF NOT EXISTS idx_traj_obs_scene_batch_camera_local_time
    ON trajectory_observations(scene_id, batch_id, camera_name, local_track_id, timestamp_sec);
CREATE INDEX IF NOT EXISTS idx_traj_merges_scene_batch_kept
    ON trajectory_merges(scene_id, batch_id, kept_global_id, decided_at);
CREATE INDEX IF NOT EXISTS idx_traj_stats_scene_batch_time
    ON trajectory_runtime_stats(scene_id, batch_id, timestamp_sec);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_batch_id(batch_id: str) -> str:
    value = str(batch_id or "").strip()
    if not _BATCH_ID_RE.fullmatch(value):
        raise ValueError(
            '--batch 1-128'
            'See the HSG-SRL reproduction notes.'
        )
    return value


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def initialize_trajectory_batch_schema(conn: sqlite3.Connection) -> None:
    """Create batch audit tables/indexes for a newly created current DB."""
    from utils.scene_geometry_db import ensure_scene_geometry_schema

    conn.executescript(_BATCH_AUDIT_SCHEMA)
    conn.executescript(_BATCH_INDEX_SCHEMA)
    ensure_scene_geometry_schema(conn)
    conn.commit()


def require_trajectory_batch_schema(conn: sqlite3.Connection) -> None:
    """Raise if trajectory data tables have not been created yet."""
    from utils.scene_geometry_db import ensure_scene_geometry_schema

    if not _table_exists(conn, "trajectory_observations"):
        raise RuntimeError(
            'trajectory schema'
            'pipeline/multi_camera_trajectory_fusion.py'
        )
    # Older DBs may predate scene_overlap_pairs; create on open.
    ensure_scene_geometry_schema(conn)


def ensure_batch_record(
    conn: sqlite3.Connection,
    scene_id: str,
    batch_id: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    batch_id = validate_batch_id(batch_id)
    now = utc_now()
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
    conn.execute(
        """
        INSERT INTO trajectory_batches
            (scene_id, batch_id, created_at, updated_at, metadata_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(scene_id, batch_id) DO UPDATE SET updated_at=excluded.updated_at
        """,
        (scene_id, batch_id, now, now, metadata_json),
    )


def start_batch_stage(
    conn: sqlite3.Connection,
    scene_ids: Iterable[str],
    batch_id: str,
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    batch_id = validate_batch_id(batch_id)
    scenes = list(dict.fromkeys(str(scene).strip() for scene in scene_ids if str(scene).strip()))
    if not scenes:
        raise ValueError("batch stage requires at least one scene")
    run_id = uuid.uuid4().hex
    now = utc_now()
    payload = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
    with conn:
        for scene_id in scenes:
            ensure_batch_record(conn, scene_id, batch_id)
            conn.execute(
                """
                INSERT INTO trajectory_batch_stages
                    (run_id, scene_id, batch_id, stage, status, started_at, metadata_json)
                VALUES (?, ?, ?, ?, 'running', ?, ?)
                """,
                (run_id, scene_id, batch_id, stage, now, payload),
            )
    return run_id


def finish_batch_stage(
    conn: sqlite3.Connection,
    run_id: str,
    status: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    if status not in {"completed", "failed", "interrupted"}:
        raise ValueError(f"invalid batch stage status: {status}")
    rows = conn.execute(
        "SELECT scene_id, metadata_json FROM trajectory_batch_stages WHERE run_id=?",
        (run_id,),
    ).fetchall()
    now = utc_now()
    with conn:
        for scene_id, raw in rows:
            merged: dict[str, Any] = {}
            try:
                parsed = json.loads(raw or "{}")
                if isinstance(parsed, dict):
                    merged.update(parsed)
            except json.JSONDecodeError:
                pass
            merged.update(metadata or {})
            conn.execute(
                """
                UPDATE trajectory_batch_stages
                   SET status=?, finished_at=?, metadata_json=?
                 WHERE run_id=? AND scene_id=?
                """,
                (
                    status,
                    now,
                    json.dumps(merged, ensure_ascii=False, sort_keys=True),
                    run_id,
                    scene_id,
                ),
            )
            conn.execute(
                "UPDATE trajectory_batches SET updated_at=? WHERE scene_id=? AND batch_id="
                "(SELECT batch_id FROM trajectory_batch_stages WHERE run_id=? AND scene_id=?)",
                (now, scene_id, run_id, scene_id),
            )
