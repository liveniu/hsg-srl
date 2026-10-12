#!/usr/bin/env python3
"""Mark ChArUco inner corners by hand and write them back to the scene YAML.

Use this when the image is too blurred for automatic ArUco detection.
Corners are stored on cameras[].manual_charuco_corners. Stitch and diagnose
prefer those corners and do not require a successful automatic detection.

The window shows the camera image on the left, a board diagram on the right,
and a status line on top. Saved coordinates are original-image pixels. The
view may scale the long side to 1100 px; the wheel and middle button change
only the view.

Mouse, on the image: left places or moves the selected corner, right deletes
it, middle-drag pans, and the wheel zooms. A left click on the diagram
selects that corner.

Keys: N or Tab next corner, p previous, V two-point manual horizon, Shift+P
preview homography refinement, +/- zoom, arrows pan, 0 reset the view, Enter
or space confirm a board with at least four corners, S skip, H compute the
homography horizon, R reset this board, Esc or Q quit and keep confirmed cameras.

    python tools/mark_charuco_corners.py --config configs/board_capture/scene.yaml
    python tools/mark_charuco_corners.py --config configs/board_capture/scene.yaml --cameras A B
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import math
import re
from typing import Any

import cv2 as cv
import numpy as np
import yaml

from utils.image_text import draw_text_block, put_text, warn_if_no_pil
from utils.project_paths import resolve_project_path
from pipeline.multi_camera_bev_stitch import (
    charuco_corner_local_xy,
    charuco_marker_id_grid,
    estimate_image_to_bev_homography,
    homography_horizon_line,
    horizon_line_from_points,
    horizon_line_to_segment,
    load_manual_horizon_line,
    refine_homography,
    CALIBRATION_DIAGNOSTICS_DIR,
    run_bfs_joint_refine_preview,
    world_to_bev_points,
)

# Maximum image-view width and height in pixels.
MAX_IMG_PX = 1100
# Right-panel width.
PANEL_W = 250
# Status-bar height.
BAR_H = 30

# Colors are BGR.
COLOR_MARKED   = (50, 220, 50)    # marked corner, green
COLOR_SEL_SET  = (0, 100, 255)    # selected marked corner
COLOR_AUTO     = (60, 180, 240)   # automatic prefill, blue
COLOR_LABEL    = (0, 255, 255)    # corner ID label
COLOR_INFO     = (200, 200, 60)   # help text
COLOR_WARN     = (80, 80, 255)    # warning text
COLOR_HORIZON  = (255, 80, 220)   # homography horizon
COLOR_MANUAL_HORIZON = (255, 255, 0)  # manual horizon, cyan
COLOR_REFINED_HORIZON = (60, 220, 60)  # refined homography horizon
FONT = cv.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────────────────────────────────────────
# YAML text helpers.
# ──────────────────────────────────────────────────────────────────────────────

def _find_camera_block(lines: list[str], cam_name: str) -> tuple[int, int]:
    """Find the start and end lines of one camera block."""
    pattern = re.compile(
        r"^\s+-\s+name:\s+['\"]?" + re.escape(cam_name) + r"['\"]?\s*$"
    )
    start = None
    for i, line in enumerate(lines):
        if pattern.match(line):
            start = i
            break
    if start is None:
        raise ValueError(f"YAML has no camera '{cam_name}'")
    next_cam = re.compile(r"^\s+-\s+name:\s+")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if next_cam.match(lines[i]):
            end = i
            break
    return start, end


def _find_key_block(
    lines: list[str], block_start: int, block_end: int, key: str
) -> tuple[int, int] | None:
    """Find a key block inside one camera, inclusive."""
    key_re = re.compile(r"^(\s+)" + re.escape(key) + r"\s*:")
    for i in range(block_start, block_end):
        m = key_re.match(lines[i])
        if not m:
            continue
        indent = len(m.group(1))
        key_end = block_end
        for j in range(i + 1, block_end):
            stripped = lines[j].lstrip()
            if stripped and not stripped.startswith("#"):
                if len(lines[j]) - len(stripped) <= indent:
                    key_end = j
                    break
        return i, key_end
    return None


def _find_charuco_boards_line(
    lines: list[str], block_start: int, block_end: int
) -> int | None:
    """Find the charuco_boards line so manual_charuco_corners can be inserted before it."""
    for i in range(block_start, block_end):
        if re.match(r"\s+charuco_boards\s*:", lines[i]):
            return i
    return None


def _manual_corners_yaml_lines(
    corners_by_board: dict[str, dict[int, list[int]]],
) -> list[str]:
    """Render manual_charuco_corners YAML lines at a 4-space indent.

    Each record is - [corner_id, image_x, image_y].
    """
    out = ["    manual_charuco_corners:"]
    for board_name in sorted(corners_by_board):
        corners = corners_by_board[board_name]
        out.append(f"      {board_name}:")
        for cid in sorted(corners):
            x, y = corners[cid]
            out.append(f"        - [{cid}, {x}, {y}]")
    return out


def update_manual_corners_in_yaml_text(
    text: str,
    cam_name: str,
    board_name: str,
    corners: dict[int, list[int]] | None,
) -> str:
    """Update manual_charuco_corners for one camera and board.

    corners=None deletes that board. Other boards stay.
    """
    # Parse existing entries so other boards are kept.
    cfg = yaml.safe_load(text)
    cam_data = next(
        (c for c in cfg.get("cameras", []) if c.get("name") == cam_name), None
    )
    if cam_data is None:
        raise ValueError(f"YAML has no camera '{cam_name}'")

    existing: dict[str, dict[int, list[int]]] = {}
    for bname, entries in (cam_data.get("manual_charuco_corners") or {}).items():
        if entries:
            existing[str(bname)] = {int(e[0]): [int(e[1]), int(e[2])] for e in entries}

    if corners is not None:
        existing[board_name] = {cid: [int(x), int(y)] for cid, (x, y) in corners.items()}
    else:
        existing.pop(board_name, None)

    lines = text.splitlines()
    block_start, block_end = _find_camera_block(lines, cam_name)
    mc_block = _find_key_block(lines, block_start, block_end, "manual_charuco_corners")

    new_lines = _manual_corners_yaml_lines(existing) if existing else []

    if mc_block is not None:
        ms, me = mc_block
        result = (lines[:ms] + new_lines + lines[me:]) if new_lines else (lines[:ms] + lines[me:])
    else:
        if not new_lines:
            return text
        insert_at = _find_charuco_boards_line(lines, block_start, block_end)
        insert_at = insert_at if insert_at is not None else block_end
        result = lines[:insert_at] + new_lines + lines[insert_at:]

    return "\n".join(result)


def _fmt_float(value: float, digits: int = 10) -> str:
    return f"{float(value):.{digits}g}"


def _horizon_yaml_lines(horizons_by_board: dict[str, dict[str, Any]]) -> list[str]:
    """Render manual_homography_horizon for the horizon computed from one board."""
    out = ["    manual_homography_horizon:"]
    for board_name in sorted(horizons_by_board):
        item = horizons_by_board[board_name]
        line = [float(v) for v in item.get("line", [])]
        out.append(f"      {board_name}:")
        if len(line) == 3:
            out.append(
                "        line: ["
                + ", ".join(_fmt_float(v) for v in line)
                + "]"
            )
        out.append(
            "        ground_side_positive: "
            + ("true" if bool(item.get("ground_side_positive", True)) else "false")
        )
        for key in ("point_count", "inlier_count"):
            if key in item:
                out.append(f"        {key}: {int(item[key])}")
        if "reproj_rms_px" in item:
            out.append(f"        reproj_rms_px: {_fmt_float(float(item['reproj_rms_px']), 6)}")
        homography = item.get("homography")
        if homography is not None:
            out.append("        homography:")
            for row in homography:
                out.append(
                    "          - ["
                    + ", ".join(_fmt_float(float(v)) for v in row)
                    + "]"
                )
    return out


def update_horizon_in_yaml_text(
    text: str,
    cam_name: str,
    board_name: str,
    horizon: dict[str, Any] | None,
) -> str:
    """Update manual_homography_horizon for one camera and board."""
    cfg = yaml.safe_load(text)
    cam_data = next(
        (c for c in cfg.get("cameras", []) if c.get("name") == cam_name), None
    )
    if cam_data is None:
        raise ValueError(f"YAML has no camera '{cam_name}'")

    existing: dict[str, dict[str, Any]] = {}
    raw = cam_data.get("manual_homography_horizon") or {}
    if isinstance(raw, dict):
        for bname, item in raw.items():
            if isinstance(item, dict):
                existing[str(bname)] = dict(item)

    if horizon is not None:
        existing[board_name] = horizon
    else:
        existing.pop(board_name, None)

    lines = text.splitlines()
    block_start, block_end = _find_camera_block(lines, cam_name)
    horizon_block = _find_key_block(lines, block_start, block_end, "manual_homography_horizon")
    new_lines = _horizon_yaml_lines(existing) if existing else []

    if horizon_block is not None:
        hs, he = horizon_block
        result = (lines[:hs] + new_lines + lines[he:]) if new_lines else (lines[:hs] + lines[he:])
    else:
        if not new_lines:
            return text
        insert_at = _find_charuco_boards_line(lines, block_start, block_end)
        insert_at = insert_at if insert_at is not None else block_end
        result = lines[:insert_at] + new_lines + lines[insert_at:]

    return "\n".join(result)


def _manual_horizon_line_yaml_lines(horizon: dict[str, Any]) -> list[str]:
    """Render the camera-level manual_horizon_line block."""
    line = [float(v) for v in horizon.get("line", [])]
    points = horizon.get("points") or []
    out = ["    manual_horizon_line:"]
    if len(line) == 3:
        out.append(
            "      line: ["
            + ", ".join(_fmt_float(v) for v in line)
            + "]"
        )
    if len(points) >= 2:
        out.append("      points:")
        for p in points[:2]:
            out.append(
                "        - ["
                + ", ".join(_fmt_float(float(v), 3) for v in p[:2])
                + "]"
            )
    out.append(
        "      ground_side_positive: "
        + ("true" if bool(horizon.get("ground_side_positive", True)) else "false")
    )
    return out


def update_manual_horizon_line_in_yaml_text(
    text: str,
    cam_name: str,
    horizon: dict[str, Any] | None,
) -> str:
    """Update the camera-level manual_horizon_line."""
    lines = text.splitlines()
    block_start, block_end = _find_camera_block(lines, cam_name)
    old_block = _find_key_block(lines, block_start, block_end, "manual_horizon_line")
    new_lines = _manual_horizon_line_yaml_lines(horizon) if horizon is not None else []
    if old_block is not None:
        hs, he = old_block
        result = (lines[:hs] + new_lines + lines[he:]) if new_lines else (lines[:hs] + lines[he:])
    else:
        if not new_lines:
            return text
        insert_at = _find_charuco_boards_line(lines, block_start, block_end)
        result = lines[:insert_at] + new_lines + lines[insert_at:]
    return "\n".join(result) + "\n"
def _line_segment_in_image(
    line: list[float] | tuple[float, float, float],
    width: int,
    height: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return the visible segment of ax+by+c=0 inside the image rectangle."""
    a, b, c = [float(v) for v in line]
    eps = 1e-9
    max_x = max(float(width - 1), 0.0)
    max_y = max(float(height - 1), 0.0)
    pts: list[tuple[float, float]] = []

    if abs(b) > eps:
        for x in (0.0, max_x):
            y = -(a * x + c) / b
            if -1e-6 <= y <= max_y + 1e-6:
                pts.append((x, min(max(y, 0.0), max_y)))
    if abs(a) > eps:
        for y in (0.0, max_y):
            x = -(b * y + c) / a
            if -1e-6 <= x <= max_x + 1e-6:
                pts.append((min(max(x, 0.0), max_x), y))

    unique: list[tuple[float, float]] = []
    for pt in pts:
        if all((pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2 > 1e-6 for q in unique):
            unique.append(pt)
    if len(unique) < 2:
        return None

    best = (unique[0], unique[1])
    best_d2 = -1.0
    for i, p in enumerate(unique):
        for q in unique[i + 1:]:
            d2 = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
            if d2 > best_d2:
                best = (p, q)
                best_d2 = d2
    return best


def compute_manual_horizon(
    corners: dict[int, list[float]],
    board_cfg: dict[str, Any],
    scale_px_per_meter: float,
    ransac_reproj_threshold: float,
    frame_wh: tuple[int, int],
) -> tuple[dict[str, Any], tuple[tuple[float, float], tuple[float, float]] | None]:
    """Compute the homography horizon with the same findHomography path as stitch."""
    if len(corners) < 4:
        raise ValueError("At least four corners are required to estimate a homography")

    sx = int(board_cfg["squares_x"])
    sq_len = float(board_cfg["square_length"])
    image_pts: list[list[float]] = []
    world_pts: list[list[float]] = []
    for cid, (x, y) in sorted(corners.items()):
        lx, ly = charuco_corner_local_xy(int(cid), sx, sq_len)
        image_pts.append([float(x), float(y)])
        world_pts.append([lx, ly])

    image_arr = np.asarray(image_pts, dtype=np.float32)
    bev_arr = world_to_bev_points(np.asarray(world_pts, dtype=np.float32), scale_px_per_meter)
    H, inliers = estimate_image_to_bev_homography(
        image_arr,
        bev_arr,
        ransac_reproj_threshold,
    )

    if inliers is None:
        inlier_mask = np.ones(len(image_arr), dtype=bool)
    else:
        inlier_mask = inliers.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < 4:
        raise RuntimeError(f"Homography has fewer than 4 inliers (now {inlier_count})")

    src_h = np.c_[image_arr[inlier_mask], np.ones(inlier_count, dtype=np.float64)]
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    reproj_rms = float(np.sqrt(np.mean(np.sum((proj - bev_arr[inlier_mask]) ** 2, axis=1))))

    line = H[2, :].astype(np.float64)
    norm = math.hypot(float(line[0]), float(line[1]))
    if norm < 1e-12:
        raise RuntimeError("The homography is degenerate, so the horizon is not stable")
    line = line / norm

    fw, fh = frame_wh
    ground_probe = np.array([float(fw) * 0.5, max(float(fh) - 1.0, 0.0), 1.0])
    if float(line @ ground_probe) < 0.0:
        line = -line

    result = {
        "line": [float(line[0]), float(line[1]), float(line[2])],
        "ground_side_positive": True,
        "point_count": int(len(image_arr)),
        "inlier_count": inlier_count,
        "reproj_rms_px": reproj_rms,
        "homography": [[float(v) for v in row] for row in H.tolist()],
    }
    segment = _line_segment_in_image(result["line"], fw, fh)
    return result, segment


# ──────────────────────────────────────────────────────────────────────────────
# Interactive marker.
# ──────────────────────────────────────────────────────────────────────────────

class CornerMarker:
    """Marker for one image and one ChArUco board.

    Corners are stored in original-image pixels. View zoom and pan are not written.
    """

    ZOOM_MIN = 1.0
    ZOOM_MAX = 20.0
    ZOOM_STEP = 1.15

    def __init__(
        self,
        image: np.ndarray,
        camera_name: str,
        board_name: str,
        board_cfg: dict[str, Any],
        existing_corners: dict[int, list[float]] | None,
        auto_detected: dict[int, list[float]] | None = None,
        scale_px_per_meter: float = 100.0,
        ransac_reproj_threshold: float = 4.0,
        existing_manual_horizon: dict[str, Any] | None = None,
        preview_lambda_horizon: float = 1.0,
        preview_max_iters: int = 30,
    ) -> None:
        self.orig = image
        self.cam_name = camera_name
        self.board_name = board_name
        self.cfg = board_cfg
        self.scale_px_per_meter = float(scale_px_per_meter)
        self.ransac_reproj_threshold = float(ransac_reproj_threshold)
        self.preview_lambda_horizon = float(preview_lambda_horizon)
        self.preview_max_iters = int(preview_max_iters)

        self.sx = int(board_cfg["squares_x"])
        self.sy = int(board_cfg["squares_y"])
        self.total = (self.sx - 1) * (self.sy - 1)
        self.marker_grid = charuco_marker_id_grid(board_cfg)

        # Initial fit scale. Display only.
        h, w = image.shape[:2]
        self.base_scale = min(MAX_IMG_PX / w, MAX_IMG_PX / h, 1.0)
        self.disp_w = max(1, int(w * self.base_scale))
        self.disp_h = max(1, int(h * self.base_scale))

        # View zoom and pan. Display only.
        self.view_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_drag: tuple[int, int, float, float] | None = None

        # Marked corners {corner_id: [orig_x, orig_y]} in original pixels.
        self.marked: dict[int, list[float]] = {}
        self.auto_ids: set[int] = set()

        if auto_detected:
            for cid, (ox, oy) in auto_detected.items():
                self.marked[cid] = [float(ox), float(oy)]
                self.auto_ids.add(cid)

        if existing_corners:
            for cid, (ox, oy) in existing_corners.items():
                self.marked[cid] = [float(ox), float(oy)]
                self.auto_ids.discard(cid)

        self.selected_id = 0
        self.confirmed = False
        self.skipped = False
        self.quit = False
        self.horizon_result: dict[str, Any] | None = None
        self.horizon_segment: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.horizon_line_mode = False
        self.manual_horizon_result: dict[str, Any] | None = existing_manual_horizon
        self.manual_horizon_dirty = False
        self.manual_horizon_points: list[list[float]] = []
        if existing_manual_horizon and existing_manual_horizon.get("points"):
            self.manual_horizon_points = [
                [float(p[0]), float(p[1])] for p in existing_manual_horizon.get("points", [])[:2]
            ]
        self.manual_horizon_segment = (
            horizon_line_to_segment(existing_manual_horizon["line"], (self.orig.shape[1], self.orig.shape[0]))
            if existing_manual_horizon and existing_manual_horizon.get("line") is not None
            else None
        )
        self.refined_horizon_segment: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.refine_preview_result: dict[str, Any] | None = None
        self.status_note: str | None = None
        self.status_warn = False

    # ── coordinates and view ───────────────────────────────────────────────────────────

    @property
    def eff_scale(self) -> float:
        return self.base_scale * self.view_zoom

    def orig_to_view(self, ox: float, oy: float) -> tuple[float, float]:
        return ox * self.eff_scale - self.pan_x, oy * self.eff_scale - self.pan_y

    def view_to_orig(self, vx: float, vy: float) -> tuple[float, float]:
        s = self.eff_scale
        return (vx + self.pan_x) / s, (vy + self.pan_y) / s

    def _scaled_size(self) -> tuple[int, int]:
        h, w = self.orig.shape[:2]
        return int(round(w * self.eff_scale)), int(round(h * self.eff_scale))

    def _clamp_pan(self) -> None:
        sw, sh = self._scaled_size()
        self.pan_x = float(np.clip(self.pan_x, 0.0, max(0.0, sw - self.disp_w)))
        self.pan_y = float(np.clip(self.pan_y, 0.0, max(0.0, sh - self.disp_h)))

    def _reset_view(self) -> None:
        self.view_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    def _zoom_at(self, vx: float, vy: float, factor: float) -> None:
        ox, oy = self.view_to_orig(vx, vy)
        self.view_zoom = float(np.clip(self.view_zoom * factor, self.ZOOM_MIN, self.ZOOM_MAX))
        self.pan_x = ox * self.eff_scale - vx
        self.pan_y = oy * self.eff_scale - vy
        self._clamp_pan()

    def _pan_by(self, dx: float, dy: float) -> None:
        self.pan_x -= dx
        self.pan_y -= dy
        self._clamp_pan()

    def to_orig(self) -> dict[int, list[int]]:
        """Return original-image pixels. YAML stores the rounded values."""
        return {
            cid: [int(round(xy[0])), int(round(xy[1]))]
            for cid, xy in self.marked.items()
        }

    # ── diagram coordinates ────────────────────────────────────────────────────────

    def _schema_params(self) -> tuple[float, float, float]:
        """Return (ox, oy, cell_sz) for the diagram and hit tests."""
        margin = 14
        max_w = PANEL_W - 2 * margin
        max_h = self.disp_h // 2
        cell_sz = max(min(max_w / self.sx, max_h / self.sy), 10.0)
        grid_w = cell_sz * self.sx
        grid_h = cell_sz * self.sy
        ox = margin + (max_w - grid_w) / 2
        oy = margin + 8.0
        return ox, oy, cell_sz

    def _schema_corner_pos(self, cid: int) -> tuple[int, int]:
        ox, oy, cell_sz = self._schema_params()
        cols = self.sx - 1
        col = cid % cols
        row = cid // cols
        return int(ox + (col + 1) * cell_sz), int(oy + (row + 1) * cell_sz)

    # ── rendering ──────────────────────────────────────────────────────────────────

    def _draw_schematic(self, panel: np.ndarray) -> None:
        ox, oy, cell_sz = self._schema_params()
        grid_w = cell_sz * self.sx
        grid_h = cell_sz * self.sy

        # Checker fill and the ArUco ID in each square.
        for row in range(self.sy):
            for col in range(self.sx):
                x0 = int(ox + col * cell_sz)
                y0 = int(oy + row * cell_sz)
                x1 = int(ox + (col + 1) * cell_sz)
                y1 = int(oy + (row + 1) * cell_sz)
                marker_id = int(self.marker_grid[row, col])
                if marker_id >= 0:
                    cv.rectangle(panel, (x0, y0), (x1, y1), (45, 45, 45), -1)
                    label = str(marker_id)
                    font_scale = max(0.28, min(0.55, cell_sz / 22.0))
                    thick = 1 if cell_sz < 28 else 2
                    (tw, th), baseline = cv.getTextSize(label, FONT, font_scale, thick)
                    tx = x0 + max(1, int((cell_sz - tw) / 2))
                    ty = y0 + max(th + 1, int((cell_sz + th) / 2))
                    cv.putText(
                        panel, label, (tx, ty), FONT, font_scale,
                        (180, 220, 255), thick, cv.LINE_AA,
                    )
                elif (row + col) % 2 == 0:
                    cv.rectangle(panel, (x0, y0), (x1, y1), (34, 34, 34), -1)

        # Grid lines.
        for i in range(self.sx + 1):
            x = int(ox + i * cell_sz)
            cv.line(panel, (x, int(oy)), (x, int(oy + grid_h)), (80, 80, 80), 1)
        for j in range(self.sy + 1):
            y = int(oy + j * cell_sz)
            cv.line(panel, (int(ox), y), (int(ox + grid_w), y), (80, 80, 80), 1)

        # Inner corners and their ChArUco indexes, not ArUco IDs.
        cols = self.sx - 1
        for cid in range(self.total):
            ix, iy = self._schema_corner_pos(cid)
            if cid == self.selected_id:
                color = (0, 100, 255)
                cv.circle(panel, (ix, iy), 5, color, -1)
                cv.circle(panel, (ix, iy), 6, (255, 255, 255), 1)
            elif cid in self.auto_ids and cid in self.marked:
                color = COLOR_AUTO
                cv.circle(panel, (ix, iy), 4, color, -1)
            elif cid in self.marked:
                color = COLOR_MARKED
                cv.circle(panel, (ix, iy), 4, color, -1)
            else:
                color = (80, 80, 80)
                cv.circle(panel, (ix, iy), 3, color, 1)

            font_scale = max(0.27, min(0.38, cell_sz / 28))
            col = cid % cols
            row = cid // cols
            label = str(cid)
            tx = ix + 3 if col < cols - 1 else ix - 11
            ty = iy - 3 if row > 0 else iy + 10
            cv.putText(panel, label, (tx, ty), FONT, font_scale, (150, 150, 150), 1, cv.LINE_AA)

        # Outer-corner labels.
        ext = [(int(ox) - 2, int(oy) - 2, "TL"),
               (int(ox + grid_w) - 12, int(oy) - 2, "TR"),
               (int(ox) - 2, int(oy + grid_h) + 10, "BL"),
               (int(ox + grid_w) - 12, int(oy + grid_h) + 10, "BR")]
        for ex, ey, label in ext:
            cv.putText(panel, label, (ex, ey), FONT, 0.3, (70, 140, 70), 1, cv.LINE_AA)

    def _make_panel(self) -> np.ndarray:
        panel = np.full((self.disp_h, PANEL_W, 3), (28, 28, 28), dtype=np.uint8)
        self._draw_schematic(panel)

        n = len(self.marked)
        n_auto = len(self.auto_ids & self.marked.keys())
        zoom_pct = int(round(self.view_zoom * 100))
        info_lines = [
            f"corner #{self.selected_id}",
            f"marked: {n}/{self.total} (automatic:{n_auto})",
            f"zoom: {zoom_pct}%",
            "",
            "number in a square = ArUco ID",
            "number at a crossing = inner-corner index",
            "",
            "left button = place corner",
            "right button = delete selected corner",
            "middle drag = pan",
            "wheel = zoom",
            "click the diagram = select corner",
            "",
            "N / Tab = next corner",
            "p       = previous corner",
            "V       = two-point manual horizon",
            "Shift+P = preview refinement",
            "+ / -   = zoom",
            "arrows  = pan",
            "0       = reset view",
            "H       = homography horizon",
            "",
            "Enter   = confirm (at least 4)",
            "S       = skip",
            "R       = reset this board",
            "Esc/Q   = quit",
        ]
        y0 = self.disp_h // 2 + 8
        draw_text_block(
            panel,
            info_lines,
            origin=(8, y0),
            color_bgr=COLOR_INFO,
            font_size=13,
            line_height=19,
            pad=4,
            bg=False,
        )
        return panel

    def _render_image_view(self) -> np.ndarray:
        """Crop the original image to the current view."""
        sw, sh = self._scaled_size()
        interp = cv.INTER_LINEAR if self.view_zoom > 1.0 else cv.INTER_AREA
        scaled = cv.resize(self.orig, (sw, sh), interpolation=interp)
        self._clamp_pan()
        px, py = int(round(self.pan_x)), int(round(self.pan_y))
        return scaled[py : py + self.disp_h, px : px + self.disp_w].copy()

    def _draw_horizon_overlay(self, canvas: np.ndarray) -> None:
        if self.horizon_segment is not None:
            p0, p1 = self.horizon_segment
            x0, y0 = self.orig_to_view(p0[0], p0[1])
            x1, y1 = self.orig_to_view(p1[0], p1[1])
            ok, c0, c1 = cv.clipLine(
                (0, 0, self.disp_w, self.disp_h),
                (int(round(x0)), int(round(y0))),
                (int(round(x1)), int(round(y1))),
            )
            if ok:
                cv.line(canvas, c0, c1, COLOR_HORIZON, 2, cv.LINE_AA)
                cv.line(canvas, c0, c1, (255, 255, 255), 1, cv.LINE_AA)

        if self.manual_horizon_segment is not None:
            p0, p1 = self.manual_horizon_segment
            x0, y0 = self.orig_to_view(p0[0], p0[1])
            x1, y1 = self.orig_to_view(p1[0], p1[1])
            ok, c0, c1 = cv.clipLine(
                (0, 0, self.disp_w, self.disp_h),
                (int(round(x0)), int(round(y0))),
                (int(round(x1)), int(round(y1))),
            )
            if ok:
                cv.line(canvas, c0, c1, COLOR_MANUAL_HORIZON, 2, cv.LINE_AA)
                cv.putText(canvas, "manual", c0, FONT, 0.55, COLOR_MANUAL_HORIZON, 2, cv.LINE_AA)
        for hp in self.manual_horizon_points[:2]:
            vx, vy = self.orig_to_view(hp[0], hp[1])
            if 0 <= vx < self.disp_w and 0 <= vy < self.disp_h:
                cv.circle(canvas, (int(round(vx)), int(round(vy))), 7, COLOR_MANUAL_HORIZON, -1, cv.LINE_AA)
                cv.circle(canvas, (int(round(vx)), int(round(vy))), 8, (0, 0, 0), 1, cv.LINE_AA)

        if self.refined_horizon_segment is not None:
            p0, p1 = self.refined_horizon_segment
            x0, y0 = self.orig_to_view(p0[0], p0[1])
            x1, y1 = self.orig_to_view(p1[0], p1[1])
            ok, c0, c1 = cv.clipLine(
                (0, 0, self.disp_w, self.disp_h),
                (int(round(x0)), int(round(y0))),
                (int(round(x1)), int(round(y1))),
            )
            if ok:
                cv.line(canvas, c0, c1, COLOR_REFINED_HORIZON, 2, cv.LINE_AA)
                cv.putText(canvas, "refined", c0, FONT, 0.55, COLOR_REFINED_HORIZON, 2, cv.LINE_AA)

        if self.horizon_line_mode:
            put_text(canvas, "V mode: left-click two horizon points; right-click clears; V exits", (8, 32), COLOR_MANUAL_HORIZON, bg=True, font_size=14)
        if self.status_note:
            color = COLOR_WARN if self.status_warn else COLOR_HORIZON
            put_text(canvas, self.status_note, (8, 8), color, bg=True, font_size=14)

    def render(self) -> np.ndarray:
        canvas = self._render_image_view()
        self._draw_horizon_overlay(canvas)

        for cid, (ox, oy) in self.marked.items():
            vx, vy = self.orig_to_view(ox, oy)
            x, y = int(round(vx)), int(round(vy))
            if not (0 <= x < self.disp_w and 0 <= y < self.disp_h):
                continue
            if cid == self.selected_id:
                color = COLOR_SEL_SET
            elif cid in self.auto_ids:
                color = COLOR_AUTO
            else:
                color = COLOR_MARKED
            cv.circle(canvas, (x, y), 8, color, -1, cv.LINE_AA)
            cv.circle(canvas, (x, y), 9, (255, 255, 255), 1, cv.LINE_AA)
            prefix = "A" if cid in self.auto_ids else ""
            cv.putText(canvas, f"{prefix}{cid}", (x + 11, y - 6),
                       FONT, 0.48, COLOR_LABEL, 1, cv.LINE_AA)

        n_marked = len(self.marked)
        oh, ow = self.orig.shape[:2]
        status = (
            f"  camera {self.cam_name}  |  board {self.board_name}  "
            f"|  image {ow}×{oh}  |  corner #{self.selected_id}  "
            f"|  marked {n_marked}/{self.total}  |  zoom {int(round(self.view_zoom * 100))}%"
        )
        bar = np.full((BAR_H, self.disp_w + PANEL_W, 3), (38, 38, 38), dtype=np.uint8)
        put_text(bar, status, (8, 4), COLOR_INFO, bg=False, font_size=13)
        if n_marked < 4:
            warn = "  [need at least 4 corners to confirm]"
            put_text(bar, warn, (bar.shape[1] - 360, 4), COLOR_WARN, bg=False, font_size=13)

        row = np.hstack([canvas, self._make_panel()])
        return np.vstack([bar, row])

    # ── mouse callback ──────────────────────────────────────────────────────────────

    def mouse_cb(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        img_y = y - BAR_H
        if img_y < 0:
            return

        if x >= self.disp_w:
            if event == cv.EVENT_LBUTTONDOWN:
                panel_x = x - self.disp_w
                panel_y = img_y
                min_dist_sq = 22 ** 2
                nearest = -1
                for cid in range(self.total):
                    ix, iy = self._schema_corner_pos(cid)
                    d = (panel_x - ix) ** 2 + (panel_y - iy) ** 2
                    if d < min_dist_sq:
                        min_dist_sq = d
                        nearest = cid
                if nearest >= 0:
                    self.selected_id = nearest
            return

        if event == cv.EVENT_MOUSEWHEEL:
            if hasattr(cv, "getMouseWheelDelta"):
                delta = cv.getMouseWheelDelta(flags)
            else:
                delta = flags >> 16
                if delta == 0:
                    delta = flags
            factor = self.ZOOM_STEP if delta > 0 else 1.0 / self.ZOOM_STEP
            self._zoom_at(float(x), float(img_y), factor)
            return

        if event == cv.EVENT_MBUTTONDOWN:
            self._pan_drag = (x, img_y, self.pan_x, self.pan_y)
            return

        if event == cv.EVENT_MBUTTONUP:
            self._pan_drag = None
            return

        if event == cv.EVENT_MOUSEMOVE and self._pan_drag is not None:
            sx, sy, px0, py0 = self._pan_drag
            self.pan_x = px0 - (x - sx)
            self.pan_y = py0 - (img_y - sy)
            self._clamp_pan()
            return

        if self.horizon_line_mode:
            if event == cv.EVENT_LBUTTONDOWN:
                ox, oy = self.view_to_orig(float(x), float(img_y))
                if len(self.manual_horizon_points) >= 2:
                    self.manual_horizon_points = []
                self.manual_horizon_points.append([float(ox), float(oy)])
                self._update_manual_horizon_from_points()
            elif event == cv.EVENT_RBUTTONDOWN:
                self._clear_manual_horizon()
            return

        if event == cv.EVENT_LBUTTONDOWN:
            ox, oy = self.view_to_orig(float(x), float(img_y))
            self.marked[self.selected_id] = [ox, oy]
            self.auto_ids.discard(self.selected_id)
            self._invalidate_horizon()
            self._advance_to_next_unmarked()
        elif event == cv.EVENT_RBUTTONDOWN:
            if self.selected_id in self.marked:
                del self.marked[self.selected_id]
                self.auto_ids.discard(self.selected_id)
                self._invalidate_horizon()

    def _advance_to_next_unmarked(self) -> None:
        for delta in range(1, self.total + 1):
            nxt = (self.selected_id + delta) % self.total
            if nxt not in self.marked:
                self.selected_id = nxt
                return

    def _invalidate_horizon(self) -> None:
        if self.horizon_result is not None or self.horizon_segment is not None:
            self.horizon_result = None
            self.horizon_segment = None
            self.status_note = "Corners changed. Press H to recompute the horizon."
            self.status_warn = True

    def _compute_homography_horizon(self) -> None:
        if len(self.marked) < 4:
            self.status_note = "Need at least 4 corners to compute the horizon"
            self.status_warn = True
            return
        try:
            result, segment = compute_manual_horizon(
                self.marked,
                self.cfg,
                self.scale_px_per_meter,
                self.ransac_reproj_threshold,
                (self.orig.shape[1], self.orig.shape[0]),
            )
        except Exception as exc:
            self.horizon_result = None
            self.horizon_segment = None
            self.status_note = f"Horizon failed: {exc}"
            self.status_warn = True
            print(f"  [{self.cam_name}/{self.board_name}] Horizon failed: {exc}")
            return

        self.horizon_result = result
        self.horizon_segment = segment
        self.status_note = (
            f"Horizon computed  inliers={result['inlier_count']}/{result['point_count']}  "
            f"RMS={result['reproj_rms_px']:.2f}px"
        )
        self.status_warn = False
        print(
            f"  [{self.cam_name}/{self.board_name}] Horizon computed: "
            f"inliers={result['inlier_count']}/{result['point_count']}  "
            f"RMS={result['reproj_rms_px']:.2f}px  "
            f"line={[round(v, 6) for v in result['line']]}"
        )

    def _current_board_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        image_pts: list[list[float]] = []
        world_pts: list[list[float]] = []
        sq_len = float(self.cfg["square_length"])
        for cid, (x, y) in sorted(self.marked.items()):
            lx, ly = charuco_corner_local_xy(int(cid), self.sx, sq_len)
            image_pts.append([float(x), float(y)])
            world_pts.append([lx, ly])
        image_arr = np.asarray(image_pts, dtype=np.float32)
        bev_arr = world_to_bev_points(np.asarray(world_pts, dtype=np.float32), self.scale_px_per_meter)
        return image_arr, bev_arr

    def _update_manual_horizon_from_points(self) -> None:
        self.refined_horizon_segment = None
        self.refine_preview_result = None
        if len(self.manual_horizon_points) < 2:
            # Keep the previous line until two new points exist. Right-click deletes the YAML entry.
            self.status_note = "Manual horizon: click the second point"
            self.status_warn = False
            return
        try:
            frame_wh = (self.orig.shape[1], self.orig.shape[0])
            line = horizon_line_from_points(
                self.manual_horizon_points[0],
                self.manual_horizon_points[1],
                frame_wh,
            )
            self.manual_horizon_result = {
                "line": [float(v) for v in line.tolist()],
                "points": [[float(v) for v in p] for p in self.manual_horizon_points[:2]],
                "ground_side_positive": True,
            }
            self.manual_horizon_segment = horizon_line_to_segment(line, frame_wh)
            self.manual_horizon_dirty = True
            self.status_note = "Manual horizon stored. Shift+P previews refinement."
            self.status_warn = False
        except Exception as exc:
            self.manual_horizon_result = None
            self.manual_horizon_segment = None
            self.status_note = f"Manual horizon is invalid: {exc}"
            self.status_warn = True

    def _clear_manual_horizon(self) -> None:
        self.manual_horizon_points = []
        self.manual_horizon_result = None
        self.manual_horizon_segment = None
        self.refined_horizon_segment = None
        self.refine_preview_result = None
        self.manual_horizon_dirty = True
        self.status_note = "Manual horizon cleared"
        self.status_warn = False

    def _preview_refine(self) -> None:
        if len(self.marked) < 4:
            self.status_note = "Need at least 4 corners to preview refinement"
            self.status_warn = True
            return
        try:
            image_pts, bev_pts = self._current_board_arrays()
            h0, inliers = estimate_image_to_bev_homography(
                image_pts,
                bev_pts,
                self.ransac_reproj_threshold,
            )
            mask = inliers.ravel().astype(bool) if inliers is not None else np.ones(len(image_pts), dtype=bool)
            frame_wh = (self.orig.shape[1], self.orig.shape[0])
            manual_line = (
                self.manual_horizon_result["line"]
                if self.manual_horizon_result and self.manual_horizon_result.get("line") is not None
                else None
            )
            h1, diag = refine_homography(
                image_pts[mask],
                bev_pts[mask],
                h0,
                frame_wh=frame_wh,
                manual_horizon_line=manual_line,
                lambda_horizon=self.preview_lambda_horizon,
                max_iters=self.preview_max_iters,
            )
            self.refined_horizon_segment = horizon_line_to_segment(
                homography_horizon_line(h1, frame_wh),
                frame_wh,
            )
            self.refine_preview_result = diag
            note = (
                f"Refinement preview RMS {diag['reproj_rms_before_px']:.2f} -> "
                f"{diag['reproj_rms_after_px']:.2f}px"
            )
            if "horizon_delta_before_px" in diag:
                note += (
                    f" | horizon {diag['horizon_delta_before_px']:.1f} -> "
                    f"{diag['horizon_delta_after_px']:.1f}px"
                )
            self.status_note = note
            self.status_warn = False
            print(f"  [{self.cam_name}/{self.board_name}] {note}")
        except Exception as exc:
            self.refined_horizon_segment = None
            self.refine_preview_result = None
            self.status_note = f"Refinement preview failed: {exc}"
            self.status_warn = True
            print(f"  [{self.cam_name}/{self.board_name}] Refinement preview failed: {exc}")
    # ── main loop ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        win = f"ChArUco corners - {self.cam_name} / {self.board_name}"
        cv.namedWindow(win, cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback(win, self.mouse_cb)

        while True:
            cv.imshow(win, self.render())
            key = cv.waitKey(30) & 0xFF

            if key in (13, 32):  # Enter / space
                if len(self.marked) >= 4:
                    self.confirmed = True
                    break
            elif key in (ord('n'), ord('N'), 9):  # N / Tab
                self.selected_id = (self.selected_id + 1) % self.total
            elif key == ord('p'):
                self.selected_id = (self.selected_id - 1) % self.total
            elif key == ord('P'):
                self._preview_refine()
            elif key in (8, ord('z'), ord('Z')):  # Backspace / Z
                if self.selected_id in self.marked:
                    del self.marked[self.selected_id]
                    self.auto_ids.discard(self.selected_id)
                    self._invalidate_horizon()
            elif key in (ord('r'), ord('R')):
                self.marked.clear()
                self.auto_ids.clear()
                self.selected_id = 0
                self._invalidate_horizon()
            elif key in (ord('+'), ord('=')):
                self._zoom_at(self.disp_w / 2, self.disp_h / 2, self.ZOOM_STEP)
            elif key in (ord('-'), ord('_')):
                self._zoom_at(self.disp_w / 2, self.disp_h / 2, 1.0 / self.ZOOM_STEP)
            elif key == ord('0'):
                self._reset_view()
            elif key in (81, 2424832):  # left arrow, including some OpenCV codes
                self._pan_by(-40.0, 0.0)
            elif key in (83, 2555904):  # right
                self._pan_by(40.0, 0.0)
            elif key in (82, 2490368):  # up
                self._pan_by(0.0, -40.0)
            elif key in (84, 2621440):  # down
                self._pan_by(0.0, 40.0)
            elif key in (ord('v'), ord('V')):
                self.horizon_line_mode = not self.horizon_line_mode
                self.status_note = (
                    "manual horizon: left-click two points, right-click clears"
                    if self.horizon_line_mode else "exited manual-horizon mode"
                )
                self.status_warn = False
            elif key in (ord('h'), ord('H')):
                self._compute_homography_horizon()
            elif key in (ord('s'), ord('S')):
                self.skipped = True
                break
            elif key in (27, ord('q'), ord('Q')):
                self.quit = True
                break

        cv.destroyWindow(win)


# ──────────────────────────────────────────────────────────────────────────────
# Load stored manual corners and automatic detections.
# ──────────────────────────────────────────────────────────────────────────────

def load_existing_manual_corners(
    cam_cfg: dict[str, Any],
) -> dict[str, dict[int, list[float]]]:
    """Read manual_charuco_corners from one camera."""
    result: dict[str, dict[int, list[float]]] = {}
    for bname, entries in (cam_cfg.get("manual_charuco_corners") or {}).items():
        if entries:
            result[str(bname)] = {int(e[0]): [float(e[1]), float(e[2])] for e in entries}
    return result


def try_auto_detect(
    image: np.ndarray,
    board_cfg: dict[str, Any],
) -> dict[int, list[float]]:
    """Try automatic ChArUco corners as a prefill.

    Return an empty dict when detection fails.
    """
    try:
        from pipeline.multi_camera_bev_stitch import _detect_charuco_corners_raw
        corners, ids = _detect_charuco_corners_raw(image, board_cfg)
        if corners is None or ids is None:
            return {}
        return {int(cid): [float(c[0]), float(c[1])] for cid, c in zip(ids, corners)}
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    config_path = resolve_project_path(args.config)
    out_path = resolve_project_path(args.out) if args.out else config_path

    yaml_text = config_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(yaml_text)
    scale_px_per_meter = float(cfg.get("scale_px_per_meter", 100.0))
    ransac_reproj_threshold = float(cfg.get("ransac_reproj_threshold_px", 4.0))

    cameras: list[dict[str, Any]] = cfg.get("cameras", [])
    if not cameras:
        sys.exit("The YAML has no cameras")

    if args.cameras:
        wanted = set(args.cameras)
        cameras = [c for c in cameras if c.get("name") in wanted]
        if not cameras:
            sys.exit(f"Cameras not found: {args.cameras}")

    boards_cfg: dict[str, dict[str, Any]] = cfg.get("charuco_boards", {})
    if not boards_cfg:
        sys.exit("The YAML has no charuco_boards")

    if args.preview_refine:
        run_bfs_joint_refine_preview(
            str(config_path),
            scene_id=args.scene,
            diagnostics_dir=args.refine_diagnostics_dir,
            enable_horizon_refine=args.enable_horizon_refine,
            lambda_horizon=args.refine_lambda_horizon,
            max_iters=args.refine_max_iters,
        )
        return
    print(
        "\n"
        "══════════════════════════════════════════════════════\n"
        "  ChArUco corner marker\n"
        "──────────────────────────────────────────────────────\n"
        "  left=place  right=delete  middle-drag=pan  wheel=zoom\n"
        "  N/p=change corner  +/-=zoom  arrows=pan  0=reset view\n"
        "  V=manual horizon  Shift+P=preview refinement  H=homography horizon\n"
        "  Enter=confirm (>=4)  S=skip  Esc/Q=quit\n"
        "  click the diagram to select a corner\n"
        "══════════════════════════════════════════════════════\n"
    )
    warn_if_no_pil("corner marker")

    do_quit = False

    for cam_cfg in cameras:
        if do_quit:
            break

        name = cam_cfg.get("name", "?")
        img_path = resolve_project_path(cam_cfg.get("image", ""))

        if not img_path.is_file():
            print(f"  [{name}] image is missing: {img_path}, skipped")
            continue

        image = cv.imread(str(img_path), cv.IMREAD_COLOR)
        if image is None:
            print(f"  [{name}] cannot read the image, skipped")
            continue

        existing_by_board = load_existing_manual_corners(cam_cfg)
        existing_manual_horizon = load_manual_horizon_line(
            cam_cfg,
            (image.shape[1], image.shape[0]),
        )

        cam_boards: list[str] = cam_cfg.get("charuco_boards", [])
        if not cam_boards:
            print(f"  [{name}] has no charuco_boards, skipped")
            continue

        if args.boards:
            cam_boards = [b for b in cam_boards if b in set(args.boards)]
            if not cam_boards:
                print(f"  [{name}] no boards left after the filter, skipped")
                continue

        for board_name in cam_boards:
            if do_quit:
                break
            if board_name not in boards_cfg:
                print(f"  [{name}] board '{board_name}' is not in charuco_boards, skipped")
                continue

            board_cfg_entry = boards_cfg[board_name]
            existing = existing_by_board.get(board_name)

            # Automatic prefill only when this board has no manual corners yet.
            auto_detected: dict[int, list[float]] = {}
            if not existing and not args.no_auto_init:
                print(f"  [{name}] board '{board_name}' trying automatic corners ...", end=" ", flush=True)
                auto_detected = try_auto_detect(image, board_cfg_entry)
                n_auto = len(auto_detected)
                total = (int(board_cfg_entry["squares_x"]) - 1) * (int(board_cfg_entry["squares_y"]) - 1)
                status = f"prefilled {n_auto}/{total} corners"
                print(status if n_auto > 0 else "none detected; starting empty")

            n_exist = len(existing) if existing else 0
            print(f"  [{name}] opening board '{board_name}' marker window"
                  f" (existing manual corners: {n_exist})")

            marker = CornerMarker(
                image, name, board_name, board_cfg_entry,
                existing_corners=existing,
                auto_detected=auto_detected,
                scale_px_per_meter=scale_px_per_meter,
                ransac_reproj_threshold=ransac_reproj_threshold,
                existing_manual_horizon=existing_manual_horizon,
                preview_lambda_horizon=args.refine_lambda_horizon,
                preview_max_iters=args.refine_max_iters,
            )
            marker.run()

            if marker.quit:
                print(f"\n  Quit. Confirmed cameras were saved.")
                do_quit = True
                break
            elif marker.skipped:
                print(f"  [{name}/{board_name}] skipped (existing marks kept)")
                continue
            elif marker.confirmed:
                orig = marker.to_orig()
                n = len(orig)
                sample = dict(list(orig.items())[:4])
                print(f"  [{name}/{board_name}] confirmed {n} corners: {sample}"
                      f"{'…' if n > 4 else ''}")
                yaml_text = update_manual_corners_in_yaml_text(
                    yaml_text, name, board_name, orig
                )
                if marker.manual_horizon_dirty:
                    yaml_text = update_manual_horizon_line_in_yaml_text(
                        yaml_text, name, marker.manual_horizon_result
                    )
                    existing_manual_horizon = marker.manual_horizon_result
                    action = "written" if marker.manual_horizon_result is not None else "deleted"
                    print(f"  [{name}] manual_horizon_line {action}")
                if marker.horizon_result is not None:
                    yaml_text = update_horizon_in_yaml_text(
                        yaml_text, name, board_name, marker.horizon_result
                    )
                    print(f"  [{name}/{board_name}] horizon written to manual_homography_horizon")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml_text, encoding="utf-8")

    if out_path == config_path:
        print(f"\n  updated {config_path}")
    else:
        print(f"\n  saved {out_path} (original file unchanged)")
    print("  Next: python src/pipeline/multi_camera_bev_stitch.py --config ... --save-cal\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mark ChArUco inner corners and write the YAML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
----
  # Every camera and board
  py tools/mark_charuco_corners.py --config config/my_scene.yaml

  # Only cameras A and B
  py tools/mark_charuco_corners.py --config config/my_scene.yaml --cameras A B

  # Only the named boards
  py tools/mark_charuco_corners.py --config config/my_scene.yaml --boards AB_overlap

  # Write a new file and keep the original
  py tools/mark_charuco_corners.py --config config/my_scene.yaml --out config/my_scene_v2.yaml

  # No automatic prefill
  py tools/mark_charuco_corners.py --config config/my_scene.yaml --no-auto-init

  # Homography-refinement dry run only
  py tools/mark_charuco_corners.py --config config/my_scene.yaml --preview-refine --enable-horizon-refine
        """,
    )
    p.add_argument("--config", required=True, help="Scene YAML path")
    p.add_argument(
        "--cameras", nargs="+", metavar="NAME",
        help="Only these cameras. Default is every camera.",
    )
    p.add_argument(
        "--boards", nargs="+", metavar="NAME",
        help="Only these boards. Default is every board.",
    )
    p.add_argument(
        "--out", default=None,
        help="Output YAML. Default overwrites --config.",
    )
    p.add_argument(
        "--no-auto-init", action="store_true",
        help="Do not prefill from automatic detection.",
    )
    p.add_argument(
        "--preview-refine", action="store_true",
        help="Skip the window. Run a homography-refinement dry run and write diagnostics.",
    )
    p.add_argument(
        "--scene", default="parking",
        help="Scene id for the dry run. Default is parking.",
    )
    p.add_argument(
        "--enable-horizon-refine", action="store_true",
        help="Use the manual horizon as a soft constraint in the refinement preview.",
    )
    p.add_argument(
        "--refine-lambda-horizon", type=float, default=1.0,
        help="Manual-horizon residual weight. Default 1.0.",
    )
    p.add_argument(
        "--refine-max-iters", type=int, default=30,
        help="Maximum LM iterations for homography refinement. Default 30.",
    )
    p.add_argument(
        "--refine-diagnostics-dir", default=None,
        help=f"Dry-run output directory (default {CALIBRATION_DIAGNOSTICS_DIR})",
    )
    return p

if __name__ == "__main__":
    run(build_parser().parse_args())
