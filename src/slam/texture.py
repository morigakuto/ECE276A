"""Texture mapping utilities for Part 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .mapping import OccupancyGridMap


def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return R_z @ R_y @ R_x


@dataclass(frozen=True)
class CameraExtrinsics:
    translation: Tuple[float, float, float]
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def rotation_matrix(self) -> np.ndarray:
        return _rotation_from_rpy(self.roll, self.pitch, self.yaw)

    def translation_vector(self) -> np.ndarray:
        return np.array(self.translation, dtype=float)


class TextureMapper:
    """Projects Kinect RGB-D measurements onto the occupancy grid plane."""

    def __init__(
        self,
        grid: OccupancyGridMap,
        depth_camera_pose: Optional[CameraExtrinsics] = None,
        depth_intrinsics: Optional[np.ndarray] = None,
        floor_height: float = 0.0,
        height_tolerance: float = 0.1,
    ) -> None:
        self.grid = grid
        self.depth_pose = depth_camera_pose or CameraExtrinsics(
            translation=(0.18, 0.005, 0.36),
            roll=0.0,
            pitch=0.36,
            yaw=0.021,
        )
        if depth_intrinsics is None:
            depth_intrinsics = np.array(
                [
                    [585.05, 0.0, 242.94],
                    [0.0, 585.05, 315.84],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        self.depth_intrinsics = depth_intrinsics
        self.floor_height = float(floor_height)
        self.height_tolerance = float(height_tolerance)

        self.depth_rotation = self.depth_pose.rotation_matrix()
        self.depth_translation = self.depth_pose.translation_vector()

        self.color_sum = np.zeros((grid.size_y, grid.size_x, 3), dtype=np.float64)
        self.hit_counts = np.zeros((grid.size_y, grid.size_x), dtype=np.uint32)

    def integrate_frame(
        self,
        robot_pose: np.ndarray,
        disparity_image: np.ndarray,
        rgb_image: np.ndarray,
        *,
        min_depth: float = 0.2,
        max_depth: float = 5.0,
        max_points: Optional[int] = None,
    ) -> int:
        """Fuse a single RGB-D frame into the texture map."""
        disp = np.asarray(disparity_image)
        if disp.ndim != 2:
            raise ValueError("disparity_image must be grayscale (H, W).")
        disp = disp.astype(np.float32)

        rgb = np.asarray(rgb_image)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError("rgb_image must have shape (H, W, 3).")
        rgb = self._ensure_rgb_float(rgb)

        valid_mask = disp > 0.0
        if not np.any(valid_mask):
            return 0

        rows, cols = np.indices(disp.shape)
        rows = rows[valid_mask].astype(np.float32)
        cols = cols[valid_mask].astype(np.float32)
        disparities = disp[valid_mask]

        dd = -0.00304 * disparities + 3.31
        depth = 1.03 / dd

        depth_mask = (depth > min_depth) & (depth < max_depth)
        if not np.any(depth_mask):
            return 0

        rows = rows[depth_mask]
        cols = cols[depth_mask]
        depth = depth[depth_mask]
        dd = dd[depth_mask]

        fx = float(self.depth_intrinsics[0, 0])
        fy = float(self.depth_intrinsics[1, 1])
        cx = float(self.depth_intrinsics[0, 2])
        cy = float(self.depth_intrinsics[1, 2])

        x = (cols - cx) * depth / fx
        y = (rows - cy) * depth / fy
        z = depth
        points_depth = np.stack((x, y, z), axis=1)

        rgbi = (526.37 * cols + 19276.0 - 7877.07 * dd) / 585.051
        rgbj = (526.37 * rows + 16662.0) / 585.051
        rgbi = np.round(rgbi).astype(int)
        rgbj = np.round(rgbj).astype(int)

        rgb_h, rgb_w = rgb.shape[:2]
        color_mask = (
            (rgbi >= 0)
            & (rgbi < rgb_w)
            & (rgbj >= 0)
            & (rgbj < rgb_h)
        )
        if not np.any(color_mask):
            return 0

        points_depth = points_depth[color_mask]
        rgbi = rgbi[color_mask]
        rgbj = rgbj[color_mask]
        colors = rgb[rgbj, rgbi, :3]

        if max_points and points_depth.shape[0] > max_points:
            step = int(np.ceil(points_depth.shape[0] / max_points))
            points_depth = points_depth[::step]
            colors = colors[::step]

        points_body = (self.depth_rotation @ points_depth.T).T + self.depth_translation

        pose = np.asarray(robot_pose, dtype=float)
        theta = pose[2]
        R_world_body = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        t_world_body = np.array([pose[0], pose[1], 0.0], dtype=float)

        points_world = (R_world_body @ points_body.T).T + t_world_body
        if points_world.shape[0] == 0:
            return 0
        z_world = points_world[:, 2]
        z_min = np.min(z_world)
        height_mask = z_world < (z_min + self.height_tolerance)
        if not np.any(height_mask):
            return 0

        points_world = points_world[height_mask]
        colors = colors[height_mask]

        cells = self.grid.world_to_grid(points_world[:, :2])
        valid_cells = self.grid.cells_in_bounds(cells)
        if not np.any(valid_cells):
            return 0

        cells = cells[valid_cells]
        colors = colors[valid_cells]

        rows = cells[:, 0]
        cols = cells[:, 1]
        for channel in range(3):
            np.add.at(self.color_sum[:, :, channel], (rows, cols), colors[:, channel])
        np.add.at(self.hit_counts, (rows, cols), 1)
        return cells.shape[0]

    def build_texture_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the averaged RGB map and a mask of observed cells."""
        texture = np.zeros_like(self.color_sum, dtype=np.uint8)
        valid = self.hit_counts > 0
        if np.any(valid):
            averages = self.color_sum[valid] / self.hit_counts[valid, None]
            texture[valid] = np.clip(averages, 0.0, 255.0).astype(np.uint8)
        return texture, valid

    @staticmethod
    def _ensure_rgb_float(rgb: np.ndarray) -> np.ndarray:
        arr = rgb.astype(np.float32, copy=False)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        return np.clip(arr, 0.0, 255.0)
