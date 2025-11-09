"""Occupancy grid mapping utilities for Part 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


@dataclass(frozen=True)
class SensorPose2D:
    """Pose of a 2-D sensor expressed in the robot body frame."""

    x: float
    y: float
    yaw: float = 0.0

    def as_translation(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


class OccupancyGridMap:
    """Maintains a log-odds occupancy grid in the robot world frame."""

    def __init__(
        self,
        resolution: float = 0.05,
        x_limits: Tuple[float, float] = (-100.0, 100.0),
        y_limits: Tuple[float, float] = (-100.0, 100.0),
        log_odds_occ: float = 0.85,
        log_odds_free: float = -0.4,
        log_odds_min: float = -10.0,
        log_odds_max: float = 10.0,
        sensor_pose: Optional[SensorPose2D] = None,
    ) -> None:
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.resolution = float(resolution)
        self.x_limits = np.array(x_limits, dtype=float)
        self.y_limits = np.array(y_limits, dtype=float)
        self.size_x = int(np.ceil((self.x_limits[1] - self.x_limits[0]) / self.resolution))
        self.size_y = int(np.ceil((self.y_limits[1] - self.y_limits[0]) / self.resolution))
        self.log_odds_occ = float(log_odds_occ)
        self.log_odds_free = float(log_odds_free)
        self.log_odds_min = float(log_odds_min)
        self.log_odds_max = float(log_odds_max)
        self.sensor_pose = sensor_pose or SensorPose2D(0.29833, 0.0, 0.0)

        self.log_odds = np.zeros((self.size_y, self.size_x), dtype=np.float32)
        self.observed_mask = np.zeros_like(self.log_odds, dtype=bool)

    def reset(self) -> None:
        self.log_odds.fill(0.0)
        self.observed_mask.fill(False)

    def update_from_points(
        self,
        robot_pose: np.ndarray,
        scan_points_sensor: np.ndarray,
        *,
        sensor_pose: Optional[SensorPose2D] = None,
        beam_stride: int = 1,
    ) -> int:
        """Update the map given LiDAR hit points expressed in the sensor frame."""
        if scan_points_sensor.size == 0:
            return 0
        if beam_stride < 1:
            raise ValueError("beam_stride must be >= 1")

        pose = np.asarray(robot_pose, dtype=float)
        points = np.asarray(scan_points_sensor, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("scan_points_sensor must have shape (N, 2)")

        sensor = sensor_pose or self.sensor_pose
        R_body_sensor = _rotation_matrix(sensor.yaw)
        sensor_translation_body = sensor.as_translation()

        if beam_stride > 1:
            points = points[::beam_stride]
        if points.size == 0:
            return 0

        # Transform points to the robot body frame and then to the world frame.
        points_body = (R_body_sensor @ points.T).T + sensor_translation_body
        sensor_origin_body = sensor_translation_body

        R_world_body = _rotation_matrix(pose[2])
        points_world = (R_world_body @ points_body.T).T + pose[:2]
        sensor_origin_world = R_world_body @ sensor_origin_body + pose[:2]

        origin_cell = self.world_to_grid(sensor_origin_world)
        if not self._cell_in_bounds(origin_cell):
            return 0

        updates = 0
        for point_world in points_world:
            cell = self.world_to_grid(point_world)
            if not self._cell_in_bounds(cell):
                continue

            ray_cells = self._bresenham(origin_cell, cell)
            if ray_cells.shape[0] == 0:
                continue

            free_cells = ray_cells[:-1]
            if free_cells.size:
                rows, cols = free_cells[:, 0], free_cells[:, 1]
                np.add.at(self.log_odds, (rows, cols), self.log_odds_free)
                self.observed_mask[rows, cols] = True

            end_row, end_col = cell
            self.log_odds[end_row, end_col] += self.log_odds_occ
            self.observed_mask[end_row, end_col] = True
            updates += 1

        np.clip(self.log_odds, self.log_odds_min, self.log_odds_max, out=self.log_odds)
        return updates

    def world_to_grid(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        cols = np.floor((xy[..., 0] - self.x_limits[0]) / self.resolution).astype(int)
        rows = np.floor((xy[..., 1] - self.y_limits[0]) / self.resolution).astype(int)
        return np.stack((rows, cols), axis=-1)

    def grid_to_world(self, cells: np.ndarray) -> np.ndarray:
        cells = np.asarray(cells, dtype=float)
        rows = cells[..., 0]
        cols = cells[..., 1]
        x = (cols + 0.5) * self.resolution + self.x_limits[0]
        y = (rows + 0.5) * self.resolution + self.y_limits[0]
        return np.stack((x, y), axis=-1)

    def probability(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def cells_in_bounds(self, cells: np.ndarray) -> np.ndarray:
        cells = np.asarray(cells, dtype=int)
        rows, cols = cells[:, 0], cells[:, 1]
        return (
            (rows >= 0)
            & (rows < self.size_y)
            & (cols >= 0)
            & (cols < self.size_x)
        )

    def _cell_in_bounds(self, cell: np.ndarray) -> bool:
        row, col = int(cell[0]), int(cell[1])
        return (
            0 <= row < self.size_y
            and 0 <= col < self.size_x
        )

    @staticmethod
    def _bresenham(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        x0, y0 = int(start[1]), int(start[0])
        x1, y1 = int(end[1]), int(end[0])

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        cells = []
        while True:
            cells.append((y, x))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return np.array(cells, dtype=int)
