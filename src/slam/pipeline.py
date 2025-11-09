"""Shared helpers for the SLAM experiments."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .icp import ICPResult, icp_2d
from .odometry import Odometry

METERS_PER_TICK = 0.0022


def compute_odometry(
    encoder_counts: np.ndarray,
    encoder_stamps: np.ndarray,
    imu_ang_vel: np.ndarray,
    imu_stamps: np.ndarray,
) -> np.ndarray:
    odom = Odometry(initial_pose=(0.0, 0.0, 0.0))
    num_steps = encoder_stamps.shape[0]
    poses = np.zeros((num_steps, 3))
    poses[0] = odom.pose

    for idx in range(num_steps - 1):
        dt = encoder_stamps[idx + 1] - encoder_stamps[idx]
        if dt <= 0:
            poses[idx + 1] = poses[idx]
            continue

        FR, FL, RR, RL = encoder_counts[:, idx + 1]
        right_distance = ((FR + RR) * 0.5) * METERS_PER_TICK
        left_distance = ((FL + RL) * 0.5) * METERS_PER_TICK
        distance = 0.5 * (right_distance + left_distance)

        imu_idx = np.argmin(np.abs(imu_stamps - encoder_stamps[idx + 1]))
        omega = imu_ang_vel[2, imu_idx]

        delta_pose = (distance, 0.0, omega * dt)
        odom.update(delta_pose)
        poses[idx + 1] = odom.pose

    return poses


def build_lidar_scans(lidar_data: dict, range_min: float | None = None, range_max: float | None = None) -> List[np.ndarray]:
    ranges = lidar_data["ranges"]
    angle_min = float(np.squeeze(lidar_data["angle_min"]))
    angle_increment = float(np.squeeze(lidar_data["angle_increment"]))
    range_min = float(np.squeeze(lidar_data["range_min"])) if range_min is None else range_min
    range_max = float(np.squeeze(lidar_data["range_max"])) if range_max is None else range_max

    num_beams = ranges.shape[1]
    beam_angles = angle_min + np.arange(num_beams) * angle_increment
    cos_angles = np.cos(beam_angles)
    sin_angles = np.sin(beam_angles)

    scans: List[np.ndarray] = []
    for scan in ranges:
        mask = (scan > range_min) & (scan < range_max)
        points = np.column_stack((scan[mask] * cos_angles[mask], scan[mask] * sin_angles[mask]))
        scans.append(points.astype(np.float32))

    return scans


def sample_poses_at_stamps(
    source_stamps: np.ndarray,
    source_poses: np.ndarray,
    query_stamps: np.ndarray,
) -> np.ndarray:
    source_stamps = np.asarray(source_stamps)
    source_poses = np.asarray(source_poses)
    query_stamps = np.asarray(query_stamps)

    sampled = np.zeros((query_stamps.shape[0], 3), dtype=float)
    for i, ts in enumerate(query_stamps):
        idx = np.searchsorted(source_stamps, ts, side="left")
        if idx == 0:
            sampled[i] = source_poses[0]
        elif idx >= len(source_stamps):
            sampled[i] = source_poses[-1]
        else:
            prev = idx - 1
            if abs(ts - source_stamps[prev]) <= abs(source_stamps[idx] - ts):
                sampled[i] = source_poses[prev]
            else:
                sampled[i] = source_poses[idx]
    return sampled


def run_sequential_icp(
    lidar_scans: Sequence[np.ndarray],
    initial_poses: np.ndarray,
    *,
    max_iterations: int = 60,
    tolerance: float = 1e-4,
    max_correspondence_distance: float = 1.0,
) -> Tuple[np.ndarray, List[Optional[ICPResult]]]:
    refined = np.zeros_like(initial_poses)
    refined[0] = initial_poses[0]
    icp_results: List[Optional[ICPResult]] = [None]

    for idx in range(1, len(lidar_scans)):
        reference_scan = lidar_scans[idx - 1]
        moving_scan = lidar_scans[idx]
        if reference_scan.size == 0 or moving_scan.size == 0:
            refined[idx] = refined[idx - 1]
            icp_results.append(icp_results[-1])
            continue

        initial_relative = current_to_previous_transform(refined[idx - 1], initial_poses[idx])

        icp_result = icp_2d(
            reference_scan,
            moving_scan,
            initial_pose=initial_relative,
            max_iterations=max_iterations,
            tolerance=tolerance,
            max_correspondence_distance=max_correspondence_distance,
        )
        icp_results.append(icp_result)

        if np.isfinite(icp_result.rmse) and icp_result.converged:
            transform_prev_from_curr = icp_result.as_homogeneous_matrix()
        else:
            transform_prev_from_curr = delta_to_matrix(initial_relative)

        T_prev_world = pose_to_matrix(refined[idx - 1])
        T_curr_world = T_prev_world @ transform_prev_from_curr
        refined[idx] = matrix_to_pose(T_curr_world)

    return refined, icp_results


def current_to_previous_transform(prev_pose: np.ndarray, curr_pose: np.ndarray) -> Tuple[float, float, float]:
    T_prev = pose_to_matrix(prev_pose)
    T_curr = pose_to_matrix(curr_pose)
    T_prev_from_curr = np.linalg.inv(T_prev) @ T_curr
    theta = np.arctan2(T_prev_from_curr[1, 0], T_prev_from_curr[0, 0])
    translation = T_prev_from_curr[:2, 2]
    return translation[0], translation[1], theta


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def matrix_to_pose(T: np.ndarray) -> np.ndarray:
    theta = np.arctan2(T[1, 0], T[0, 0])
    x, y = T[0, 2], T[1, 2]
    return np.array([x, y, wrap_angle(theta)])


def delta_to_matrix(delta: Tuple[float, float, float]) -> np.ndarray:
    tx, ty, theta = delta
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]])


def apply_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    return (rotation @ points.T).T + translation


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi
