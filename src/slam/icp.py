from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class ICPResult:
    rotation: np.ndarray
    translation: np.ndarray
    iterations: int
    converged: bool
    rmse: float

    def as_homogeneous_matrix(self) -> np.ndarray:
        """Return the SE(d) homogeneous transform corresponding to the result."""
        dim = self.rotation.shape[0]
        T = np.eye(dim + 1, dtype=self.rotation.dtype)
        T[:dim, :dim] = self.rotation
        T[:dim, -1] = self.translation
        return T

    @property
    def heading(self) -> float:
        """Return the yaw angle for 2-D results."""
        if self.rotation.shape != (2, 2):
            raise AttributeError("heading is only defined for 2-D ICP results.")
        return float(np.arctan2(self.rotation[1, 0], self.rotation[0, 0]))


def run_icp(
    target_points: np.ndarray,
    source_points: np.ndarray,
    initial_rotation: Optional[np.ndarray] = None,
    initial_translation: Optional[np.ndarray] = None,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
    max_correspondence_distance: Optional[float] = None,
    min_correspondences: Optional[int] = None,
) -> ICPResult:
    """Align source_points to target_points using point-to-point ICP."""
    target = np.asarray(target_points, dtype=float)
    source = np.asarray(source_points, dtype=float)

    if target.ndim != 2 or source.ndim != 2:
        raise ValueError("target_points and source_points must be 2-D arrays.")
    if target.shape[1] != source.shape[1]:
        raise ValueError("target_points and source_points must have the same dimension.")

    dim = target.shape[1]
    if target.shape[0] < dim or source.shape[0] < dim:
        raise ValueError("Need at least as many points as the space dimension for ICP.")

    if min_correspondences is None:
        min_correspondences = dim + 1

    R = np.eye(dim) if initial_rotation is None else np.asarray(initial_rotation, dtype=float)
    t = np.zeros(dim) if initial_translation is None else np.asarray(initial_translation, dtype=float)

    transformed = _transform_points(source, R, t)
    tree = cKDTree(target)

    prev_rmse = np.inf
    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        (
            matched_target,
            matched_transformed,
            distances,
            valid_indices,
        ) = _match_correspondences(
            tree,
            target,
            transformed,
            max_correspondence_distance,
        )

        if len(valid_indices) < min_correspondences:
            break

        R_delta, t_delta = _estimate_rigid_transform(matched_transformed, matched_target)
        R = R_delta @ R
        t = R_delta @ t + t_delta

        transformed = _transform_points(source, R, t)
        rmse = _compute_rmse(distances)

        iterations = iteration
        if abs(prev_rmse - rmse) < tolerance:
            converged = True
            break
        prev_rmse = rmse

    # Recompute correspondences for an accurate final error metric.
    _, _, final_distances, _ = _match_correspondences(
        tree,
        target,
        transformed,
        max_correspondence_distance,
    )
    final_rmse = _compute_rmse(final_distances)

    return ICPResult(
        rotation=R,
        translation=t,
        iterations=iterations,
        converged=converged,
        rmse=final_rmse,
    )


def icp_2d(
    reference_scan: np.ndarray,
    moving_scan: np.ndarray,
    initial_pose: Optional[Tuple[float, float, float]] = None,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
    max_correspondence_distance: Optional[float] = 1.0,
) -> ICPResult:
    """Convenience wrapper for 2-D LiDAR scan matching."""
    if reference_scan.shape[1] != 2 or moving_scan.shape[1] != 2:
        raise ValueError("Both scans must contain 2-D points.")

    if initial_pose is None:
        initial_rotation = np.eye(2)
        initial_translation = np.zeros(2)
    else:
        tx, ty, theta = initial_pose
        initial_rotation = _rotation_matrix(theta)
        initial_translation = np.array([tx, ty], dtype=float)

    return run_icp(
        target_points=reference_scan,
        source_points=moving_scan,
        initial_rotation=initial_rotation,
        initial_translation=initial_translation,
        max_iterations=max_iterations,
        tolerance=tolerance,
        max_correspondence_distance=max_correspondence_distance,
        min_correspondences=5,
    )


def _match_correspondences(
    tree: cKDTree,
    target_points: np.ndarray,
    transformed_points: np.ndarray,
    max_correspondence_distance: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distances, indices = tree.query(transformed_points, k=1)
    if max_correspondence_distance is not None:
        mask = distances <= max_correspondence_distance
    else:
        mask = np.ones_like(distances, dtype=bool)

    valid_indices = np.nonzero(mask)[0]
    if valid_indices.size == 0:
        return (
            np.empty((0, target_points.shape[1])),
            np.empty((0, target_points.shape[1])),
            np.empty(0),
            valid_indices,
        )

    matched_target = target_points[indices[valid_indices]]
    matched_transformed = transformed_points[valid_indices]
    matched_distances = distances[valid_indices]

    return matched_target, matched_transformed, matched_distances, valid_indices


def _estimate_rigid_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if source_points.shape != target_points.shape:
        raise ValueError("source_points and target_points must have identical shapes.")
    if source_points.shape[0] == 0:
        raise ValueError("At least one correspondence is required to estimate a transform.")

    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    centered_source = source_points - source_centroid
    centered_target = target_points - target_centroid

    H = centered_source.T @ centered_target
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = target_centroid - R @ source_centroid
    return R, t


def _transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (rotation @ points.T).T + translation


def _compute_rmse(distances: np.ndarray) -> float:
    if distances.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean(distances**2)))


def _rotation_matrix(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)
