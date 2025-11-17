"""Pose graph optimization utilities for Part 4 (loop closures with GTSAM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

try:
    import gtsam
except ImportError as exc:  # pragma: no cover - gtsam should be installed per pyproject
    raise ImportError(
        "PoseGraphOptimizer requires the GTSAM Python bindings. "
        "Install them with `pip install gtsam` and re-run the experiment."
    ) from exc

from .icp import ICPResult, icp_2d


def _array_to_pose2(pose: np.ndarray) -> gtsam.Pose2:
    x, y, theta = pose
    return gtsam.Pose2(float(x), float(y), float(theta))


def _pose2_to_array(pose: gtsam.Pose2) -> np.ndarray:
    return np.array([pose.x(), pose.y(), pose.theta()], dtype=float)


def _pose2_from_icp(result: ICPResult) -> gtsam.Pose2:
    heading = float(np.arctan2(result.rotation[1, 0], result.rotation[0, 0]))
    tx = float(result.translation[0])
    ty = float(result.translation[1])
    return gtsam.Pose2(tx, ty, heading)


@dataclass(frozen=True)
class LoopClosure:
    """Metadata describing an accepted loop-closure edge."""

    from_idx: int
    to_idx: int
    rmse: float
    iterations: int
    kind: str


@dataclass(frozen=True)
class OptimizationSummary:
    """Container for the optimized trajectory and diagnostics."""

    optimized_poses: np.ndarray
    initial_error: float
    final_error: float
    iterations: int
    loop_closures: Sequence[LoopClosure]
    marginals: Optional[gtsam.Marginals]


class PoseGraphOptimizer:
    """Builds and solves a Pose2 graph with odometry + loop-closure constraints."""

    def __init__(
        self,
        initial_poses: np.ndarray,
        lidar_scans: Sequence[np.ndarray],
        icp_results: Sequence[Optional[ICPResult]],
        *,
        prior_sigma: Tuple[float, float, float] = (0.2, 0.2, np.deg2rad(5.0)),
        odometry_sigma: Tuple[float, float, float] = (0.05, 0.05, np.deg2rad(2.0)),
        loop_sigma: Tuple[float, float, float] = (0.03, 0.03, np.deg2rad(1.0)),
        loop_rmse_threshold: float = 0.2,
        loop_translation_threshold: float = 5.0,
        loop_max_correspondence: float = 1.5,
        loop_icp_max_iterations: int = 80,
        use_icp_accepted_flag: bool = True,
    ) -> None:
        poses = np.asarray(initial_poses, dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError("initial_poses must have shape (N, 3)")
        if len(lidar_scans) != poses.shape[0]:
            raise ValueError("lidar_scans length must match number of poses")
        if len(icp_results) != poses.shape[0]:
            raise ValueError("icp_results length must match number of poses")

        self.initial_poses = poses
        self.lidar_scans = lidar_scans
        self.icp_results = icp_results
        self.num_nodes = poses.shape[0]
        self.use_icp_accepted_flag = use_icp_accepted_flag

        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.keys = [gtsam.symbol("x", idx) for idx in range(self.num_nodes)]

        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(prior_sigma, dtype=float))
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(odometry_sigma, dtype=float))
        self.loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(loop_sigma, dtype=float))

        self.loop_rmse_threshold = float(loop_rmse_threshold)
        self.loop_translation_threshold = float(loop_translation_threshold)
        self.loop_max_correspondence = float(loop_max_correspondence)
        self.loop_icp_max_iterations = int(loop_icp_max_iterations)
        self.loop_closures: List[LoopClosure] = []
        self._loop_edges: set[Tuple[int, int]] = set()

        for idx, pose in enumerate(self.initial_poses):
            self.values.insert(self.keys[idx], _array_to_pose2(pose))

        self._add_prior()
        self._add_odometry_chain()

    def _add_prior(self) -> None:
        self.graph.add(
            gtsam.PriorFactorPose2(
                self.keys[0],
                self.values.atPose2(self.keys[0]),
                self.prior_noise,
            )
        )

    def _add_odometry_chain(self) -> None:
        for idx in range(1, self.num_nodes):
            measurement = self._odometry_measurement(idx)
            self.graph.add(
                gtsam.BetweenFactorPose2(
                    self.keys[idx - 1],
                    self.keys[idx],
                    measurement,
                    self.odometry_noise,
                )
            )

    def _odometry_measurement(self, idx: int) -> gtsam.Pose2:
        icp_result = self.icp_results[idx] if idx < len(self.icp_results) else None

        use_icp = False
        if isinstance(icp_result, ICPResult):
            if self.use_icp_accepted_flag:
                use_icp = icp_result.accepted
            else:
                use_icp = icp_result.converged and np.isfinite(icp_result.rmse)

        if use_icp:
            return _pose2_from_icp(icp_result)
        else:
            prev_pose = _array_to_pose2(self.initial_poses[idx - 1])
            curr_pose = _array_to_pose2(self.initial_poses[idx])
        return prev_pose.between(curr_pose)

    def add_fixed_interval_loops(self, interval: int = 10, max_pairs: Optional[int] = None) -> int:
        if interval < 1:
            raise ValueError("interval must be >= 1")
        added = 0
        for to_idx in range(interval, self.num_nodes):
            from_idx = to_idx - interval
            if self._try_add_loop(from_idx, to_idx, kind=f"interval-{interval}"):
                added += 1
                if max_pairs is not None and added >= max_pairs:
                    break
        return added

    def detect_proximity_loops(
        self,
        radius: float = 0.8,
        min_separation: int = 25,
        max_loops: Optional[int] = 50,
        max_neighbors: int = 5,
    ) -> int:
        """Search for pose pairs that are spatially close and try to close loops."""
        if radius <= 0:
            raise ValueError("radius must be positive")
        if min_separation < 1:
            raise ValueError("min_separation must be >= 1")
        positions = self.initial_poses[:, :2]
        tree = cKDTree(positions)

        added = 0
        for idx in range(self.num_nodes):
            neighbors = tree.query_ball_point(positions[idx], radius)
            candidates = sorted(
                n for n in neighbors if n > idx + min_separation
            )
            if not candidates:
                continue
            for neighbor in candidates[:max_neighbors]:
                if max_loops is not None and added >= max_loops:
                    return added
                if self._try_add_loop(idx, neighbor, kind="proximity"):
                    added += 1
        return added

    def _try_add_loop(self, from_idx: int, to_idx: int, kind: str) -> bool:
        edge_id = (min(from_idx, to_idx), max(from_idx, to_idx))
        if edge_id in self._loop_edges:
            return False
        measurement = self._estimate_loop_measurement(from_idx, to_idx)
        if measurement is None:
            return False
        pose_measurement, icp_result = measurement

        self.graph.add(
            gtsam.BetweenFactorPose2(
                self.keys[from_idx],
                self.keys[to_idx],
                pose_measurement,
                self.loop_noise,
            )
        )
        self.loop_closures.append(
            LoopClosure(
                from_idx=from_idx,
                to_idx=to_idx,
                rmse=float(icp_result.rmse),
                iterations=icp_result.iterations,
                kind=kind,
            )
        )
        self._loop_edges.add(edge_id)
        return True

    def _estimate_loop_measurement(
        self,
        from_idx: int,
        to_idx: int,
    ) -> Optional[Tuple[gtsam.Pose2, ICPResult]]:
        scan_ref = self.lidar_scans[from_idx]
        scan_cmp = self.lidar_scans[to_idx]
        if scan_ref.size == 0 or scan_cmp.size == 0:
            return None

        initial_guess = self._initial_relative_guess(from_idx, to_idx)
        icp_result = icp_2d(
            scan_ref,
            scan_cmp,
            initial_pose=initial_guess,
            max_iterations=self.loop_icp_max_iterations,
            tolerance=1e-4,
            max_correspondence_distance=self.loop_max_correspondence,
        )
        if not icp_result.converged or not np.isfinite(icp_result.rmse):
            return None
        if icp_result.rmse > self.loop_rmse_threshold:
            return None
        translation_norm = float(np.linalg.norm(icp_result.translation))
        if translation_norm > self.loop_translation_threshold:
            return None

        return _pose2_from_icp(icp_result), icp_result

    def _initial_relative_guess(self, from_idx: int, to_idx: int) -> Tuple[float, float, float]:
        pose_from = _array_to_pose2(self.initial_poses[from_idx])
        pose_to = _array_to_pose2(self.initial_poses[to_idx])
        delta = pose_from.between(pose_to)
        return float(delta.x()), float(delta.y()), float(delta.theta())

    def optimize(
        self,
        max_iterations: int = 100,
        damping: float = 1e-3,
        compute_marginals: bool = False,
    ) -> OptimizationSummary:
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(int(max_iterations))
        if hasattr(params, "setlambdaInitial"):
            params.setlambdaInitial(damping)
        elif hasattr(params, "setLambdaInitial"):
            params.setLambdaInitial(damping)

        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, params)
        initial_error = optimizer.error()
        result = optimizer.optimize()
        final_error = optimizer.error()
        iterations = optimizer.iterations()

        optimized = np.zeros_like(self.initial_poses)
        for idx, key in enumerate(self.keys):
            optimized[idx] = _pose2_to_array(result.atPose2(key))

        marginals = gtsam.Marginals(self.graph, result) if compute_marginals else None

        return OptimizationSummary(
            optimized_poses=optimized,
            initial_error=float(initial_error),
            final_error=float(final_error),
            iterations=int(iterations),
            loop_closures=tuple(self.loop_closures),
            marginals=marginals,
        )

    def loop_summary(self) -> str:
        """Return a short text summary of accepted loop closures."""
        if not self.loop_closures:
            return "No loop closures added."
        counts: dict[str, int] = {}
        for closure in self.loop_closures:
            counts[closure.kind] = counts.get(closure.kind, 0) + 1
        parts = [f"{kind}:{count}" for kind, count in counts.items()]
        return f"{len(self.loop_closures)} loops ({', '.join(parts)})"
