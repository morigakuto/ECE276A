"""Quick visualization utility: occupancy grid map with multiple trajectories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.slam.mapping import OccupancyGridMap
from src.slam.optimizer import PoseGraphOptimizer
from src.slam.pipeline import (
    build_lidar_scans,
    compute_odometry,
    run_sequential_icp,
    sample_poses_at_stamps,
)
from src.utils.data_loader import load_part1_data, load_part2_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render occupancy grid with Odometry/ICP/Pose-graph trajectories overlaid."
    )
    parser.add_argument("--dataset", type=int, default=20, help="Dataset number (20 or 21).")
    parser.add_argument("--map-resolution", type=float, default=0.05, help="Occupancy grid resolution [m].")
    parser.add_argument(
        "--map-extent",
        type=float,
        default=30.0,
        help="Half-length of the square occupancy grid in meters (map spans [-extent, extent]).",
    )
    parser.add_argument("--beam-stride", type=int, default=2, help="LiDAR down-sampling stride when mapping.")
    parser.add_argument("--loop-interval", type=int, default=10, help="Fixed interval for loop-closure seeds.")
    parser.add_argument("--max-interval-loops", type=int, default=80, help="Limit on interval-based loops.")
    parser.add_argument("--loop-radius", type=float, default=0.8, help="Radius for proximity-based loop search [m].")
    parser.add_argument("--min-loop-separation", type=int, default=25, help="Skip near-identical poses when looping.")
    parser.add_argument("--max-proximity-loops", type=int, default=50, help="Limit proximity-based loop closures.")
    parser.add_argument("--max-iterations", type=int, default=100, help="Pose graph optimizer iteration budget.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results" / "figures",
        help="Directory where the overlay figure will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Overlay] Loading dataset {args.dataset} ...")
    enc_counts, enc_stamps, imu_ang_vel, imu_stamps = load_part1_data(args.dataset)
    lidar_data = load_part2_data(args.dataset)

    print("[Overlay] Computing odometry and aligning LiDAR scans ...")
    odom_full = compute_odometry(enc_counts, enc_stamps, imu_ang_vel, imu_stamps)
    lidar_scans = build_lidar_scans(lidar_data)
    lidar_stamps = lidar_data["time_stamps"]
    odom_at_lidar = sample_poses_at_stamps(enc_stamps, odom_full, lidar_stamps)

    print("[Overlay] Running sequential ICP refinement ...")
    icp_traj, icp_results = run_sequential_icp(
        lidar_scans,
        odom_at_lidar,
        max_iterations=60,
        tolerance=1e-4,
        max_correspondence_distance=1.0,
    )

    print("[Overlay] Building and optimizing pose graph ...")
    optimizer = PoseGraphOptimizer(
        icp_traj,
        lidar_scans,
        icp_results,
        prior_sigma=(0.3, 0.3, np.deg2rad(6.0)),
        odometry_sigma=(0.05, 0.05, np.deg2rad(2.0)),
        loop_sigma=(0.03, 0.03, np.deg2rad(1.0)),
        loop_rmse_threshold=0.18,
        loop_translation_threshold=4.0,
        loop_max_correspondence=1.2,
        loop_icp_max_iterations=90,
    )
    optimizer.add_fixed_interval_loops(
        interval=args.loop_interval,
        max_pairs=args.max_interval_loops,
    )
    optimizer.detect_proximity_loops(
        radius=args.loop_radius,
        min_separation=args.min_loop_separation,
        max_loops=args.max_proximity_loops,
        max_neighbors=6,
    )
    summary = optimizer.optimize(
        max_iterations=args.max_iterations,
        compute_marginals=False,
    )
    pose_graph_traj = summary.optimized_poses
    print(
        "[Overlay] Pose graph completed: "
        f"error {summary.initial_error:.2f} -> {summary.final_error:.2f} "
        f"in {summary.iterations} iterations."
    )

    print("[Overlay] Rebuilding occupancy grid with optimized poses ...")
    grid = OccupancyGridMap(
        resolution=args.map_resolution,
        x_limits=(-args.map_extent, args.map_extent),
        y_limits=(-args.map_extent, args.map_extent),
    )
    for pose, scan in zip(pose_graph_traj, lidar_scans):
        grid.update_from_points(pose, scan, beam_stride=args.beam_stride)

    figure_path = output_dir / f"occupancy_overlay_dataset{args.dataset}.png"
    plot_occupancy_with_trajectories(
        grid,
        odom_traj=odom_at_lidar,
        icp_traj=icp_traj,
        pose_graph_traj=pose_graph_traj,
        output_path=figure_path,
    )
    print(f"[Overlay] Overlay figure saved to {figure_path}")


def plot_occupancy_with_trajectories(
    grid: OccupancyGridMap,
    *,
    odom_traj: np.ndarray,
    icp_traj: np.ndarray,
    pose_graph_traj: np.ndarray,
    output_path: Path,
) -> None:
    probability = grid.probability()
    extent = [
        float(grid.x_limits[0]),
        float(grid.x_limits[1]),
        float(grid.y_limits[0]),
        float(grid.y_limits[1]),
    ]

    plt.figure(figsize=(10, 10))
    plt.imshow(
        probability,
        origin="lower",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
    )
    plt.plot(odom_traj[:, 0], odom_traj[:, 1], label="Odometry", linewidth=1.0, alpha=0.5)
    plt.plot(icp_traj[:, 0], icp_traj[:, 1], label="Sequential ICP", linewidth=1.2, alpha=0.8)
    plt.plot(pose_graph_traj[:, 0], pose_graph_traj[:, 1], label="Pose graph", linewidth=2.0)
    plt.scatter(pose_graph_traj[0, 0], pose_graph_traj[0, 1], c="green", s=40, label="Start")
    plt.scatter(pose_graph_traj[-1, 0], pose_graph_traj[-1, 1], c="red", s=40, label="End")

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Occupancy grid with trajectories")
    plt.axis("equal")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    main()
