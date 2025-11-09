"""ECE276A Project 2 - Part 4 pose graph optimization experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

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
from src.slam.texture import TextureMapper
from src.utils.data_loader import (
    load_kinect_timestamps,
    load_part1_data,
    load_part2_data,
)
from src.utils.visualizer import save_occupancy_map, save_texture_map
import imageio.v2 as iio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Part 4: Pose graph optimization with loop closures."
    )
    parser.add_argument("--dataset", type=int, default=20, help="Dataset number (20 or 21).")
    parser.add_argument("--loop-interval", type=int, default=10, help="Pose interval for fixed loop closures.")
    parser.add_argument("--max-interval-loops", type=int, default=80, help="Cap on interval-based loops.")
    parser.add_argument("--loop-radius", type=float, default=0.8, help="Search radius for proximity loops [m].")
    parser.add_argument(
        "--min-loop-separation", type=int, default=25, help="Minimum pose index gap for proximity loops."
    )
    parser.add_argument("--max-proximity-loops", type=int, default=50, help="Cap on proximity-based loops.")
    parser.add_argument("--max-iterations", type=int, default=100, help="Optimizer iteration budget.")
    parser.add_argument("--map-resolution", type=float, default=0.05, help="Occupancy grid resolution [m].")
    parser.add_argument(
        "--map-extent",
        type=float,
        default=30.0,
        help="Half-length of the square occupancy grid in meters.",
    )
    parser.add_argument("--beam-stride", type=int, default=2, help="Down-sample LiDAR beams when mapping.")
    parser.add_argument(
        "--rebuild-texture",
        action="store_true",
        help="Recompute the floor texture map using the optimized poses.",
    )
    parser.add_argument("--kinect-stride", type=int, default=10, help="Fuse every N-th RGB-D frame.")
    parser.add_argument(
        "--max-kinect-frames",
        type=int,
        default=None,
        help="Optional limit on RGB-D frames when rebuilding texture.",
    )
    parser.add_argument("--max-points-per-frame", type=int, default=15000, help="RGB-D down-sampling budget.")
    parser.add_argument("--floor-height", type=float, default=0.0, help="Floor plane height for texture mapping.")
    parser.add_argument(
        "--height-tolerance",
        type=float,
        default=0.3,
        help="Tolerance around the floor plane for texture mapping.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results",
        help="Base directory for figures and map outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir: Path = Path(args.output_dir)
    figures_dir = results_dir / "figures"
    maps_dir = results_dir / "maps"
    figures_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Part4] Loading dataset {args.dataset} ...")
    enc_counts, enc_stamps, imu_ang_vel, imu_stamps = load_part1_data(args.dataset)
    lidar_data = load_part2_data(args.dataset)

    print("[Part4] Computing differential-drive odometry ...")
    odom_poses = compute_odometry(enc_counts, enc_stamps, imu_ang_vel, imu_stamps)

    print("[Part4] Building LiDAR scans and aligning timestamps ...")
    lidar_scans = build_lidar_scans(lidar_data)
    lidar_stamps = lidar_data["time_stamps"]
    odom_pose_at_lidar = sample_poses_at_stamps(enc_stamps, odom_poses, lidar_stamps)

    print("[Part4] Running sequential ICP scan matching ...")
    refined_poses, icp_results = run_sequential_icp(
        lidar_scans,
        odom_pose_at_lidar,
        max_iterations=60,
        tolerance=1e-4,
        max_correspondence_distance=1.0,
    )

    print("[Part4] Building pose graph ...")
    optimizer = PoseGraphOptimizer(
        refined_poses,
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

    interval_loops = optimizer.add_fixed_interval_loops(
        interval=args.loop_interval,
        max_pairs=args.max_interval_loops,
    )
    proximity_loops = optimizer.detect_proximity_loops(
        radius=args.loop_radius,
        min_separation=args.min_loop_separation,
        max_loops=args.max_proximity_loops,
        max_neighbors=6,
    )
    print(f"[Part4] Added {interval_loops} interval loops and {proximity_loops} proximity loops.")

    print("[Part4] Optimizing factor graph ... (this step requires GTSAM)")
    summary = optimizer.optimize(
        max_iterations=args.max_iterations,
        compute_marginals=False,
    )
    optimized_poses = summary.optimized_poses

    print(
        "[Part4] Optimization finished: "
        f"error {summary.initial_error:.2f} -> {summary.final_error:.2f} "
        f"in {summary.iterations} iterations, {len(summary.loop_closures)} loop edges."
    )

    traj_plot = figures_dir / f"part4_pose_graph_dataset{args.dataset}.png"
    plot_pose_graph_trajectory(
        odom_pose_at_lidar,
        refined_poses,
        optimized_poses,
        summary.loop_closures,
        traj_plot,
    )
    print(f"[Part4] Saved trajectory comparison to {traj_plot}")

    print("[Part4] Rebuilding occupancy grid with optimized poses ...")
    occ_grid = OccupancyGridMap(
        resolution=args.map_resolution,
        x_limits=(-args.map_extent, args.map_extent),
        y_limits=(-args.map_extent, args.map_extent),
    )
    for pose, scan in zip(optimized_poses, lidar_scans):
        occ_grid.update_from_points(pose, scan, beam_stride=args.beam_stride)

    occ_map_path = maps_dir / f"part4_occupancy_dataset{args.dataset}.png"
    save_occupancy_map(
        occ_grid.probability(),
        occ_map_path,
        title="Part 4: Occupancy (optimized)",
    )
    print(f"[Part4] Saved occupancy map to {occ_map_path}")

    if args.rebuild_texture:
        if iio is None:
            raise RuntimeError(
                "imageio is required to rebuild the texture map. "
                "Re-run `uv add imageio` or `pip install imageio`."
            )
        rebuild_texture_map(
            dataset=args.dataset,
            optimized_poses=optimized_poses,
            lidar_stamps=lidar_stamps,
            grid=occ_grid,
            stride=args.kinect_stride,
            max_frames=args.max_kinect_frames,
            max_points=args.max_points_per_frame,
            floor_height=args.floor_height,
            height_tolerance=args.height_tolerance,
            maps_dir=maps_dir,
        )


def plot_pose_graph_trajectory(
    odom_traj: np.ndarray,
    refined_traj: np.ndarray,
    optimized_traj: np.ndarray,
    loop_closures: Sequence,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 8))
    plt.plot(odom_traj[:, 0], odom_traj[:, 1], label="Odometry seed", linewidth=1.0, alpha=0.5)
    plt.plot(refined_traj[:, 0], refined_traj[:, 1], label="Sequential ICP", linewidth=1.2, alpha=0.8)
    plt.plot(optimized_traj[:, 0], optimized_traj[:, 1], label="Pose graph", linewidth=2.0)
    plt.scatter(optimized_traj[0, 0], optimized_traj[0, 1], c="green", s=40, label="Start")
    plt.scatter(optimized_traj[-1, 0], optimized_traj[-1, 1], c="red", s=40, label="End")

    if loop_closures:
        for closure in loop_closures:
            start = optimized_traj[closure.from_idx]
            end = optimized_traj[closure.to_idx]
            plt.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="k",
                linestyle="--",
                linewidth=0.8,
                alpha=0.3,
            )

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Part 4: Pose graph optimization")
    plt.axis("equal")
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def rebuild_texture_map(
    dataset: int,
    optimized_poses: np.ndarray,
    lidar_stamps: np.ndarray,
    grid: OccupancyGridMap,
    stride: int,
    max_frames: Optional[int],
    max_points: Optional[int],
    floor_height: float,
    height_tolerance: float,
    maps_dir: Path,
) -> None:
    print("[Part4] Rebuilding texture map with optimized poses ...")
    disp_stamps, rgb_stamps = load_kinect_timestamps(dataset)
    disp_paths = collect_kinect_files(dataset, "Disparity")
    rgb_paths = collect_kinect_files(dataset, "RGB")

    frame_count = min(len(disp_paths), len(rgb_paths), len(disp_stamps), len(rgb_stamps))
    if frame_count == 0:
        raise RuntimeError("No RGB-D frames available for texture mapping.")
    if max_frames:
        frame_count = min(frame_count, max_frames)

    disp_paths = disp_paths[:frame_count]
    rgb_paths = rgb_paths[:frame_count]
    disp_stamps = disp_stamps[:frame_count]

    poses_at_kinect = sample_poses_at_stamps(lidar_stamps, optimized_poses, disp_stamps)

    texture_mapper = TextureMapper(
        grid=grid,
        floor_height=floor_height,
        height_tolerance=height_tolerance,
    )

    fused_points = 0
    processed = 0
    for idx in range(0, frame_count, max(stride, 1)):
        disparity = iio.imread(disp_paths[idx])
        rgb = iio.imread(rgb_paths[idx])
        pose = poses_at_kinect[idx]
        fused = texture_mapper.integrate_frame(
            pose,
            disparity,
            rgb,
            max_points=max_points,
        )
        fused_points += fused
        processed += 1

    texture, observed = texture_mapper.build_texture_image()
    texture_path = maps_dir / f"part4_texture_dataset{dataset}.png"
    save_texture_map(
        texture,
        texture_path,
        observed_mask=observed,
        title="Part 4: Texture (optimized)",
    )
    print(
        f"[Part4] Saved texture map to {texture_path} "
        f"({processed} frames, {fused_points} floor pixels)."
    )


def collect_kinect_files(dataset: int, kind: str) -> list[Path]:
    folder = project_root / "dataRGBD" / f"{kind}{dataset}"
    if not folder.exists():
        raise FileNotFoundError(f"Kinect folder not found: {folder}")
    prefix = f"{kind.lower()}{dataset}_"

    def frame_index(path: Path) -> int:
        stem = path.stem
        if "_" not in stem:
            return 0
        try:
            return int(stem.split("_")[1])
        except ValueError:
            return 0

    paths = sorted(folder.glob("*.png"), key=frame_index)
    return [p for p in paths if p.name.startswith(prefix)]


if __name__ == "__main__":
    main()
