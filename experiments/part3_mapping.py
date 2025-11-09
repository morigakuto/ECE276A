import argparse
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.slam.mapping import OccupancyGridMap
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ECE276A Project 2 - Part 3 mapping pipeline")
    parser.add_argument("--dataset", type=int, default=20, help="Dataset number (20 or 21)")
    parser.add_argument("--map-resolution", type=float, default=0.05, help="Grid resolution in meters/cell")
    parser.add_argument(
        "--map-extent",
        type=float,
        default=30.0,
        help="Half-length of the square map in meters (map spans [-extent, extent])",
    )
    parser.add_argument("--beam-stride", type=int, default=2, help="LiDAR ray down-sampling stride")
    parser.add_argument("--kinect-stride", type=int, default=10, help="Fuse every N-th RGB-D frame")
    parser.add_argument(
        "--max-kinect-frames",
        type=int,
        default=None,
        help="Optional limit on the number of RGB-D frames to use",
    )
    parser.add_argument(
        "--max-points-per-frame",
        type=int,
        default=15000,
        help="Down-sample RGB-D points per frame to this budget (set None to disable)",
    )
    parser.add_argument("--floor-height", type=float, default=0.5, help="Floor plane height in meters")
    parser.add_argument("--height-tolerance", type=float, default=0.3, help="Tolerance when filtering floor points")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results" / "maps",
        help="Directory where the occupancy and texture maps will be stored",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Part3] Loading dataset {args.dataset} ...")
    enc_counts, enc_stamps, imu_ang_vel, imu_stamps = load_part1_data(args.dataset)
    lidar_data = load_part2_data(args.dataset)

    print("[Part3] Computing odometry seed ...")
    odom_poses = compute_odometry(enc_counts, enc_stamps, imu_ang_vel, imu_stamps)

    lidar_scans = build_lidar_scans(lidar_data)
    lidar_stamps = lidar_data["time_stamps"]
    odom_pose_at_lidar = sample_poses_at_stamps(enc_stamps, odom_poses, lidar_stamps)

    print("[Part3] Running sequential ICP scan matching ...")
    refined_poses, icp_results = run_sequential_icp(
        lidar_scans,
        odom_pose_at_lidar,
        max_iterations=60,
        tolerance=1e-4,
        max_correspondence_distance=1.0,
    )
    converged = sum(1 for result in icp_results[1:] if result and result.converged)
    print(f"[Part3] ICP finished. Converged frames: {converged}/{len(icp_results) - 1}")

    grid = OccupancyGridMap(
        resolution=args.map_resolution,
        x_limits=(-args.map_extent, args.map_extent),
        y_limits=(-args.map_extent, args.map_extent),
    )

    print("[Part3] Integrating LiDAR scans into occupancy grid ...")
    total_hits = 0
    for pose, scan in zip(refined_poses, lidar_scans):
        total_hits += grid.update_from_points(pose, scan, beam_stride=args.beam_stride)
    print(f"[Part3] Occupancy grid updated with {total_hits} LiDAR endpoints.")

    occ_map_path = output_dir / f"part3_occupancy_dataset{args.dataset}.png"
    save_occupancy_map(grid.probability(), occ_map_path)

    print("[Part3] Preparing RGB-D frames ...")
    disp_stamps, rgb_stamps = load_kinect_timestamps(args.dataset)
    disp_paths = collect_kinect_files(args.dataset, kind="Disparity")
    rgb_paths = collect_kinect_files(args.dataset, kind="RGB")

    frame_count = min(len(disp_paths), len(rgb_paths), len(disp_stamps), len(rgb_stamps))
    if args.max_kinect_frames:
        frame_count = min(frame_count, args.max_kinect_frames)
    if frame_count == 0:
        raise RuntimeError("No RGB-D frames found for the selected dataset.")

    disp_paths = disp_paths[:frame_count]
    rgb_paths = rgb_paths[:frame_count]
    disp_stamps = disp_stamps[:frame_count]

    poses_at_kinect = sample_poses_at_stamps(lidar_stamps, refined_poses, disp_stamps)

    texture_mapper = TextureMapper(
        grid=grid,
        floor_height=args.floor_height,
        height_tolerance=args.height_tolerance,
    )

    processed_frames = 0
    fused_points = 0
    for idx in range(0, frame_count, max(args.kinect_stride, 1)):
        pose = poses_at_kinect[idx]
        disparity = iio.imread(disp_paths[idx])
        rgb = iio.imread(rgb_paths[idx])
        fused = texture_mapper.integrate_frame(
            pose,
            disparity,
            rgb,
            max_points=args.max_points_per_frame,
        )
        fused_points += fused
        processed_frames += 1

    print(
        f"[Part3] Integrated {processed_frames} RGB-D frames "
        f"({fused_points} projected floor pixels)."
    )

    texture, observed_mask = texture_mapper.build_texture_image()
    texture_path = output_dir / f"part3_texture_dataset{args.dataset}.png"
    save_texture_map(texture, texture_path, observed_mask=observed_mask)

    print(f"[Part3] Results saved to: {occ_map_path} and {texture_path}")


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
    paths = [p for p in paths if p.name.startswith(prefix)]
    return paths


if __name__ == "__main__":
    main()
