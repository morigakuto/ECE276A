import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.slam.pipeline import (
    apply_transform,
    build_lidar_scans,
    compute_odometry,
    run_sequential_icp,
    sample_poses_at_stamps,
)
from src.utils.data_loader import load_part1_data, load_part2_data
from src.utils.visualizer import plot_2d_scan_matching


def main():
    dataset_num = 20
    enc_counts, enc_stamps, imu_ang_vel, imu_stamps = load_part1_data(dataset_num)
    lidar_data = load_part2_data(dataset_num)

    odom_poses = compute_odometry(enc_counts, enc_stamps, imu_ang_vel, imu_stamps)

    lidar_scans = build_lidar_scans(lidar_data)
    lidar_stamps = lidar_data["time_stamps"]

    odom_pose_at_lidar = sample_poses_at_stamps(enc_stamps, odom_poses, lidar_stamps)

    refined_poses, icp_results = run_sequential_icp(
        lidar_scans,
        odom_pose_at_lidar,
        max_iterations=60,
        tolerance=1e-4,
        max_correspondence_distance=1.0,
    )

    output_dir = project_root / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path = output_dir / f"part2_scan_matching_dataset{dataset_num}.png"
    plot_trajectory_comparison(
        odom_pose_at_lidar,
        refined_poses,
        trajectory_path,
    )

    sample_idx = min(50, len(lidar_scans) - 1)
    if sample_idx > 0 and icp_results[sample_idx] is not None:
        aligned_scan = apply_transform(
            lidar_scans[sample_idx],
            icp_results[sample_idx].rotation,
            icp_results[sample_idx].translation,
        )
        sample_plot_path = (
            output_dir / f"part2_scan_matching_pair_dataset{dataset_num}.png"
        )
        plot_2d_scan_matching(
            lidar_scans[sample_idx - 1],
            lidar_scans[sample_idx],
            aligned_scan,
            sample_plot_path,
        )

    valid_errors = [
        result.rmse for result in icp_results[1:] if result and np.isfinite(result.rmse)
    ]
    mean_error = float(np.mean(valid_errors)) if valid_errors else float("nan")

    print(f"Processed {len(lidar_scans)} LiDAR scans.")
    print(f"Average ICP RMSE: {mean_error:.4f} m")
    print(f"Trajectory figure saved to: {trajectory_path}")
    if sample_idx > 0:
        print(f"Sample alignment figure saved to: {sample_plot_path}")
def plot_trajectory_comparison(odom_traj, refined_traj, output_path):
    plt.figure(figsize=(10, 8))
    plt.plot(odom_traj[:, 0], odom_traj[:, 1], label="Odometry", linewidth=1.0, alpha=0.7)
    plt.plot(refined_traj[:, 0], refined_traj[:, 1], label="ICP-refined", linewidth=1.5)
    plt.scatter(odom_traj[0, 0], odom_traj[0, 1], color="green", label="Start", s=40)
    plt.scatter(refined_traj[-1, 0], refined_traj[-1, 1], color="red", label="End", s=40)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Part 2: LiDAR Scan Matching Trajectory")
    plt.axis("equal")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    main()
