"""Validate the ICP fix by checking rejection rates and correction ratios."""

import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from src.slam.pipeline import (
    build_lidar_scans,
    compute_odometry,
    run_sequential_icp,
    sample_poses_at_stamps,
    current_to_previous_transform,
)
from src.utils.data_loader import load_part1_data, load_part2_data


def validate_icp_fix(dataset_num: int = 20):
    # Load data
    enc_counts, enc_stamps, imu_ang_vel, imu_stamps = load_part1_data(dataset_num)
    lidar_data = load_part2_data(dataset_num)

    # Compute odometry
    odom_poses = compute_odometry(enc_counts, enc_stamps, imu_ang_vel, imu_stamps)

    # Build lidar scans
    lidar_scans = build_lidar_scans(lidar_data)
    lidar_stamps = lidar_data["time_stamps"]

    # Sample odometry at lidar timestamps
    odom_pose_at_lidar = sample_poses_at_stamps(enc_stamps, odom_poses, lidar_stamps)

    # Run ICP with gating
    refined_poses, icp_results = run_sequential_icp(
        lidar_scans,
        odom_pose_at_lidar,
        max_iterations=60,
        tolerance=1e-4,
        max_correspondence_distance=1.0,
        max_translation_ratio=3.0,
        max_rotation_diff=np.deg2rad(15),
        max_absolute_translation=0.5,
    )

    # Calculate statistics
    corrections = []
    rejection_count = 0
    accepted_count = 0
    large_correction_frames = []

    for idx in range(1, len(icp_results)):
        odom_delta = current_to_previous_transform(
            odom_pose_at_lidar[idx - 1],
            odom_pose_at_lidar[idx]
        )
        odom_tx, odom_ty, _ = odom_delta
        odom_trans = np.sqrt(odom_tx**2 + odom_ty**2)

        result = icp_results[idx]
        if result is None:
            rejection_count += 1
        elif result is not None and np.isfinite(result.rmse):
            if result.accepted:
                accepted_count += 1
                icp_trans = np.linalg.norm(result.translation)
                if odom_trans > 1e-3:
                    ratio = icp_trans / odom_trans
                    corrections.append(ratio)
                    if ratio > 5.0:
                        large_correction_frames.append({
                            'frame': idx,
                            'ratio': ratio,
                            'icp_trans': icp_trans,
                            'odom_trans': odom_trans,
                            'rmse': result.rmse
                        })
            else:
                rejection_count += 1

    total_frames = len(icp_results) - 1
    print(f"\n=== ICP Fix Validation Results ===")
    print(f"Total frames processed: {total_frames}")
    print(f"ICP accepted: {accepted_count} ({100*accepted_count/total_frames:.1f}%)")
    print(f"ICP rejected: {rejection_count} ({100*rejection_count/total_frames:.1f}%)")

    if corrections:
        print(f"\n=== Correction Statistics (accepted ICPs only) ===")
        print(f"Mean correction ratio: {np.mean(corrections):.3f}")
        print(f"Median correction ratio: {np.median(corrections):.3f}")
        print(f"Max correction ratio: {np.max(corrections):.3f}")
        print(f"Corrections > 3.0x: {sum(1 for r in corrections if r > 3.0)}")
        print(f"Corrections > 5.0x: {sum(1 for r in corrections if r > 5.0)}")
        print(f"Corrections > 10.0x: {sum(1 for r in corrections if r > 10.0)}")

    if large_correction_frames:
        print(f"\n=== Large corrections that were ACCEPTED (> 5x) ===")
        print(f"Found {len(large_correction_frames)} frames with large corrections:")
        for frame_info in large_correction_frames[:5]:  # Show first 5
            print(f"  Frame {frame_info['frame']}: "
                  f"ratio={frame_info['ratio']:.2f}, "
                  f"ICP={frame_info['icp_trans']:.3f}m, "
                  f"Odom={frame_info['odom_trans']:.3f}m, "
                  f"RMSE={frame_info['rmse']:.3f}")
    else:
        print(f"\n=== No large corrections (> 5x) were accepted! ===")

    # Compute trajectory deviation from odometry
    odom_trajectory = odom_pose_at_lidar[:, :2]
    refined_trajectory = refined_poses[:, :2]
    deviations = np.linalg.norm(refined_trajectory - odom_trajectory, axis=1)

    print(f"\n=== Trajectory Deviation from Odometry ===")
    print(f"Mean deviation: {np.mean(deviations):.3f} m")
    print(f"Max deviation: {np.max(deviations):.3f} m")
    print(f"90th percentile deviation: {np.percentile(deviations, 90):.3f} m")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot trajectories
    ax = axes[0, 0]
    ax.plot(odom_trajectory[:, 0], odom_trajectory[:, 1], 'b-', label='Odometry', alpha=0.7)
    ax.plot(refined_trajectory[:, 0], refined_trajectory[:, 1], 'r-', label='ICP Refined', alpha=0.7)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Plot correction ratio distribution
    ax = axes[0, 1]
    if corrections:
        ax.hist(corrections, bins=50, range=(0, 5), alpha=0.7, edgecolor='black')
        ax.axvline(x=1.0, color='g', linestyle='--', label='Perfect match (1.0)')
        ax.axvline(x=3.0, color='r', linestyle='--', label='Rejection threshold (3.0)')
        ax.set_xlabel('ICP/Odometry Translation Ratio')
        ax.set_ylabel('Count')
        ax.set_title('Correction Ratio Distribution (Accepted ICPs)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot deviation over time
    ax = axes[1, 0]
    ax.plot(deviations)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Deviation from Odometry [m]')
    ax.set_title('Trajectory Deviation Over Time')
    ax.grid(True, alpha=0.3)

    # Plot acceptance rate over time
    ax = axes[1, 1]
    acceptance = []
    window_size = 100
    for i in range(1, len(icp_results)):
        start = max(1, i - window_size // 2)
        end = min(len(icp_results), i + window_size // 2)
        window_accepted = sum(
            1 for j in range(start, end)
            if icp_results[j] is not None and icp_results[j].accepted
        )
        window_total = end - start
        acceptance.append(100 * window_accepted / window_total if window_total > 0 else 0)

    ax.plot(acceptance)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Acceptance Rate [%]')
    ax.set_title(f'ICP Acceptance Rate (window={window_size} frames)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    plt.suptitle(f'ICP Fix Validation - Dataset {dataset_num}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_path = Path("results/figures") / f"icp_fix_validation_dataset{dataset_num}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nValidation plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    validate_icp_fix(20)