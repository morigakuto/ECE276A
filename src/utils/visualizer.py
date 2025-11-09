import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_trajectory(trajectory, output_filename):
    
    x_coords = trajectory[:, 0]
    y_coords = trajectory[:, 1]
    
    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, y_coords, 'b-', linewidth=1, label='Robot trajectory')
    plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Part 1: Robot Odometry (Encoder + IMU)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.savefig(output_filename)
    print(f"軌跡を '{output_filename}' に保存しました。")
    plt.close()

def plot_2d_scan_matching(reference_scan, target_scan, aligned_scan, output_filename):
    
    plt.figure(figsize=(10, 10))

    if len(reference_scan) > 0:
        plt.scatter(reference_scan[:, 0], reference_scan[:, 1],
                    s=5, c='tab:blue', alpha=0.6, label='Reference scan')
    if len(target_scan) > 0:
        plt.scatter(target_scan[:, 0], target_scan[:, 1],
                    s=5, c='tab:red', alpha=0.3, label='Moving scan (before ICP)')
    if len(aligned_scan) > 0:
        plt.scatter(aligned_scan[:, 0], aligned_scan[:, 1],
                    s=5, c='tab:green', alpha=0.6, label='Moving scan (after ICP)')

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('LiDAR Scan Matching Result')
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.grid(True)

    plt.savefig(output_filename)
    print(f"スキャンマッチング結果を '{output_filename}' に保存しました。")
    plt.close()


def save_occupancy_map(prob_map, output_filename, title="Part 3: Occupancy Grid"):
    grid = np.asarray(prob_map)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
    plt.colorbar(label="P(occupied)")
    plt.title(title)
    plt.xlabel("X cells")
    plt.ylabel("Y cells")
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"占有格子地図を '{output_filename}' に保存しました。")
    plt.close()


def save_texture_map(texture_map, output_filename, observed_mask=None, title="Part 3: Texture Map"):
    texture = np.asarray(texture_map)
    display = texture.astype(np.uint8, copy=True)
    if observed_mask is not None:
        mask = np.asarray(observed_mask, dtype=bool)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2)
        if mask.shape != display.shape[:2]:
            raise ValueError("observed_mask shape must match the texture height/width.")
        background_color = np.array([20, 20, 20], dtype=np.uint8)
        background = np.broadcast_to(background_color.reshape(1, 1, -1), display.shape)
        display = np.where(mask[..., None], display, background)

    plt.figure(figsize=(8, 8))
    plt.imshow(display, origin="lower")
    plt.title(title)
    plt.xlabel("X cells")
    plt.ylabel("Y cells")
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"テクスチャマップを '{output_filename}' に保存しました。")
    plt.close()
