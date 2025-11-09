import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.slam.odometry import Odometry
from src.utils.data_loader import load_part1_data
from src.utils.visualizer import plot_trajectory

def main():
    print("="*50)
    print("=== ECE276A PR2 Part 1: オドメトリ計算 ===")
    print("="*50)

    DATASET_NUM = 20
    METERS_PER_TICK = 0.0022

    enc_counts, enc_stamps, imu_ang_vel, imu_stamps = load_part1_data(DATASET_NUM)

    num_steps = enc_stamps.shape[0]
    trajectory = np.zeros((num_steps, 3))
    
    odom = Odometry(initial_pose=(0.0, 0.0, 0.0))
    trajectory[0] = odom.pose

    for i in range(num_steps - 1):
        
        t_curr = enc_stamps[i]
        t_next = enc_stamps[i+1]
        dt = t_next - t_curr
        
        if dt <= 0: continue

        FR, FL, RR, RL = enc_counts[:, i+1]  # エンコーダーは累積カウントでないから
        
        right_dist = (FR + RR) / 2.0 * METERS_PER_TICK
        left_dist = (FL + RL) / 2.0 * METERS_PER_TICK
        
        distance = (right_dist + left_dist) / 2.0
        v = distance / dt
        
        imu_idx = np.argmin(np.abs(imu_stamps - t_next)) # データ同期させる
        omega = imu_ang_vel[2, imu_idx] # projectでz軸がヨーに指定されてる
        
        delta_x_local = v * dt
        delta_y_local = 0.0
        delta_theta = omega * dt
        
        delta_pose = (delta_x_local, delta_y_local, delta_theta)

        odom.update(delta_pose)
        
        trajectory[i+1] = odom.pose

    output_path = f"results/figures/part1_odometry_dataset{DATASET_NUM}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plot_trajectory(trajectory, output_path)

if __name__ == "__main__":
    main()