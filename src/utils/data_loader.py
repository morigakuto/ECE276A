import numpy as np
import os
import sys

def load_part1_data(dataset_num=20):
    """
    Part 1用のエンコーダーとIMUデータを読み込みます。

    Args:
        dataset_num (int): データセット番号 (20 または 21)

    Returns:
        tuple: (encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps)
    """
    # エンコーダーデータ
    with np.load(f"data/Encoders{dataset_num}.npz") as data:
        encoder_counts = data["counts"]  # 4 x n
        encoder_stamps = data["time_stamps"]  # n

    # IMUデータ
    with np.load(f"data/Imu{dataset_num}.npz") as data:
        imu_angular_velocity = data["angular_velocity"]  # 3 x m
        imu_stamps = data["time_stamps"]  # m

    return encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps


def load_part2_data(dataset_num=20):
    lidar_data = {}

    with np.load(f"data/Hokuyo{dataset_num}.npz") as data:
        lidar_data['angle_min'] = data["angle_min"]  # スキャン開始角度 [rad]
        lidar_data['angle_max'] = data["angle_max"]  # スキャン終了角度 [rad]
        lidar_data['angle_increment'] = data["angle_increment"]  # 角度分解能 [rad]
        lidar_data['range_min'] = data["range_min"]  # 最小レンジ [m]
        lidar_data['range_max'] = data["range_max"]  # 最大レンジ [m]
        lidar_data['ranges'] = data["ranges"].T  # レンジデータ [m] (num_scans, num_beams)
        lidar_data['time_stamps'] = data["time_stamps"]  # タイムスタンプ

    return lidar_data



def load_full_slam_data(dataset_num=20):
    """
    SLAM用の全データ（エンコーダー、IMU、LiDAR、Kinect）を読み込みます。

    Args:
        dataset_num (int): データセット番号 (20 または 21)

    Returns:
        dict: 全センサーデータを含む辞書
    """
    data_dict = {}

    # エンコーダーデータ
    with np.load(f"data/Encoders{dataset_num}.npz") as data:
        data_dict['encoder_counts'] = data["counts"]
        data_dict['encoder_stamps'] = data["time_stamps"]

    # IMUデータ
    with np.load(f"data/Imu{dataset_num}.npz") as data:
        data_dict['imu_angular_velocity'] = data["angular_velocity"]
        data_dict['imu_linear_acceleration'] = data["linear_acceleration"]
        data_dict['imu_stamps'] = data["time_stamps"]

    # LiDARデータ
    with np.load(f"data/Hokuyo{dataset_num}.npz") as data:
        data_dict['lidar_angle_min'] = data["angle_min"]
        data_dict['lidar_angle_max'] = data["angle_max"]
        data_dict['lidar_angle_increment'] = data["angle_increment"]
        data_dict['lidar_range_min'] = data["range_min"]
        data_dict['lidar_range_max'] = data["range_max"]
        data_dict['lidar_ranges'] = data["ranges"]
        data_dict['lidar_stamps'] = data["time_stamps"]

    # Kinectデータ
    with np.load(f"data/Kinect{dataset_num}.npz") as data:
        data_dict['disp_stamps'] = data["disparity_time_stamps"]
        data_dict['rgb_stamps'] = data["rgb_time_stamps"]

    return data_dict


def load_kinect_timestamps(dataset_num=20):
    with np.load(f"data/Kinect{dataset_num}.npz") as data:
        disparity_stamps = data["disparity_time_stamps"]
        rgb_stamps = data["rgb_time_stamps"]
    return disparity_stamps, rgb_stamps
