# LiDAR-Based SLAM

## Project Overview

This project implements a comprehensive SLAM (Simultaneous Localization and Mapping) system for the PR2 robot, developed as part of ECE276A at UC San Diego. The implementation combines dead reckoning, scan matching, occupancy grid mapping, and pose graph optimization to create accurate maps from real sensor data.

## Technical Report

For a detailed analysis of the algorithms, implementation details, and experimental results, please refer to the **[Technical Report (PDF)](ECE276A_PR2_report.pdf)**.

## Key Features

- **Occupancy Grid Mapping**: Log-odds based probabilistic mapping with Bresenham ray tracing
- **ICP-based Scan Matching**: Sequential point cloud alignment with intelligent gating to reject outliers
- **Pose Graph Optimization**: Loop closure detection and global consistency enforcement using GTSAM
- **Texture Mapping**: RGB-D sensor fusion for colored point cloud generation
- **Dead Reckoning**: Differential drive odometry with IMU-based heading correction

## Implementation Highlights

### 1. Dead Reckoning (Part 1)
- Fused wheel encoder data with IMU measurements
- Implemented differential drive kinematics model
- Achieved smooth trajectory estimation as baseline

### 2. Scan Matching (Part 2)
- Iterative Closest Point (ICP) algorithm for LiDAR scan alignment
- Added intelligent gating mechanism to reject anomalous corrections
- Reduced trajectory drift from meters to centimeters

### 3. Occupancy Grid Mapping with Texture (Part 3)
- Log-odds based probabilistic grid mapping
- Bresenham's line algorithm for efficient ray tracing
- RGB-D texture mapping integrated with occupancy grid
- Real-time map updates from LiDAR scans with configurable resolution

### 4. Loop Closure (Part 4)
- Pose graph construction with odometry and loop constraints
- GTSAM-based optimization for global consistency
- Automatic loop detection using spatial proximity and ICP validation

## Results

The system successfully processes two datasets (Dataset 20 and Dataset 21) containing:
- ~5000 LiDAR scans per dataset
- Wheel encoder measurements at 40Hz
- IMU data at 100Hz
- RGB-D images from Kinect sensor

Key achievements:
- **Map Quality**: Generated high-resolution occupancy grids matching ground truth
- **Localization Accuracy**: Sub-meter average position error
- **Computational Efficiency**: Real-time capable processing
- **Robustness**: Handles sensor noise and scan matching failures gracefully

## Project Structure

```
ECE276A_PR2/
|-- src/
|   |-- slam/                   # Core SLAM algorithms
|   |   |-- icp.py              # ICP implementation with gating
|   |   |-- mapping.py          # Occupancy grid mapping
|   |   |-- odometry.py         # Odometry pose composition
|   |   |-- optimizer.py        # Pose graph optimization
|   |   |-- pipeline.py         # Processing pipeline
|   |   `-- texture.py          # RGB-D texture mapping
|   `-- utils/                  # Utility functions
|       |-- data_loader.py     # Sensor data loading utilities
|       `-- visualizer.py      # Map visualization functions
|-- experiments/                # Experiment scripts for each part
|   |-- part1_odometry.py      # Dead reckoning experiment
|   |-- part2_scan_matching.py # ICP scan matching experiment
|   |-- part3_mapping.py        # Occupancy & texture mapping
|   `-- part4_pose_graph.py    # Loop closure optimization
|-- results/                    # Generated maps and visualizations
|-- data/                       # Sensor data (not included)
`-- ECE276A_PR2_report.pdf      # Technical report
```
