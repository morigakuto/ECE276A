# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ECE276A Project 2 (PR2) codebase focused on robotics perception and SLAM (Simultaneous Localization and Mapping). The project involves working with sensor data from a PR2 robot including:

- **LiDAR data** (Hokuyo sensor): Range measurements for mapping and localization
- **IMU data**: Angular velocity and linear acceleration measurements  
- **Encoder data**: Wheel encoder counts for odometry
- **RGB-D data**: Kinect sensor data with disparity and RGB images

## Code Architecture

### Core Components

- **`code/pr2_utils.py`**: Core utilities including:
  - Bresenham's ray tracing algorithm for grid mapping (`bresenham2D`)
  - LiDAR visualization functions (`show_lidar`, `plot_map`)
  - Grid map initialization and testing (`test_map`)
  - Timing utilities (`tic`, `toc`)

- **`code/load_data.py`**: Data loading script that reads sensor data from `.npz` files:
  - Loads encoder, LiDAR (Hokuyo), IMU, and Kinect data for specified datasets (20 or 21)
  - Provides template structure for accessing all sensor streams

- **`code/icp_warm_up/`**: ICP (Iterative Closest Point) implementation:
  - `test_icp.py`: Main testing script for ICP algorithm on drill/container objects
  - `utils.py`: Utilities for loading canonical models, point clouds, and visualization using Open3D

### Data Structure

- **`data/`**: Contains sensor data files:
  - `Encoders{20,21}.npz`: Wheel encoder data
  - `Hokuyo{20,21}.npz`: LiDAR scan data  
  - `Imu{20,21}.npz`: IMU measurements
  - `Kinect{20,21}.npz`: RGB-D sensor data

- **`dataRGBD/`**: Large collection of disparity images organized by dataset

## Required Dependencies

The project requires these Python packages:
- `numpy` - Core numerical computing
- `matplotlib` - Plotting and visualization  
- `scipy` - Scientific computing (used in ICP warm-up)
- `open3d` - 3D data processing (for ICP visualization)
- `gtsam` - Graph-based SLAM library (install may be required)

## Common Development Tasks

### Testing GTSAM Installation
```bash
cd code
python3 test_gtsam.py
```

### Running PR2 Utilities Tests
```bash
cd code  
python3 pr2_utils.py
```

### Testing ICP Implementation  
```bash
cd code/icp_warm_up
python3 test_icp.py
```

### Loading and Exploring Data
```bash
cd code
python3 load_data.py
```

## Key Implementation Notes

- **Dataset Selection**: Code typically uses `dataset = 20` or `dataset = 21` 
- **Coordinate Systems**: LiDAR data uses polar coordinates that need conversion to Cartesian
- **Grid Mapping**: Uses occupancy grid representation with configurable resolution
- **Ray Tracing**: Bresenham's algorithm is optimized for real-time performance
- **ICP Workflow**: Point cloud alignment for object recognition/pose estimation

## File Naming Conventions

- Sensor data files follow pattern: `{SensorType}{DatasetNumber}.npz`
- Disparity images: `disparity{dataset}_{frame_number}.png`
- Point cloud data: `{object_name}/{id}.npy` format