# Dual Drone - Path Optimization and Simulation

This repository contains a robust robotics environment dedicated to **drone path optimization** and simulation. It leverages **ROS Noetic** (Robot Operating System), the **PX4 Autopilot**, and an **Ant Colony Optimization (ACO)** algorithm to calculate, simulate, and visualize optimal flight paths for drones based on a set of target geotags.

## Features

- **Path Optimization**: A Python-based script (`test.py`) that clusters targets using K-Means and finds the optimal drone route using ACO.
- **Dockerized Environment**: A fully isolated Ubuntu 20.04 environment pre-configured with ROS Noetic and MAVROS.
- **PX4 Autopilot Integration**: Ready-to-use structure for PX4 simulation (`px4-dev`).
- **Map Visualization**: Automatically generates an interactive Folium map to visualize the optimized drone path.

## Prerequisites

Before running the project, ensure you have the following installed on your system:
- [Docker](https://docs.docker.com/get-docker/)
- [Python 3.x](https://www.python.org/downloads/)
- `pip` (Python package installer)

## Installation & Setup

### 1. Python Dependencies
To run the path optimization script locally, install the required Python libraries:

```bash
pip install -r requirements.txt
```

### 2. Docker & ROS Setup
The project uses Docker to ensure consistency across different systems. The included Dockerfile sets up ROS Noetic and MAVROS.

To build the Docker image (if not already built by `run_ros.sh`), you can run:
```bash
docker build -t dual_drone_final2:latest .
```

## Usage

### Running the Drone Path Optimizer
1. Ensure you have a `geotags.json` file in the root directory. This file should contain the coordinates the drone needs to visit.
2. Run the optimization script:
```bash
python3 test.py
```
3. The script will output:
   - `global_aco_path.json`: The optimized path in JSON format.
   - `combined_aco_partitioned_map.html`: An interactive map visualizing the clusters, ghost points, and the drone's final route.

### Launching the ROS Environment
To start the ROS environment inside the Docker container, use the provided bash script. This script automatically mounts the `catkin_ws` and `px4-dev` directories.

```bash
./run_ros.sh
```
Inside the container, you will have access to the ROS Noetic installation, MAVROS, and your local workspace.

## Project Structure
- `test.py`: Main path optimization logic (ACO algorithm).
- `Dockerfile`: Configuration for the ROS Noetic Docker image.
- `run_ros.sh`: Startup script for the Docker container.
- `catkin_ws/`: Local ROS workspace mounted inside the container.
- `px4-dev/`: Local PX4 Autopilot environment mounted inside the container.
- `requirements.txt`: Python package dependencies.
