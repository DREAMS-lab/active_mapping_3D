# Active Mapping 3D

Autonomous 3D reconstruction using live incremental Gaussian Splatting on a PX4 drone in Gazebo simulation. The drone flies orbit patterns around objects, captures RGB-D images, and builds 3D Gaussian Splat reconstructions in real-time using gsplat.

## Dependencies

- ROS 2 Jazzy
- PX4-Autopilot (with custom airframe and models — see `simulation/README.md`)
- Micro-XRCE-DDS-Agent
- Python 3.12 venv with:
  - `torch >= 2.6` (CUDA 12.4)
  - `gsplat == 1.5.3`
  - `opencv-python`
  - `matplotlib`
  - `numpy < 2`

## Clone

```bash
git clone --recursive https://github.com/DREAMS-lab/active_mapping_3D.git
```

## Build

```bash
cd active_mapping_3D
source /opt/ros/jazzy/setup.bash
colcon build --packages-select mapping gsplat --symlink-install
```

## Simulation Setup

Before the first run, copy custom PX4 models, airframe, and world files into your PX4-Autopilot installation. See [`simulation/README.md`](simulation/README.md) for instructions.

## Run

Three terminals:

**T1 — PX4 SITL:**
```bash
cd ~/PX4-Autopilot
PX4_GZ_NO_FOLLOW=1 PX4_GZ_WORLD=sample_15016 make px4_sitl gz_px4_gsplat
```

**T2 — DDS Agent:**
```bash
MicroXRCEAgent udp4 -p 8888
```

**T3 — Active Reconstruction:**
```bash
source active_mapping_3D/install/setup.bash
ros2 launch mapping active_recon.launch.py rock_x:=0.0 rock_y:=8.0
```

This will:
1. Take off and fly two orbit passes around the rock (24 waypoints each)
2. Train a 3D Gaussian Splat model live during flight
3. Open an OpenCV splat viewer (drag to orbit, scroll to zoom)
4. Show voxel coverage and camera poses in RViz2
5. Save evaluation outputs (renders, PSNR/SSIM, plots) after landing

## Launch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rock_x` | 0.0 | Rock position X (NED, meters) |
| `rock_y` | 8.0 | Rock position Y (NED, meters) |
| `orbit_waypoints` | 24 | Waypoints per orbit pass |
| `orbit_radius` | 2.0 | Orbit radius (meters) |
| `iters_per_keyframe` | 500 | Training iterations per round |
| `window_size` | 10 | Spatial window size for training |
| `max_gaussians` | 200000 | Maximum gaussian count |
| `pts_per_frame` | 40000 | Points sampled per keyframe |
| `enable_viewer` | true | OpenCV splat viewer |
| `enable_orbslam3` | false | ORB-SLAM3 (requires separate build) |

## Output

Data is saved to `data/mapping/run_NNN/`:
- `images/` — RGB frames
- `depth/` — Depth frames (uint16, millimeters)
- `transforms.json` — Camera poses and intrinsics
- `model/` — Gaussian splat checkpoint
- `eval/` — Rendered views, loss curves, PSNR/SSIM plots, coverage plots

## Architecture

Single-process, three-thread design:
- **Thread 1 (ROS main):** Flight FSM, keyframe capture, PX4 offboard control, voxel coverage, RViz2
- **Thread 2 (CUDA optimizer):** Incremental 3DGS training with gsplat, spatial windowed training, adaptive iteration budgeting
- **Thread 3 (Viewer):** OpenCV splat viewer with mouse orbit controls

Camera: 640x480 @ 30fps, fx=fy=465.7412
