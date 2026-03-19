# Active Mapping 3D

Autonomous 3D object reconstruction using live incremental Gaussian Splatting on a PX4 drone in Gazebo simulation. The system includes three modes: fixed orbit capture, depth-variance guided Next-Best-View (NBV) planning, and a pose-uncertainty study variant.

## Packages

| Package | Description | Launch file |
|---------|-------------|-------------|
| `mapping` | Orbit baseline — fixed dual-altitude flight pattern with spatial-windowed 3DGS training | `active_recon.launch.py` |
| `nbv` | NBV planner — depth-variance acquisition function scores 95 Fibonacci hemisphere candidates, selects diverse batches of 4 viewpoints per round | `nbv_recon.launch.py` |
| `nbv_pos_uncert` | Pose uncertainty study — extends NBV with per-keyframe EKF position variance logging and drift tracking | `pu_recon.launch.py` |
| `gsplat` | Gazebo world, models, and bridge config (no executables) | — |

## Dependencies

- Ubuntu 24.04
- ROS 2 Jazzy
- PX4-Autopilot (v1.15+)
- Gazebo Harmonic
- Micro-XRCE-DDS-Agent
- NVIDIA GPU with CUDA 12.4+
- Python 3.12

## Setup

### 1. Clone

```bash
git clone https://github.com/DREAMS-lab/active_mapping_3D.git
cd active_mapping_3D
```

### 2. Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 3. Simulation assets and SDF paths

The setup script resolves texture paths in SDF files, copies airframes/models/worlds into PX4, and validates the venv:

```bash
bash setup.sh                        # uses ~/PX4-Autopilot by default
bash setup.sh /path/to/PX4-Autopilot # custom PX4 location
```

### 4. Build

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select mapping gsplat nbv nbv_pos_uncert
```

### 5. Offline reconstruction (optional)

For offline 3DGS training with Kerbl et al.'s original codebase:

```bash
mkdir -p repos && cd repos
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
cd ..
```

## Run

Three terminals are needed for all modes:

**T1 — PX4 SITL:**
```bash
cd ~/PX4-Autopilot
PX4_GZ_NO_FOLLOW=1 PX4_GZ_WORLD=sample_15016 make px4_sitl gz_px4_gsplat
```

**T2 — DDS Agent:**
```bash
MicroXRCEAgent udp4 -p 8888
```

**T3 — Launch (pick one):**

```bash
source install/setup.bash

# Orbit baseline
ros2 launch mapping active_recon.launch.py rock_x:=0.0 rock_y:=8.0

# NBV planner
ros2 launch nbv nbv_recon.launch.py rock_x:=0.0 rock_y:=8.0

# Pose uncertainty study
ros2 launch nbv_pos_uncert pu_recon.launch.py rock_x:=0.0 rock_y:=8.0
```

Or use the convenience script (opens 3 Konsole tabs automatically):
```bash
bash src/nbv/scripts/run_nbv.sh rock_x:=0.0 rock_y:=8.0
```

## Launch Parameters

### Orbit baseline (`active_recon.launch.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rock_x` | 0.0 | Rock position X (NED, meters) |
| `rock_y` | 8.0 | Rock position Y (NED, meters) |
| `orbit_waypoints` | 24 | Waypoints per orbit pass |
| `orbit_radius` | 2.0 | Orbit radius (meters) |
| `iters_per_keyframe` | 500 | Training iterations per round |
| `window_size` | 10 | Spatial window size for training |
| `max_gaussians` | 200000 | Maximum Gaussian count |
| `pts_per_frame` | 40000 | Points sampled per keyframe |
| `enable_viewer` | true | OpenCV splat viewer |
| `enable_orbslam3` | false | ORB-SLAM3 (requires separate build) |

### NBV planner (`nbv_recon.launch.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rock_x` | 0.0 | Rock position X (NED, meters) |
| `rock_y` | 8.0 | Rock position Y (NED, meters) |
| `rock_z_up` | 0.8 | Rock height above ground (meters) |
| `bbox_size` | 2.0 | Bounding box size (meters, auto-detected if 0) |
| `orbit_radius` | 2.5 | Fibonacci hemisphere radius (meters) |
| `kf_budget` | 48 | Total keyframe budget |
| `seed_kfs` | 4 | Number of seed orbit keyframes |
| `batch_size` | 4 | Viewpoints per NBV round |
| `iters_per_keyframe` | 2000 | Training iterations per round |
| `window_size` | 0 | Training window (0 = all views) |
| `max_gaussians` | 500000 | Maximum Gaussian count |
| `pts_per_frame` | 40000 | Points sampled per keyframe |
| `enable_viewer` | true | OpenCV splat viewer |
| `skip_adaptive` | false | Skip NBV, orbit-only seed flight |

### Pose uncertainty study (`pu_recon.launch.py`)

Same parameters as NBV. Additionally logs per-keyframe EKF position variance and commanded-vs-actual position drift.

## Post-Mission Pipeline

After a run completes, convert poses to COLMAP format and run offline 3DGS training:

```bash
# Full pipeline: COLMAP conversion + offline training + render + metrics
bash install/nbv/share/nbv/scripts/postprocess.sh data/nbv/run_001

# Or step by step:
bash install/nbv/share/nbv/scripts/offline_train.sh data/nbv/run_001 30000
```

## Output

Data is saved to `data/<package>/run_NNN/`:
- `images/` — RGB keyframes
- `depth/` — Depth frames (uint16, millimeters)
- `transforms.json` — Camera poses and intrinsics (NED frame)
- `model/` — Live Gaussian splat checkpoint
- `eval/` — Rendered views, loss curves, PSNR/SSIM plots, coverage plots
- `run_metadata.json` — Parameters, timing, per-keyframe metrics
- `offline/` — Offline 3DGS reconstruction (after postprocess)

## Architecture

Single-process, three-thread design:
- **Thread 1 (ROS main):** Flight FSM, keyframe capture, PX4 offboard control, voxel coverage, RViz2
- **Thread 2 (CUDA optimizer):** Incremental 3DGS training with gsplat, spatial windowed training, adaptive iteration budgeting
- **Thread 3 (Viewer):** OpenCV splat viewer with mouse orbit controls

The NBV planner adds a scoring step between capture rounds: render depth second moments at 95 candidate viewpoints, compute per-pixel variance, and select the top-k most uncertain views with angular diversity.

Camera: 640x480 @ 30 fps, fx=fy=465.7412

## Project Structure

```
active_mapping_3D/
├── setup.sh                    # One-command setup
├── requirements.txt            # Python dependencies
├── simulation/
│   ├── airframes/              # PX4 custom airframe
│   ├── models/                 # Gazebo models (gimbal, drone, rocks)
│   ├── textures/               # Moon surface PBR textures
│   └── worlds/                 # Gazebo world SDF
├── src/
│   ├── mapping/                # Orbit baseline package
│   │   ├── launch/             # active_recon.launch.py
│   │   ├── scripts/
│   │   │   ├── active_recon/   # Live orbit mapper + 3DGS
│   │   │   ├── orbit/          # Orbit-only mapper
│   │   │   └── exploration/    # Manual flight utilities
│   │   └── config/             # RViz2 config, bridge YAML
│   ├── nbv/                    # NBV planner package
│   │   ├── launch/             # nbv_recon.launch.py
│   │   └── scripts/            # NBV mapper, planner, viewer, offline tools
│   ├── nbv_pos_uncert/         # Pose uncertainty study package
│   │   ├── launch/             # pu_recon.launch.py
│   │   └── scripts/            # PU mapper, pose noise study
│   └── gsplat/                 # Gazebo world + model definitions
│       ├── models/             # Lunar sample meshes
│       └── worlds/             # World SDF
├── venv/                       # Python virtual environment (created by setup.sh)
├── repos/                      # External repos for offline training (optional)
└── data/                       # Run output (gitignored)
```
