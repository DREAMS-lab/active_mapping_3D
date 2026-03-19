#!/usr/bin/env bash
# Setup script for active_mapping_3D
#
# 1. Resolves SDF texture paths to this repo's absolute location
# 2. Copies PX4 simulation assets (airframes, models, worlds) into PX4-Autopilot
# 3. Creates a Python venv with torch + gsplat (if not already present)
#
# Usage:
#   bash setup.sh [PX4_DIR]
#
# Example:
#   bash setup.sh                          # defaults to ~/PX4-Autopilot
#   bash setup.sh /opt/PX4-Autopilot       # custom PX4 location

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PX4_DIR="${1:-$HOME/PX4-Autopilot}"

echo "============================================"
echo "  Active Mapping 3D — Setup"
echo "============================================"
echo "  Repo root: $REPO_ROOT"
echo "  PX4 dir:   $PX4_DIR"
echo ""

# ── Step 1: Fix SDF texture paths ──
echo "--- Step 1: Resolving SDF texture paths ---"
for sdf in "$REPO_ROOT/simulation/worlds/sample_15016.sdf" \
           "$REPO_ROOT/src/gsplat/worlds/sample_15016.sdf"; do
    if [ -f "$sdf" ]; then
        sed -i "s|__REPO_ROOT__|$REPO_ROOT|g" "$sdf"
        echo "  Updated: $sdf"
    fi
done
echo ""

# ── Step 2: Copy PX4 simulation assets ──
if [ -d "$PX4_DIR" ]; then
    echo "--- Step 2: Copying simulation assets to PX4 ---"

    # Airframe
    cp "$REPO_ROOT/simulation/airframes/4022_gz_px4_gsplat" \
       "$PX4_DIR/ROMFS/px4fmu_common/init.d-posix/airframes/"
    echo "  Copied airframe"

    # Gazebo models
    for model in gimbal_rgbd px4_gsplat rock; do
        cp -r "$REPO_ROOT/simulation/models/$model" \
           "$PX4_DIR/Tools/simulation/gz/models/"
    done

    # Lunar sample models (from src/gsplat/models/)
    for model in lunar_sample_15016 lunar_sample_65035; do
        cp -r "$REPO_ROOT/src/gsplat/models/$model" \
           "$PX4_DIR/Tools/simulation/gz/models/"
    done
    echo "  Copied Gazebo models"

    # World SDF (with resolved paths)
    cp "$REPO_ROOT/simulation/worlds/sample_15016.sdf" \
       "$PX4_DIR/Tools/simulation/gz/worlds/"
    echo "  Copied world SDF"
    echo ""
else
    echo "--- Step 2: Skipped (PX4 dir not found at $PX4_DIR) ---"
    echo "  Run again with: bash setup.sh /path/to/PX4-Autopilot"
    echo ""
fi

# ── Step 3: Python venv ──
VENV_DIR="$REPO_ROOT/venv"
if [ -d "$VENV_DIR" ]; then
    echo "--- Step 3: Python venv already exists at $VENV_DIR ---"
else
    echo "--- Step 3: Creating Python venv ---"
    python3 -m venv "$VENV_DIR"
    echo "  Created venv at $VENV_DIR"
    echo "  Install dependencies with:"
    echo "    source $VENV_DIR/bin/activate"
    echo "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
    echo "    pip install gsplat==1.5.3 opencv-python matplotlib numpy"
fi
echo ""

echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Build:  colcon build --symlink-install --packages-select mapping gsplat nbv"
echo "  Run:    source install/setup.bash"
echo "          ros2 launch nbv nbv_recon.launch.py rock_x:=0.0 rock_y:=8.0"
echo "============================================"
