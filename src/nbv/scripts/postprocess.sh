#!/usr/bin/env bash
# Post-mission pipeline: COLMAP conversion + depth prep + offline 3DGS + render + metrics
#
# Runs everything needed after the drone lands and the ROS node exits.
#
# Usage:
#   bash postprocess.sh <run_dir> [iterations]
#
# Example:
#   bash postprocess.sh <workspace>/data/nbv/run_007
#   bash postprocess.sh <workspace>/data/nbv/run_007 50000

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_dir> [iterations]"
    exit 1
fi

RUN_DIR="$(realpath "$1")"
ITERATIONS="${2:-30000}"

# Resolve paths — derive workspace root from script install location
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
GS_WS="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
VENV_PYTHON="$GS_WS/venv/bin/python3"
POSES_SCRIPT="$SCRIPT_DIR/poses_to_colmap.py"
OFFLINE_SCRIPT="$SCRIPT_DIR/offline_train.sh"
GS_REPO="$GS_WS/repos/gaussian-splatting"

# Validate
if [ ! -f "$RUN_DIR/transforms.json" ]; then
    echo "Error: $RUN_DIR/transforms.json not found"
    exit 1
fi

echo "============================================"
echo "  NBV Post-Mission Pipeline"
echo "============================================"
echo "  Run dir:    $RUN_DIR"
echo "  Iterations: $ITERATIONS"
echo ""

# Step 1: Convert poses to COLMAP format + generate inverse depth
echo "--- Step 1/4: Poses to COLMAP + inverse depth ---"
"$VENV_PYTHON" "$POSES_SCRIPT" "$RUN_DIR"
echo ""

# Step 2: Offline 3DGS training (with depth supervision if available)
echo "--- Step 2/4: Offline 3DGS training ---"
bash "$OFFLINE_SCRIPT" "$RUN_DIR" "$ITERATIONS"
echo ""

# Step 3: Render from training viewpoints
echo "--- Step 3/4: Rendering ---"
cd "$GS_REPO"
"$VENV_PYTHON" render.py -m "$RUN_DIR/offline"
echo ""

# Step 4: Compute metrics (copy train→test since no held-out set)
echo "--- Step 4/4: Computing metrics ---"
OFFLINE_DIR="$RUN_DIR/offline"
TEST_METHOD="$OFFLINE_DIR/test/ours_${ITERATIONS}"
TRAIN_METHOD="$OFFLINE_DIR/train/ours_${ITERATIONS}"

if [ -d "$TRAIN_METHOD" ] && [ -d "$TEST_METHOD" ]; then
    # Replace empty test dir with train renders
    rm -rf "$TEST_METHOD"
    cp -r "$TRAIN_METHOD" "$TEST_METHOD"
fi

"$VENV_PYTHON" metrics.py -m "$OFFLINE_DIR"

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Model:   $OFFLINE_DIR/point_cloud/iteration_${ITERATIONS}/"
echo "  Renders: $OFFLINE_DIR/train/ours_${ITERATIONS}/renders/"
echo "============================================"
