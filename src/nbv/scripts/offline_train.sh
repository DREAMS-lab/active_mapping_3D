#!/usr/bin/env bash
# Offline 3DGS training using the gaussian-splatting submodule (Kerbl et al.)
#
# 1. Converts transforms.json to COLMAP text format (cameras.txt, images.txt, points3D.txt)
# 2. Runs gaussian-splatting train.py at full 1280×960 for 30K iterations
#
# Usage:
#   bash offline_train.sh <run_dir> [iterations]
#
# Example:
#   bash offline_train.sh <workspace>/data/nbv/run_001
#   bash offline_train.sh <workspace>/data/nbv/run_001 50000

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_dir> [iterations]"
    echo "  run_dir:    path to run directory containing transforms.json and images/"
    echo "  iterations: number of training iterations (default: 30000)"
    exit 1
fi

RUN_DIR="$(realpath "$1")"
ITERATIONS="${2:-30000}"

# Resolve paths — derive workspace root from script install location
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
GS_WS="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
VENV_PYTHON="$GS_WS/venv/bin/python3"
POSES_SCRIPT="$SCRIPT_DIR/poses_to_colmap.py"
GS_REPO="$GS_WS/repos/gaussian-splatting"

# Validate
if [ ! -f "$RUN_DIR/transforms.json" ]; then
    echo "Error: $RUN_DIR/transforms.json not found"
    exit 1
fi

if [ ! -d "$RUN_DIR/images" ]; then
    echo "Error: $RUN_DIR/images/ not found"
    exit 1
fi

if [ ! -f "$POSES_SCRIPT" ]; then
    echo "Error: poses_to_colmap.py not found at $POSES_SCRIPT"
    exit 1
fi

if [ ! -f "$GS_REPO/train.py" ]; then
    echo "Error: gaussian-splatting train.py not found at $GS_REPO/train.py"
    exit 1
fi

echo "=== Offline 3DGS Training ==="
echo "  Run dir:    $RUN_DIR"
echo "  Iterations: $ITERATIONS"
echo "  Resolution: full (1280x960, -r 1)"
echo ""

# Step 1: Convert poses to COLMAP format
echo "--- Step 1: Converting poses to COLMAP format ---"
"$VENV_PYTHON" "$POSES_SCRIPT" "$RUN_DIR"
echo ""

# Step 2: Train gaussian-splatting
echo "--- Step 2: Training gaussian-splatting ---"
OUTPUT_DIR="$RUN_DIR/offline"
mkdir -p "$OUTPUT_DIR"

cd "$GS_REPO"
# Use depth supervision if inverse-depth images exist
DEPTH_FLAG=""
if [ -d "$RUN_DIR/depth_inv" ] && [ "$(ls -A "$RUN_DIR/depth_inv" 2>/dev/null)" ]; then
    echo "  Depth supervision: enabled (depth_inv/)"
    DEPTH_FLAG="--depths depth_inv"
else
    echo "  Depth supervision: disabled (no depth_inv/ found)"
fi

"$VENV_PYTHON" train.py \
    -s "$RUN_DIR" \
    -m "$OUTPUT_DIR" \
    -r 1 \
    --iterations "$ITERATIONS" \
    $DEPTH_FLAG

echo ""
echo "=== Offline training complete ==="
echo "  Model saved to: $OUTPUT_DIR"
echo "  Run 'python render.py -m $OUTPUT_DIR' to render images"
