#!/bin/bash
# COLMAP sparse reconstruction for gaussian splatting
# Usage: ./colmap.sh <run_number> [--hd]
#   --hd  use 1080p intrinsics (default: 640x480 dev mode)

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <run_number> [--hd]"
    exit 1
fi

RUN_NUM=$(printf "%03d" "$1")
WS="$HOME/workspaces/gs_ws"
RUN="$WS/data/mapping/run_${RUN_NUM}"

if [ ! -d "$RUN/images" ]; then
    echo "Error: $RUN/images not found"
    exit 1
fi

# Camera intrinsics
if [ "$2" = "--hd" ]; then
    PARAMS="1397.2235,1397.2235,960.0,540.0"
    echo "Using 1080p intrinsics"
else
    PARAMS="465.7412,465.7412,320.0,240.0"
    echo "Using 640x480 intrinsics"
fi

NUM_IMAGES=$(ls "$RUN/images/"*.png 2>/dev/null | wc -l)
echo "=== COLMAP for run_${RUN_NUM} ($NUM_IMAGES images) ==="

# Clean previous COLMAP data if re-running
rm -f "$RUN/colmap.db"
rm -rf "$RUN/sparse"

echo ""
echo "--- Feature Extraction ---"
colmap feature_extractor \
    --database_path "$RUN/colmap.db" \
    --image_path "$RUN/images" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_params "$PARAMS"

echo ""
echo "--- Sequential Matching ---"
colmap sequential_matcher \
    --database_path "$RUN/colmap.db"

echo ""
echo "--- Mapper (sparse reconstruction) ---"
mkdir -p "$RUN/sparse"
colmap mapper \
    --database_path "$RUN/colmap.db" \
    --image_path "$RUN/images" \
    --output_path "$RUN/sparse"

# Export to text format
if [ -d "$RUN/sparse/0" ]; then
    echo ""
    echo "--- Exporting to text format ---"
    mkdir -p "$RUN/sparse/0/txt"
    colmap model_converter \
        --input_path "$RUN/sparse/0" \
        --output_path "$RUN/sparse/0/txt" \
        --output_type TXT

    REGISTERED=$(grep -c "^[^#]" "$RUN/sparse/0/txt/images.txt" | awk '{print int($1/2)}')
    POINTS=$(grep -c "^[^#]" "$RUN/sparse/0/txt/points3D.txt")
    echo ""
    echo "=== Done ==="
    echo "  Registered images: $REGISTERED / $NUM_IMAGES"
    echo "  Sparse points:     $POINTS"
    echo "  Output:            $RUN/sparse/0/"
    echo ""
    echo "Next: train gaussian splat"
    echo "  source gsplat/bin/activate && cd gaussian-splatting"
    echo "  python train.py -s ../$RUN -r 1 --iterations 30000 -m ../data/splats/run_${RUN_NUM}"
else
    echo ""
    echo "ERROR: mapper failed"
    exit 1
fi
