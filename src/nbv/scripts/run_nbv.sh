#!/usr/bin/env bash
# Launch full NBV pipeline in 3 separate Konsole tabs
# Usage: ./run_nbv.sh [launch args...]
# Example: ./run_nbv.sh rock_x:=0.0 rock_y:=8.0

set -eo pipefail

PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
# Derive workspace root from script location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GS_WS="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
LAUNCH_ARGS="${*:-}"

# ── T1: PX4 SITL ──
konsole --new-tab -e bash -c "
    echo '=== PX4 SITL (sample_15016) ==='
    cd '$PX4_DIR'
    PX4_GZ_NO_FOLLOW=1 PX4_GZ_WORLD=sample_15016 make px4_sitl gz_px4_gsplat
    echo 'PX4 exited. Press Enter to close.'
    read
" &

echo "Started PX4 SITL tab. Waiting 15s for Gazebo..."
sleep 15

# ── T2: MicroXRCE-DDS Agent ──
konsole --new-tab -e bash -c "
    echo '=== MicroXRCE-DDS Agent ==='
    MicroXRCEAgent udp4 -p 8888
    echo 'DDS agent exited. Press Enter to close.'
    read
" &

echo "Started DDS agent tab. Waiting 5s..."
sleep 5

# ── T3: ROS2 NBV launch ──
konsole --new-tab -e bash -c "
    echo '=== NBV Reconstruction ==='
    echo 'Sourcing ROS2...'
    set +u
    source /opt/ros/jazzy/setup.bash
    echo 'Sourcing workspace...'
    source '$GS_WS/install/setup.bash'
    set -u
    echo 'Launching NBV...'
    ros2 launch nbv nbv_recon.launch.py $LAUNCH_ARGS
    echo 'NBV exited. Press Enter to close.'
    read
" &

echo ""
echo "All 3 terminals launched."
echo "Close this terminal or press Ctrl+C — the Konsole tabs keep running."
wait
