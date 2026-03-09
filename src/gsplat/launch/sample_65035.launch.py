#!/usr/bin/env python3
"""Launch bridge and gimbal control for lunar sample 65035."""

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory('gsplat')
    bridge_cfg = os.path.join(pkg, 'config', 'bridge.yaml')
    gimbal_script = os.path.join(pkg, 'scripts', 'gimbal_control.py')

    gz_bridge = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '--ros-args', '-p', f'config_file:={bridge_cfg}',
        ],
        output='screen',
    )

    gimbal_gui = ExecuteProcess(
        cmd=['python3', gimbal_script],
        output='screen',
    )

    return LaunchDescription([gz_bridge, gimbal_gui])
