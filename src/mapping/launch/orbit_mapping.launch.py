"""Autonomous orbit mapping: two-altitude orbit + ORB-SLAM3 + keyframe capture.

Launches:
  1. ros_gz_bridge (camera + gimbal topics)
  2. Offboard orbit mapper (takeoff, orbit at 1.5m/-10°, orbit at 3m/-30°, land)
  3. ORB-SLAM3 RGBD (delayed 15s)

Saves images + depth + transforms.json for Gaussian Splatting.
Requires PX4 SITL + micro-XRCE-DDS-Agent running separately.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    home = os.path.expanduser('~')
    zed_ws = os.path.join(home, 'workspaces', 'zed_ws')

    vocab_path = os.path.join(zed_ws, 'ORB_SLAM3', 'Vocabulary', 'ORBvoc.txt')
    mapping_share = get_package_share_directory('mapping')
    config_path = os.path.join(mapping_share, 'config', 'gz_rgbd.yaml')
    orbit_script = os.path.join(mapping_share, 'scripts', 'orbit', 'orbit_mapper.py')

    gsplat_share = get_package_share_directory('gsplat')
    bridge_cfg = os.path.join(gsplat_share, 'config', 'bridge.yaml')

    orbslam3_lib = os.path.join(zed_ws, 'ORB_SLAM3', 'lib')
    dbow2_lib = os.path.join(zed_ws, 'ORB_SLAM3', 'Thirdparty', 'DBoW2', 'lib')
    g2o_lib = os.path.join(zed_ws, 'ORB_SLAM3', 'Thirdparty', 'g2o', 'lib')
    pangolin_lib = os.path.join(zed_ws, 'Pangolin', 'build', 'src')
    extra_ld = ':'.join([orbslam3_lib, dbow2_lib, g2o_lib, pangolin_lib])
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    full_ld = extra_ld + ':' + current_ld if current_ld else extra_ld

    return LaunchDescription([
        DeclareLaunchArgument('rock_x', default_value='0.0'),
        DeclareLaunchArgument('rock_y', default_value='3.0'),
        DeclareLaunchArgument('hover_time', default_value='10.0'),
        DeclareLaunchArgument('settle_time', default_value='1.5'),

        SetEnvironmentVariable('LD_LIBRARY_PATH', full_ld),
        SetEnvironmentVariable('PYTHONNOUSERSITE', '1'),

        # ros_gz_bridge
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                '--ros-args', '-p', f'config_file:={bridge_cfg}',
            ],
            output='screen',
        ),

        # Orbit mapper
        ExecuteProcess(
            cmd=[
                'python3', '-s', orbit_script,
                '--ros-args',
                '-p', ['rock_x:=', LaunchConfiguration('rock_x')],
                '-p', ['rock_y:=', LaunchConfiguration('rock_y')],
                '-p', ['hover_time:=', LaunchConfiguration('hover_time')],
                '-p', ['settle_time:=', LaunchConfiguration('settle_time')],
            ],
            output='screen',
        ),

        # ORB-SLAM3 RGBD — delayed 15s
        TimerAction(
            period=15.0,
            actions=[
                Node(
                    package='orbslam3',
                    executable='rgbd',
                    name='orbslam3_rgbd',
                    output='screen',
                    arguments=[vocab_path, config_path],
                    remappings=[
                        ('camera/rgb', '/rgbd/image'),
                        ('camera/depth', '/rgbd/depth'),
                    ],
                    additional_env={'LD_LIBRARY_PATH': full_ld},
                ),
            ],
        ),
    ])
