"""Active reconstruction: orbit + live 3DGS + splat viewer + voxel coverage.

Launches:
  1. ros_gz_bridge (camera + gimbal topics)
  2. Static TF publisher (map frame)
  3. Active mapper node (orbit flight + live 3DGS + viewer + RViz2)
  4. ORB-SLAM3 RGBD (delayed 15s, optional)
  5. RViz2 with active_recon config

Requires PX4 SITL + micro-XRCE-DDS-Agent running separately.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, Shutdown,
    SetEnvironmentVariable, TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    home = os.path.expanduser('~')
    gs_ws = os.path.join(home, 'workspaces', 'gs_ws')
    zed_ws = os.path.join(home, 'workspaces', 'zed_ws')
    venv_python = os.path.join(gs_ws, 'gsplat', 'bin', 'python3')

    mapping_share = get_package_share_directory('mapping')
    gsplat_share = get_package_share_directory('gsplat')
    bridge_cfg = os.path.join(gsplat_share, 'config', 'bridge.yaml')

    active_mapper_script = os.path.join(
        mapping_share, 'scripts', 'active_recon', 'active_mapper_node.py')
    rviz_config = os.path.join(
        mapping_share, 'config', 'active_recon.rviz')

    # ORB-SLAM3 library paths
    vocab_path = os.path.join(
        zed_ws, 'ORB_SLAM3', 'Vocabulary', 'ORBvoc.txt')
    orbslam3_config = os.path.join(mapping_share, 'config', 'gz_rgbd.yaml')
    orbslam3_lib = os.path.join(zed_ws, 'ORB_SLAM3', 'lib')
    dbow2_lib = os.path.join(
        zed_ws, 'ORB_SLAM3', 'Thirdparty', 'DBoW2', 'lib')
    g2o_lib = os.path.join(
        zed_ws, 'ORB_SLAM3', 'Thirdparty', 'g2o', 'lib')
    pangolin_lib = os.path.join(zed_ws, 'Pangolin', 'build', 'src')
    extra_ld = ':'.join([orbslam3_lib, dbow2_lib, g2o_lib, pangolin_lib])
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    full_ld = extra_ld + ':' + current_ld if current_ld else extra_ld

    return LaunchDescription([
        DeclareLaunchArgument('rock_x', default_value='0.0'),
        DeclareLaunchArgument('rock_y', default_value='8.0'),
        DeclareLaunchArgument('rock_z_up', default_value='0.8'),
        DeclareLaunchArgument('bbox_size', default_value='2.0'),
        DeclareLaunchArgument('orbit_waypoints', default_value='24'),
        DeclareLaunchArgument('orbit_radius', default_value='2.0'),
        DeclareLaunchArgument('orbit_altitude_3', default_value='0.5'),
        DeclareLaunchArgument('gimbal_pitch_3', default_value='-5.0'),
        DeclareLaunchArgument('iters_per_keyframe', default_value='500'),
        DeclareLaunchArgument('window_size', default_value='10'),
        DeclareLaunchArgument('pts_per_frame', default_value='40000'),
        DeclareLaunchArgument('max_gaussians', default_value='200000'),
        DeclareLaunchArgument('enable_viewer', default_value='true'),
        DeclareLaunchArgument('enable_orbslam3', default_value='false'),

        SetEnvironmentVariable('TORCH_CUDA_ARCH_LIST', '8.9'),
        SetEnvironmentVariable('LD_LIBRARY_PATH', full_ld),
        SetEnvironmentVariable('PYTHONNOUSERSITE', '1'),

        # Static TF: map frame
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'tf2_ros', 'static_transform_publisher',
                '--x', '0', '--y', '0', '--z', '0',
                '--roll', '0', '--pitch', '0', '--yaw', '0',
                '--frame-id', 'map', '--child-frame-id', 'base_link',
            ],
            output='screen',
        ),

        # ros_gz_bridge
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                '--ros-args', '-p', f'config_file:={bridge_cfg}',
            ],
            output='screen',
        ),

        # Active mapper node (venv python for torch/gsplat)
        ExecuteProcess(
            cmd=[
                venv_python, '-s', active_mapper_script,
                '--ros-args',
                '-p', ['rock_x:=', LaunchConfiguration('rock_x')],
                '-p', ['rock_y:=', LaunchConfiguration('rock_y')],
                '-p', ['rock_z_up:=', LaunchConfiguration('rock_z_up')],
                '-p', ['bbox_size:=', LaunchConfiguration('bbox_size')],
                '-p', ['orbit_waypoints:=',
                       LaunchConfiguration('orbit_waypoints')],
                '-p', ['orbit_radius:=',
                       LaunchConfiguration('orbit_radius')],
                '-p', ['orbit_altitude_3:=',
                       LaunchConfiguration('orbit_altitude_3')],
                '-p', ['gimbal_pitch_3:=',
                       LaunchConfiguration('gimbal_pitch_3')],
                '-p', ['iters_per_keyframe:=',
                       LaunchConfiguration('iters_per_keyframe')],
                '-p', ['window_size:=',
                       LaunchConfiguration('window_size')],
                '-p', ['pts_per_frame:=',
                       LaunchConfiguration('pts_per_frame')],
                '-p', ['max_gaussians:=',
                       LaunchConfiguration('max_gaussians')],
                '-p', ['enable_viewer:=',
                       LaunchConfiguration('enable_viewer')],
            ],
            output='screen',
            on_exit=Shutdown(),
        ),

        # RViz2
        ExecuteProcess(
            cmd=['rviz2', '-d', rviz_config],
            output='screen',
        ),

        # ORB-SLAM3 RGBD — delayed 15s (off by default)
        TimerAction(
            period=15.0,
            condition=IfCondition(LaunchConfiguration('enable_orbslam3')),
            actions=[
                Node(
                    package='orbslam3',
                    executable='rgbd',
                    name='orbslam3_rgbd',
                    output='screen',
                    arguments=[vocab_path, orbslam3_config],
                    remappings=[
                        ('camera/rgb', '/rgbd/image'),
                        ('camera/depth', '/rgbd/depth'),
                    ],
                    additional_env={'LD_LIBRARY_PATH': full_ld},
                ),
            ],
        ),
    ])
