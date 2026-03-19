"""Pose-uncertainty-aware NBV reconstruction for active 3DGS.

Extends the NBV system with per-keyframe EKF position variance
for uncertainty-weighted training and pose-aware viewpoint scoring.

Launches:
  1. ros_gz_bridge (camera + gimbal topics)
  2. Static TF publisher (map frame)
  3. PU active mapper node (seed + iterative NBV scoring + pose-weighted training)
  4. RViz2 with active_recon config

Requires PX4 SITL + micro-XRCE-DDS-Agent running separately.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, Shutdown,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pu_share = get_package_share_directory('nbv_pos_uncert')
    # Derive workspace root: <ws>/install/nbv_pos_uncert/share/nbv_pos_uncert -> <ws>
    ws_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(pu_share))))
    venv_python = os.path.join(ws_root, 'venv', 'bin', 'python3')
    gsplat_share = get_package_share_directory('gsplat')
    mapping_share = get_package_share_directory('mapping')
    bridge_cfg = os.path.join(gsplat_share, 'config', 'bridge.yaml')

    active_mapper_script = os.path.join(
        pu_share, 'scripts', 'active_mapper_node_pu.py')
    rviz_config = os.path.join(
        mapping_share, 'config', 'active_recon.rviz')

    return LaunchDescription([
        # -- Rock / scene parameters --
        DeclareLaunchArgument('rock_x', default_value='0.0'),
        DeclareLaunchArgument('rock_y', default_value='8.0'),
        DeclareLaunchArgument('rock_z_up', default_value='0.8'),
        DeclareLaunchArgument('bbox_size', default_value='2.0'),

        # -- Seed orbit parameters --
        DeclareLaunchArgument('orbit_altitude_1', default_value='1.0'),
        DeclareLaunchArgument('gimbal_pitch_1', default_value='-25.0'),
        DeclareLaunchArgument('orbit_radius', default_value='2.5'),

        # -- Fully adaptive NBV parameters --
        DeclareLaunchArgument('kf_budget', default_value='48'),
        DeclareLaunchArgument('seed_kfs', default_value='4'),
        DeclareLaunchArgument('batch_size', default_value='4'),
        DeclareLaunchArgument('nbv_k', default_value='32'),
        DeclareLaunchArgument('nbv_n_azimuth', default_value='12'),
        DeclareLaunchArgument('skip_adaptive', default_value='false'),

        # -- Offline reconstruction parameters --
        DeclareLaunchArgument('offline_iters', default_value='30000'),
        DeclareLaunchArgument('offline_max_gaussians', default_value='1000000'),
        DeclareLaunchArgument('offline_train_scale', default_value='2'),

        # -- 3DGS training parameters --
        DeclareLaunchArgument('iters_per_keyframe', default_value='2000'),
        DeclareLaunchArgument('window_size', default_value='0'),
        DeclareLaunchArgument('pts_per_frame', default_value='40000'),
        DeclareLaunchArgument('max_gaussians', default_value='500000'),

        # -- Viewer --
        DeclareLaunchArgument('enable_viewer', default_value='true'),

        # -- Environment --
        SetEnvironmentVariable('TORCH_CUDA_ARCH_LIST', '8.9'),
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

        # ros_gz_bridge — sigkill_on_stop prevents zombie bridges
        # that accumulate across Ctrl+C restarts
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                '--ros-args', '-p', f'config_file:={bridge_cfg}',
            ],
            output='screen',
            sigterm_timeout='3',
            sigkill_timeout='5',
        ),

        # NBV active mapper node (venv python for torch/gsplat)
        ExecuteProcess(
            cmd=[
                venv_python, '-s', active_mapper_script,
                '--ros-args',
                '-p', ['rock_x:=', LaunchConfiguration('rock_x')],
                '-p', ['rock_y:=', LaunchConfiguration('rock_y')],
                '-p', ['rock_z_up:=', LaunchConfiguration('rock_z_up')],
                '-p', ['bbox_size:=', LaunchConfiguration('bbox_size')],
                '-p', ['orbit_altitude_1:=',
                       LaunchConfiguration('orbit_altitude_1')],
                '-p', ['gimbal_pitch_1:=',
                       LaunchConfiguration('gimbal_pitch_1')],
                '-p', ['orbit_radius:=',
                       LaunchConfiguration('orbit_radius')],
                '-p', ['kf_budget:=', LaunchConfiguration('kf_budget')],
                '-p', ['seed_kfs:=', LaunchConfiguration('seed_kfs')],
                '-p', ['batch_size:=', LaunchConfiguration('batch_size')],
                '-p', ['nbv_k:=', LaunchConfiguration('nbv_k')],
                '-p', ['nbv_n_azimuth:=',
                       LaunchConfiguration('nbv_n_azimuth')],
                '-p', ['skip_adaptive:=',
                       LaunchConfiguration('skip_adaptive')],
                '-p', ['offline_iters:=',
                       LaunchConfiguration('offline_iters')],
                '-p', ['offline_max_gaussians:=',
                       LaunchConfiguration('offline_max_gaussians')],
                '-p', ['offline_train_scale:=',
                       LaunchConfiguration('offline_train_scale')],
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
    ])
