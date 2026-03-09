"""ORB-SLAM3 RGBD with PX4 gimbal drone in Gazebo.

Launches:
  1. ros_gz_bridge (camera topics + gimbal commands)
  2. gimbal_control GUI
  3. ORB-SLAM3 RGBD node with Pangolin viewer

Requires PX4 SITL + micro-XRCE-DDS-Agent running separately.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    home = os.path.expanduser('~')
    zed_ws = os.path.join(home, 'workspaces', 'zed_ws')

    # ORB-SLAM3 vocabulary
    vocab_path = os.path.join(zed_ws, 'ORB_SLAM3', 'Vocabulary', 'ORBvoc.txt')

    # ORB-SLAM3 config for Gazebo RGBD camera
    mapping_share = get_package_share_directory('mapping')
    config_path = os.path.join(mapping_share, 'config', 'gz_rgbd.yaml')

    # gsplat package paths
    gsplat_share = get_package_share_directory('gsplat')
    bridge_cfg = os.path.join(gsplat_share, 'config', 'bridge.yaml')
    gimbal_script = os.path.join(gsplat_share, 'scripts', 'gimbal_control.py')

    # LD_LIBRARY_PATH for ORB-SLAM3 and Pangolin
    orbslam3_lib = os.path.join(zed_ws, 'ORB_SLAM3', 'lib')
    dbow2_lib = os.path.join(zed_ws, 'ORB_SLAM3', 'Thirdparty', 'DBoW2', 'lib')
    g2o_lib = os.path.join(zed_ws, 'ORB_SLAM3', 'Thirdparty', 'g2o', 'lib')
    pangolin_lib = os.path.join(zed_ws, 'Pangolin', 'build', 'src')
    extra_ld = ':'.join([orbslam3_lib, dbow2_lib, g2o_lib, pangolin_lib])
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    full_ld = extra_ld + ':' + current_ld if current_ld else extra_ld

    return LaunchDescription([
        SetEnvironmentVariable('LD_LIBRARY_PATH', full_ld),
        SetEnvironmentVariable('PYTHONNOUSERSITE', '1'),

        # ros_gz_bridge for camera + gimbal topics
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                '--ros-args', '-p', f'config_file:={bridge_cfg}',
            ],
            output='screen',
        ),

        # Gimbal control GUI
        ExecuteProcess(
            cmd=['python3', gimbal_script],
            output='screen',
        ),

        # ORB-SLAM3 RGBD node
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
    ])
