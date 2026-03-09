"""Manual flight + gimbal GUI + ORB-SLAM3.

Use the GUI to control altitude and gimbal, while ORB-SLAM3 maps.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    home = os.path.expanduser('~')
    zed_ws = os.path.join(home, 'workspaces', 'zed_ws')

    vocab_path = os.path.join(zed_ws, 'ORB_SLAM3', 'Vocabulary', 'ORBvoc.txt')
    mapping_share = get_package_share_directory('mapping')
    config_path = os.path.join(mapping_share, 'config', 'gz_rgbd.yaml')
    flight_script = os.path.join(mapping_share, 'scripts', 'manual_flight.py')

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

        # Manual flight + gimbal GUI (auto-arms and takes off to 2m)
        ExecuteProcess(
            cmd=['python3', '-s', flight_script],
            output='screen',
        ),

        # ORB-SLAM3 RGBD — delayed 10s for takeoff
        TimerAction(
            period=10.0,
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
