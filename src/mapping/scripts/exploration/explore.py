#!/usr/bin/env python3
"""Exploration flight: fly an expanding spiral to discover objects.

No prior knowledge of object positions. The drone takes off, spirals outward
at a fixed altitude while looking down, capturing RGB+depth frames continuously.
ORB-SLAM3 runs alongside to build a sparse 3D map.

After landing, use detect_roi.py to find objects in the accumulated point cloud.

Usage:
  ros2 run mapping explore
  ros2 run mapping explore --ros-args -p altitude:=2.0 -p max_radius:=6.0
"""

import math
import os
import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2

# Camera intrinsics (640x480 for dev speed)
FX = 465.7412
FY = 465.7412
CX = 320.0
CY = 240.0
W = 640
H = 480


class Explorer(Node):
    def __init__(self):
        super().__init__('explorer')

        self.declare_parameter('max_altitude', 5.0)
        self.declare_parameter('min_altitude', 2.0)
        self.declare_parameter('max_radius', 10.0)
        self.declare_parameter('spiral_spacing', 1.0)
        self.declare_parameter('points_per_loop', 36)
        self.declare_parameter('gimbal_pitch_deg', -45.0)
        self.declare_parameter('settle_time', 0.5)
        self.declare_parameter('save_interval', 3)  # save every Nth waypoint
        self.declare_parameter('data_dir', '')
        self.declare_parameter('obstacle_dist', 1.5)  # repulsive field trigger (m)
        self.declare_parameter('repulsive_gain', 2.0)  # max repulsive offset (m)

        self.max_alt = self.get_parameter('max_altitude').value
        self.min_alt = self.get_parameter('min_altitude').value
        self.max_radius = self.get_parameter('max_radius').value
        self.spiral_spacing = self.get_parameter('spiral_spacing').value
        self.points_per_loop = int(self.get_parameter('points_per_loop').value)
        self.gimbal_pitch_rad = math.radians(self.get_parameter('gimbal_pitch_deg').value)
        self.settle_time = self.get_parameter('settle_time').value
        self.save_interval = int(self.get_parameter('save_interval').value)
        self.obstacle_dist = self.get_parameter('obstacle_dist').value
        self.repulsive_gain = self.get_parameter('repulsive_gain').value

        # Build spiral waypoints
        self.waypoints = self._build_spiral()
        self.wp_idx = 0

        # Output directory
        base = self.get_parameter('data_dir').value
        if not base:
            # Default: <ws>/data/mapping (derive workspace root from install path)
            ws_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
            base = os.path.join(ws_root, 'data', 'mapping')
        os.makedirs(base, exist_ok=True)
        run_num = 1
        while os.path.exists(os.path.join(base, f'run_{run_num:03d}')):
            run_num += 1
        self.out_dir = os.path.join(base, f'run_{run_num:03d}')
        self.img_dir = os.path.join(self.out_dir, 'images')
        self.depth_dir = os.path.join(self.out_dir, 'depth')
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        self.get_logger().info(f'Spiral: {len(self.waypoints)} waypoints, '
                               f'max_r={self.max_radius}m, alt={self.min_alt}-{self.max_alt}m')
        self.get_logger().info(f'Output: {self.out_dir}')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.gimbal_pitch_pub = self.create_publisher(Float64, '/gimbal/pitch', 10)
        self.gimbal_yaw_pub = self.create_publisher(Float64, '/gimbal/yaw', 10)

        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self._pos_cb, qos)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self._status_cb, qos)
        self.create_subscription(Image, '/rgbd/image', self._rgb_cb, 10)
        self.create_subscription(Image, '/rgbd/depth', self._depth_cb, 10)

        self.bridge = CvBridge()
        self.pos = None
        self.status = None
        self.latest_rgb = None
        self.latest_depth = None
        self.phase = 'PREFLIGHT'
        self.counter = 0
        self.settle_start = None
        self.kf_count = 0
        self.frames = []
        self.saved = False

        self.timer = self.create_timer(0.05, self._loop)
        self.get_logger().info('Waiting for vehicle data...')

    def _build_spiral(self):
        """Archimedean spiral with varying altitude.

        Altitude decreases linearly from max_alt at center to min_alt at edge,
        giving close-up views of objects near the ground.
        Drone yaw faces OUTWARD so the gimbal camera scans ahead.
        """
        wps = []
        max_loops = self.max_radius / self.spiral_spacing
        total_points = int(max_loops * self.points_per_loop)
        for i in range(total_points):
            theta = 2.0 * math.pi * i / self.points_per_loop
            r = self.spiral_spacing * theta / (2.0 * math.pi)
            if r > self.max_radius:
                break
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            yaw = math.atan2(y, x)
            # Altitude varies: high at center, low at edge
            frac = r / self.max_radius if self.max_radius > 0 else 0
            alt = self.max_alt - frac * (self.max_alt - self.min_alt)
            wps.append((x, y, -abs(alt), yaw))
        return wps

    def _pos_cb(self, msg): self.pos = msg
    def _status_cb(self, msg): self.status = msg
    def _rgb_cb(self, msg): self.latest_rgb = msg
    def _depth_cb(self, msg): self.latest_depth = msg

    def _pub_offboard(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def _pub_setpoint(self, x, y, z, yaw=float('nan')):
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.velocity = [float('nan')] * 3
        msg.acceleration = [float('nan')] * 3
        msg.jerk = [float('nan')] * 3
        msg.yaw = yaw
        msg.yawspeed = float('nan')
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def _send_cmd(self, command, p1=0.0, p2=0.0, p7=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = p1
        msg.param2 = p2
        msg.param7 = p7
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)

    def _arm(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, p1=1.0)
        self.get_logger().info('ARM')

    def _offboard(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, p1=1.0, p2=6.0)
        self.get_logger().info('OFFBOARD')

    def _land(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('LAND')

    def _disarm(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, p1=0.0)
        self.get_logger().info('DISARM')

    def _pub_gimbal(self):
        self.gimbal_pitch_pub.publish(Float64(data=self.gimbal_pitch_rad))
        self.gimbal_yaw_pub.publish(Float64(data=0.0))

    def _save_keyframe(self):
        if self.latest_rgb is None or self.latest_depth is None:
            return

        idx = f'{self.kf_count:05d}'
        rgb = self.bridge.imgmsg_to_cv2(self.latest_rgb, 'bgr8')
        cv2.imwrite(os.path.join(self.img_dir, f'{idx}.png'), rgb)

        depth = self.bridge.imgmsg_to_cv2(self.latest_depth, 'passthrough')
        depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = (depth_clean * 1000.0).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(self.depth_dir, f'{idx}.png'), depth_mm)

        self.frames.append({
            'file_path': f'images/{idx}.png',
            'depth_path': f'depth/{idx}.png',
            'position_ned': [float(self.pos.x), float(self.pos.y), float(self.pos.z)],
            'heading': float(self.pos.heading),
        })
        self.kf_count += 1

    def _save_metadata(self):
        meta = {
            'camera_model': 'PINHOLE',
            'fl_x': FX, 'fl_y': FY, 'cx': CX, 'cy': CY, 'w': W, 'h': H,
            'frames': self.frames,
        }
        with open(os.path.join(self.out_dir, 'transforms.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    def _compute_repulsive_offset(self):
        """Depth-based repulsive potential field for obstacle avoidance.

        Looks at the depth image, finds the closest obstacle direction,
        and returns a (dx, dy, dz) NED offset pushing away from it.
        """
        if self.latest_depth is None or self.pos is None:
            return 0.0, 0.0, 0.0

        depth = self.bridge.imgmsg_to_cv2(self.latest_depth, 'passthrough')
        if depth is None:
            return 0.0, 0.0, 0.0

        # Use the top half of the image (forward-looking, not ground)
        h, w = depth.shape
        roi = depth[:h // 2, :]
        valid = np.isfinite(roi) & (roi > 0.1)
        if not valid.any():
            return 0.0, 0.0, 0.0

        min_depth = np.nanmin(roi[valid])
        if min_depth > self.obstacle_dist:
            return 0.0, 0.0, 0.0

        # Find the centroid of close pixels to determine obstacle direction
        close_mask = valid & (roi < self.obstacle_dist)
        if not close_mask.any():
            return 0.0, 0.0, 0.0

        vs, us = np.where(close_mask)
        u_mean = np.mean(us)
        # Direction in camera frame: obstacle is at (u_mean - CX) / FX
        # Camera X-right maps to world via yaw
        cam_x = (u_mean - CX) / FX  # +right, -left in camera

        # Repulsive strength: stronger when closer
        strength = self.repulsive_gain * (1.0 - min_depth / self.obstacle_dist)
        strength = min(strength, self.repulsive_gain)

        # Push AWAY from obstacle in horizontal plane
        # Camera frame: Z forward, X right -> in NED body frame depends on gimbal
        # Simple: push opposite to obstacle direction in yaw frame
        yaw = self.pos.heading
        # Obstacle is roughly "forward" in camera; push backward
        # Also push laterally away from the obstacle centroid
        dx_body = -strength  # push backward
        dy_body = -cam_x * strength  # push away laterally

        # Rotate body offset to NED
        cy, sy = math.cos(yaw), math.sin(yaw)
        dx_ned = cy * dx_body - sy * dy_body
        dy_ned = sy * dx_body + cy * dy_body
        dz_ned = -0.3 * strength  # also rise slightly (NED: negative = up)

        self.get_logger().warn(
            f'OBSTACLE at {min_depth:.1f}m — repulsive offset: '
            f'({dx_ned:.2f}, {dy_ned:.2f}, {dz_ned:.2f})')

        return dx_ned, dy_ned, dz_ned

    def _dist_to(self, tx, ty, tz):
        if self.pos is None:
            return 999.0
        return math.sqrt((self.pos.x - tx)**2 + (self.pos.y - ty)**2 + (self.pos.z - tz)**2)

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _loop(self):
        self._pub_offboard()
        if self.pos is None or self.status is None:
            return
        self._pub_gimbal()

        takeoff_alt = -abs(self.max_alt)

        if self.phase == 'PREFLIGHT':
            self._pub_setpoint(0.0, 0.0, takeoff_alt, 0.0)
            self.counter += 1
            if self.counter >= 40:
                self._offboard()
                self._arm()
                self.phase = 'TAKEOFF'
                self.get_logger().info(f'Takeoff to {self.max_alt}m')

        elif self.phase == 'TAKEOFF':
            self._pub_setpoint(0.0, 0.0, takeoff_alt, 0.0)
            if self._dist_to(0.0, 0.0, takeoff_alt) < 0.5:
                self.phase = 'SPIRAL'
                self.get_logger().info('Starting spiral exploration')

        elif self.phase == 'SPIRAL':
            x, y, z, yaw = self.waypoints[self.wp_idx]
            # Apply repulsive potential field for obstacle avoidance
            rx, ry, rz = self._compute_repulsive_offset()
            self._pub_setpoint(x + rx, y + ry, z + rz, yaw)

            if self._dist_to(x, y, z) < 0.5:
                if self.settle_start is None:
                    self.settle_start = self._now_sec()
                if (self._now_sec() - self.settle_start) >= self.settle_time:
                    if self.wp_idx % self.save_interval == 0:
                        self._save_keyframe()
                        self.get_logger().info(
                            f'KF {self.kf_count} wp={self.wp_idx+1}/{len(self.waypoints)}')
                    self.wp_idx += 1
                    self.settle_start = None

                    if self.wp_idx >= len(self.waypoints):
                        self.phase = 'RETURN'
                        self.get_logger().info('Spiral done, returning home')
            else:
                self.settle_start = None

        elif self.phase == 'RETURN':
            self._pub_setpoint(0.0, 0.0, -1.0, 0.0)
            if self._dist_to(0.0, 0.0, -1.0) < 1.0:
                self._land()
                self.phase = 'LANDING'

        elif self.phase == 'LANDING':
            if self.pos.z > -0.3:
                self._disarm()
                self._save_metadata()
                self.get_logger().info(f'=== SAVED {self.kf_count} frames to {self.out_dir} ===')
                raise SystemExit(0)


def main():
    rclpy.init()
    node = Explorer()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
