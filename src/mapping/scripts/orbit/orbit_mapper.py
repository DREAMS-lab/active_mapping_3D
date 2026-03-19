#!/usr/bin/env python3
"""Offboard orbit around rock at two altitudes, save data for Gaussian Splatting.

Trajectory:
  1. Takeoff to 1m, orbit rock with gimbal at -10° pitch
  2. Rise to 3m, orbit rock with gimbal at -30° pitch
  3. Return home, land, disarm, save all data

Rock at SDF (8, 0, 0.8) = NED (0, 8, -0.8).
Orbit radius = distance from spawn to rock.

Output: data/mapping/run_NNN/
  images/     - RGB PNGs
  depth/      - 16-bit depth PNGs (mm)
  transforms.json - camera intrinsics + per-frame camera-to-world poses
  orbslam3/   - ORB-SLAM3 trajectory files (copied after landing)
"""

import json
import math
import os
import shutil
import glob as globmod

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

FX = 465.7412
FY = 465.7412
CX = 320.0
CY = 240.0
W = 640
H = 480


class OrbitMapper(Node):
    def __init__(self):
        super().__init__('orbit_mapper')

        self.declare_parameter('rock_x', 0.0)
        self.declare_parameter('rock_y', 8.0)
        self.declare_parameter('rock_z_up', 0.8)
        self.declare_parameter('num_waypoints', 36)
        self.declare_parameter('settle_time', 0.8)
        self.declare_parameter('hover_time', 10.0)
        self.declare_parameter('data_dir', '')

        self.rock_x = self.get_parameter('rock_x').value
        self.rock_y = self.get_parameter('rock_y').value
        self.rock_z_up = self.get_parameter('rock_z_up').value
        self.radius = math.sqrt(self.rock_x**2 + self.rock_y**2)
        self.n_wp = int(self.get_parameter('num_waypoints').value)
        self.settle_time = self.get_parameter('settle_time').value
        self.hover_time = self.get_parameter('hover_time').value

        # Two passes: (altitude_m, gimbal_pitch_deg)
        self.passes = [
            (1.0, -10.0),
            (3.0, -30.0),
        ]
        self.current_pass = 0
        self._setup_pass(0)

        # Output directory: data/mapping/run_NNN
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
        self.slam_dir = os.path.join(self.out_dir, 'orbslam3')
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.slam_dir, exist_ok=True)

        self.get_logger().info(f'Rock at NED ({self.rock_x}, {self.rock_y}), radius={self.radius:.1f}m')
        self.get_logger().info(f'Passes: {self.passes}')
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
        self.hover_start = None
        self.kf_count = 0
        self.frames = []
        self.saved = False

        self.timer = self.create_timer(0.05, self._loop)
        self.get_logger().info('Waiting for vehicle data...')

    def _setup_pass(self, idx):
        alt, pitch_deg = self.passes[idx]
        self.alt_ned = -abs(alt)
        self.gimbal_pitch_rad = math.radians(pitch_deg)
        self.waypoints = self._build_waypoints()
        self.wp_idx = 0
        self.settle_start = None

    def _build_waypoints(self):
        wps = []
        for i in range(self.n_wp):
            angle = 2.0 * math.pi * i / self.n_wp
            x = self.rock_x + self.radius * math.cos(angle)
            y = self.rock_y + self.radius * math.sin(angle)
            z = self.alt_ned
            yaw = math.atan2(self.rock_y - y, self.rock_x - x)
            wps.append((x, y, z, yaw))
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

    def _disarm(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, p1=0.0)
        self.get_logger().info('DISARM')

    def _offboard(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, p1=1.0, p2=6.0)
        self.get_logger().info('OFFBOARD')

    def _land(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('LAND')

    def _pub_gimbal(self):
        self.gimbal_pitch_pub.publish(Float64(data=self.gimbal_pitch_rad))
        self.gimbal_yaw_pub.publish(Float64(data=0.0))

    # ── Camera pose ───────────────────────────────────────

    def _camera_to_world(self, x, y, z, yaw, gimbal_pitch):
        """4x4 camera-to-world from drone NED pose + gimbal pitch."""
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(gimbal_pitch), math.sin(gimbal_pitch)

        # Rz(yaw) @ Ry(gimbal_pitch) @ R_ned_to_opencv
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_ned2cv = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

        R = Rz @ Ry @ R_ned2cv
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    # ── Save ──────────────────────────────────────────────

    def _save_keyframe(self, x, y, z, yaw):
        if self.latest_rgb is None or self.latest_depth is None:
            self.get_logger().warn('No image data, skipping keyframe')
            return

        idx = f'{self.kf_count:05d}'

        rgb = self.bridge.imgmsg_to_cv2(self.latest_rgb, 'bgr8')
        cv2.imwrite(os.path.join(self.img_dir, f'{idx}.png'), rgb)

        depth = self.bridge.imgmsg_to_cv2(self.latest_depth, 'passthrough')
        depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = (depth_clean * 1000.0).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(self.depth_dir, f'{idx}.png'), depth_mm)

        c2w = self._camera_to_world(x, y, z, yaw, self.gimbal_pitch_rad)
        self.frames.append({
            'file_path': f'images/{idx}.png',
            'depth_path': f'depth/{idx}.png',
            'transform_matrix': c2w.tolist(),
            'position_ned': [float(x), float(y), float(z)],
            'yaw_rad': float(yaw),
            'gimbal_pitch_rad': float(self.gimbal_pitch_rad),
        })
        self.kf_count += 1

        # Save transforms.json after every keyframe so it's always up to date
        self._write_transforms()

    def _write_transforms(self):
        transforms = {
            'camera_model': 'PINHOLE',
            'fl_x': FX, 'fl_y': FY,
            'cx': CX, 'cy': CY,
            'w': W, 'h': H,
            'frames': self.frames,
        }
        path = os.path.join(self.out_dir, 'transforms.json')
        with open(path, 'w') as f:
            json.dump(transforms, f, indent=2)

    def _save_all(self):
        if self.saved:
            return

        # Final transforms.json
        self._write_transforms()

        # Copy ORB-SLAM3 output files
        cwd = os.getcwd()
        for fname in ['KeyFrameTrajectory.txt', 'CameraTrajectory.txt']:
            src = os.path.join(cwd, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(self.slam_dir, fname))
                self.get_logger().info(f'Copied {fname} to {self.slam_dir}')

        self.saved = True
        self.get_logger().info(f'=== SAVED {self.kf_count} frames to {self.out_dir} ===')

    # ── Helpers ───────────────────────────────────────────

    def _dist_to(self, tx, ty, tz):
        if self.pos is None:
            return 999.0
        return math.sqrt((self.pos.x - tx)**2 + (self.pos.y - ty)**2 + (self.pos.z - tz)**2)

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ── Main loop ─────────────────────────────────────────

    def _loop(self):
        self._pub_offboard()
        if self.pos is None or self.status is None:
            return

        self._pub_gimbal()

        # ── PREFLIGHT ──
        if self.phase == 'PREFLIGHT':
            yaw = math.atan2(self.rock_y, self.rock_x)
            self._pub_setpoint(0.0, 0.0, self.alt_ned, yaw)
            self.counter += 1
            if self.counter >= 40:
                self._offboard()
                self._arm()
                self.phase = 'TAKEOFF'
                self.get_logger().info(f'Takeoff to {abs(self.alt_ned):.1f}m')

        # ── TAKEOFF ──
        elif self.phase == 'TAKEOFF':
            yaw = math.atan2(self.rock_y, self.rock_x)
            self._pub_setpoint(0.0, 0.0, self.alt_ned, yaw)
            if self._dist_to(0.0, 0.0, self.alt_ned) < 0.5:
                self.hover_start = self._now_sec()
                self.phase = 'HOVER'
                self.get_logger().info(f'Hovering {self.hover_time}s for SLAM init')

        # ── HOVER ──
        elif self.phase == 'HOVER':
            yaw = math.atan2(self.rock_y, self.rock_x)
            self._pub_setpoint(0.0, 0.0, self.alt_ned, yaw)
            elapsed = self._now_sec() - self.hover_start
            if int(elapsed) != int(elapsed - 0.05):
                self.get_logger().info(f'Hover: {elapsed:.0f}/{self.hover_time:.0f}s')
            if elapsed >= self.hover_time:
                self.phase = 'ORBIT'
                self.get_logger().info(f'Pass {self.current_pass}: {self.passes[self.current_pass]}')

        # ── ORBIT ──
        elif self.phase == 'ORBIT':
            x, y, z, yaw = self.waypoints[self.wp_idx]
            self._pub_setpoint(x, y, z, yaw)

            at_wp = self._dist_to(x, y, z) < 0.5
            if at_wp and self.settle_start is None:
                self.settle_start = self._now_sec()
            if at_wp and self.settle_start and (self._now_sec() - self.settle_start) >= self.settle_time:
                self._save_keyframe(self.pos.x, self.pos.y, self.pos.z, yaw)
                self.get_logger().info(
                    f'KF {self.kf_count} pass={self.current_pass} wp={self.wp_idx+1}/{len(self.waypoints)}')
                self.wp_idx += 1
                self.settle_start = None

                if self.wp_idx >= len(self.waypoints):
                    self.current_pass += 1
                    if self.current_pass < len(self.passes):
                        self._setup_pass(self.current_pass)
                        self.phase = 'CLIMB'
                        self.get_logger().info('Pass done, climbing')
                    else:
                        self.phase = 'RETURN'
                        self.get_logger().info('All orbits complete, returning home')

        # ── CLIMB ──
        elif self.phase == 'CLIMB':
            x, y, z, yaw = self.waypoints[0]
            self._pub_setpoint(self.pos.x, self.pos.y, self.alt_ned, yaw)
            if abs(self.pos.z - self.alt_ned) < 0.5:
                self.phase = 'ORBIT'
                self.get_logger().info(f'Pass {self.current_pass}: {self.passes[self.current_pass]}')

        # ── RETURN ──
        elif self.phase == 'RETURN':
            self._pub_setpoint(0.0, 0.0, -1.0, 0.0)
            if self._dist_to(0.0, 0.0, -1.0) < 1.0:
                self._land()
                self.phase = 'LANDING'

        # ── LANDING ──
        elif self.phase == 'LANDING':
            if self.pos.z > -0.3:
                self._disarm()
                self._save_all()
                self.get_logger().info('Landed. Done.')
                raise SystemExit(0)


def main():
    rclpy.init()
    node = OrbitMapper()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
