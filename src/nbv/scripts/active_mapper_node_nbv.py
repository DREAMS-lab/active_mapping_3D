#!/usr/bin/env python3
"""NBV mapper: Gaussian uncertainty + angular information viewpoint selection.

Single-process, three-thread architecture:
  Thread 1 (ROS main): Flight FSM, keyframe selection, PX4 offboard control,
                        voxel coverage tracking, RViz2 publishing
  Thread 2 (CUDA optimizer): Incremental 3DGS training with gsplat
  Thread 3 (Viewer): OpenCV splat viewer with mouse orbit controls

Flight state machine:
  PREFLIGHT -> TAKEOFF -> SEED (4 KFs at 90° intervals)
            -> [NBV_SCORE -> NBV_FLY -> NBV_SETTLE] x N rounds (4 KFs each)
            -> RETURN -> LANDING -> OFFLINE_RECON -> DONE

Fully adaptive approach:
  1. Seed: 4 images from quarter-orbit (90° apart) build initial 3DGS model
  2. Score-fly loop: every 4 captures, re-score all candidates using
     Gaussian uncertainty (gradient, anisotropy, opacity, scale) combined
     with angular information, pick next 4 best viewpoints, fly + capture
  3. Each round's decisions are based on everything seen so far
  4. Offline reconstruction via gaussian-splatting at full 1080p (30K iters)
"""

import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    BatteryStatus,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge

# Local imports (same directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from gaussian_model_nbv import GaussianModel3DGS, FX, FY, CX, CY, W, H, W_NBV, H_NBV
from splat_viewer_nbv import SplatViewer
from voxel_grid_nbv import VoxelGrid
from rviz_publisher_nbv import RVizPublisher


class ActiveMapperNode(Node):
    def __init__(self):
        super().__init__('active_mapper')

        # -- Parameters --
        self.declare_parameter('rock_x', 0.0)
        self.declare_parameter('rock_y', 8.0)
        self.declare_parameter('rock_z_up', 0.8)
        self.declare_parameter('bbox_size', 2.0)
        self.declare_parameter('voxel_resolution', 0.05)
        self.declare_parameter('orbit_altitude_1', 1.0)
        self.declare_parameter('orbit_altitude_2', 2.5)
        self.declare_parameter('orbit_altitude_3', 0.5)
        self.declare_parameter('gimbal_pitch_1', -15.0)
        self.declare_parameter('gimbal_pitch_2', -30.0)
        self.declare_parameter('gimbal_pitch_3', -5.0)
        self.declare_parameter('orbit_waypoints', 24)
        self.declare_parameter('orbit_radius', 2.0)
        self.declare_parameter('settle_time', 1.0)  # seconds in SIM time
        self.declare_parameter('max_gaussians', 500000)
        self.declare_parameter('iters_per_keyframe', 2000)
        self.declare_parameter('window_size', 0)
        self.declare_parameter('pts_per_frame', 40000)
        self.declare_parameter('kf_dist_threshold', 0.3)
        self.declare_parameter('kf_rot_threshold', 10.0)
        self.declare_parameter('kf_time_threshold', 0.5)
        self.declare_parameter('densify_every', 3)
        self.declare_parameter('enable_viewer', True)

        self.rock_x = self.get_parameter('rock_x').value
        self.rock_y = self.get_parameter('rock_y').value
        self.rock_z_up = self.get_parameter('rock_z_up').value
        self.rock_ned = np.array([
            self.rock_x, self.rock_y, -self.rock_z_up], dtype=np.float32)
        self.bbox_size = self.get_parameter('bbox_size').value
        self.voxel_res = self.get_parameter('voxel_resolution').value
        self.n_wp = int(self.get_parameter('orbit_waypoints').value)
        self.settle_time = self.get_parameter('settle_time').value
        self.iters_per_kf = int(self.get_parameter('iters_per_keyframe').value)
        self.window_size = int(self.get_parameter('window_size').value)
        self.kf_dist_thresh = self.get_parameter('kf_dist_threshold').value
        self.kf_rot_thresh = math.radians(
            self.get_parameter('kf_rot_threshold').value)
        self.kf_time_thresh = self.get_parameter('kf_time_threshold').value
        self.densify_every = int(self.get_parameter('densify_every').value)
        self.enable_viewer = self.get_parameter('enable_viewer').value

        # Orbit pass (single orbit as prior)
        self.orbit_alt = self.get_parameter('orbit_altitude_1').value
        self.orbit_pitch_deg = self.get_parameter('gimbal_pitch_1').value
        self.passes = [
            (self.orbit_alt, self.orbit_pitch_deg),
        ]
        self.radius = self.get_parameter('orbit_radius').value

        # ── NBV planner (bbox-alpha + angular diversity) ──
        self.declare_parameter('kf_budget', 48)
        self.declare_parameter('seed_kfs', 4)
        self.declare_parameter('batch_size', 4)
        self.declare_parameter('nbv_n_azimuth', 12)
        self.declare_parameter('nbv_k', 24)
        self.declare_parameter('offline_iters', 30000)
        self.declare_parameter('offline_max_gaussians', 1000000)
        self.declare_parameter('offline_train_scale', 2)
        self.declare_parameter('skip_adaptive', False)
        self.kf_budget = int(self.get_parameter('kf_budget').value)
        self.seed_kfs = int(self.get_parameter('seed_kfs').value)
        self.batch_size = int(self.get_parameter('batch_size').value)
        self.nbv_k = int(self.get_parameter('nbv_k').value)
        self.skip_adaptive = bool(self.get_parameter('skip_adaptive').value)

        from nbv_planner import NBVPlanner
        self.planner = NBVPlanner(
            rock_ned=self.rock_ned.tolist(),
            radius=self.radius,
            min_altitude=0.3,
            bbox_size=self.bbox_size,
        )

        # Adaptive budget = total - seed
        self.adaptive_budget = self.kf_budget - self.seed_kfs
        self.nbv_batch_count = 0  # KFs captured in current NBV batch
        self.nbv_round = 0        # which scoring round we're on

        self.get_logger().info(
            f'NBV planner: {self.planner.n_candidates} candidates, '
            f'seed={self.seed_kfs} KFs, '
            f'adaptive={self.adaptive_budget} KFs '
            f'(batches of {self.batch_size})'
            f'{" (SKIPPED — seed-only baseline)" if self.skip_adaptive else ""}')

        # NBV state
        self.nbv_target = None
        self.nbv_queue = []          # current batch of viewpoints to visit
        self.nbv_training_done = threading.Event()
        self.scoring_ready = threading.Event()  # set when optimizer has processed seed
        self.avoid_waypoint = None          # Phase 1: climb at current XY
        self.avoid_waypoint_transit = None  # Phase 2: transit at safe alt
        self.avoid_approach = None          # Phase 3: descend at cylinder edge

        # Per-round tracking for debugging and convergence analysis
        self.round_history = []  # appended after each scoring round

        # -- Output directory --
        # Workspace root: <ws>/install/nbv/share/nbv/scripts/this.py -> <ws>
        self.ws_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        data_base = os.path.join(self.ws_root, 'data', 'nbv')
        os.makedirs(data_base, exist_ok=True)
        run_num = 1
        while os.path.exists(os.path.join(data_base, f'run_{run_num:03d}')):
            run_num += 1
        self.run_dir = os.path.join(data_base, f'run_{run_num:03d}')
        self.img_dir = os.path.join(self.run_dir, 'images')
        self.depth_dir = os.path.join(self.run_dir, 'depth')
        self.model_dir = os.path.join(self.run_dir, 'model')
        self.voxel_dir = os.path.join(self.run_dir, 'voxels')
        self.eval_dir = os.path.join(self.run_dir, 'eval')
        self.ply_dir = os.path.join(self.run_dir, 'ply')
        self.nbv_dir = os.path.join(self.run_dir, 'nbv')
        for d in [self.img_dir, self.depth_dir,
                  self.model_dir, self.voxel_dir, self.eval_dir,
                  self.ply_dir, self.nbv_dir]:
            os.makedirs(d, exist_ok=True)

        self.get_logger().info(f'Output: {self.run_dir}')
        self.get_logger().info(
            f'Rock NED: ({self.rock_ned[0]:.1f}, {self.rock_ned[1]:.1f}, '
            f'{self.rock_ned[2]:.1f})')

        # -- GPU info (collect once at startup) --
        import torch
        self.gpu_info = self._collect_gpu_info(torch)

        # -- Components --
        self.rock_ned_original = self.rock_ned.copy()

        self.gs_model = GaussianModel3DGS(
            max_gaussians=int(self.get_parameter('max_gaussians').value),
            pts_per_frame=int(self.get_parameter('pts_per_frame').value),
            bbox_center=self.rock_ned_original,
            bbox_size=self.bbox_size,
            train_scale=2)  # 640/2=320, 480/2=240 — run 072 setting
        self.voxels = VoxelGrid(
            self.rock_ned_original, self.bbox_size, self.voxel_res)
        self.rviz = RVizPublisher(self)

        # -- State (must be initialized before threads start) --
        self.pos = None
        self.status = None
        self.battery = None
        self.latest_rgb = None
        self.latest_depth = None
        self.phase = 'PREFLIGHT'

        # -- ROS setup --
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.cmd_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.gimbal_pitch_pub = self.create_publisher(
            Float64, '/gimbal/pitch', 10)
        self.gimbal_yaw_pub = self.create_publisher(
            Float64, '/gimbal/yaw', 10)

        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1', self._pos_cb, qos)
        self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v1', self._status_cb, qos)
        self.create_subscription(
            BatteryStatus,
            '/fmu/out/battery_status', self._battery_cb, qos)
        self.create_subscription(Image, '/rgbd/image', self._rgb_cb, 10)
        self.create_subscription(Image, '/rgbd/depth', self._depth_cb, 10)

        self.bridge = CvBridge()

        # -- Viewer (Thread 3) --
        self.viewer = None
        if self.enable_viewer:
            self.viewer = SplatViewer(self.rock_ned, bbox_size=self.bbox_size)
            self.viewer_thread = threading.Thread(
                target=self.viewer.run, daemon=True)
            self.viewer_thread.start()

        # -- Optimizer thread communication --
        self.kf_queue = queue.Queue()
        self.optimizer_active = True  # set False after NBV scoring

        # -- Depth keyframes (kept for potential future use) --
        self.depth_keyframes = []  # list of {'depth_np': HxW, 'viewmat_np': 4x4}
        self.depth_keyframes_lock = threading.Lock()
        self.opt_thread = threading.Thread(
            target=self._optimizer_loop, daemon=True)

        self.counter = 0
        self.current_pass = 0
        self.settle_start = None
        self.gimbal_yaw_rad = 0.0
        self._ned_calibrated = False
        self._gimbal_set = False
        self.kf_count = 0
        self._all_view_positions = []  # shared: optimizer writes, planner reads
        self.frames_meta = []
        self.saved = False

        # Keyframe selection state
        self.last_kf_pos = None
        self.last_kf_yaw = None
        self.last_kf_time = 0.0

        # Current orbit state
        self._setup_pass(0)

        # Publish bounding box once
        self.rviz.publish_bounding_box(self.rock_ned.tolist(), self.bbox_size)

        # ── Metrics tracking ──────────────────────────────────
        self.mission_start_time = None  # set on ARM
        self.phase_times = {}  # phase_name -> {'start': t, 'end': t}
        self._phase_enter_time = None

        # Training round metrics (populated by optimizer thread)
        self.training_log = []
        # Forgetting monitor: per-round PSNR on sentinel keyframes
        self.forgetting_log = []  # [{round, sentinels: [{kf_id, psnr, l1}]}]

        # Keyframe capture metrics
        self.kf_metrics = []

        # Battery log
        self.battery_log = []

        # Start optimizer thread (after all state is initialized)
        self.opt_thread.start()

        self.timer = self.create_timer(0.05, self._loop)  # 20Hz
        self.get_logger().info('Active mapper started')
        self.get_logger().info(
            f'GPU: {self.gpu_info["name"]} | '
            f'VRAM: {self.gpu_info["total_vram_mb"]:.0f} MB | '
            f'CUDA: {self.gpu_info["cuda_version"]}')

    # ── GPU info ───────────────────────────────────────────

    @staticmethod
    def _collect_gpu_info(torch):
        info = {
            'name': 'unknown',
            'total_vram_mb': 0,
            'cuda_version': torch.version.cuda or 'N/A',
            'torch_version': torch.__version__,
            'cudnn_version': str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else 'N/A',
            'cuda_arch_list': os.environ.get('TORCH_CUDA_ARCH_LIST', 'auto'),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info['name'] = props.name
            info['total_vram_mb'] = props.total_memory / 1024 / 1024
            info['compute_capability'] = f'{props.major}.{props.minor}'
            info['sm_count'] = props.multi_processor_count
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=driver_version',
                     '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=5)
                info['driver_version'] = result.stdout.strip()
            except Exception:
                info['driver_version'] = 'unknown'
        return info

    @staticmethod
    def _get_vram_stats():
        """Current VRAM usage in MB."""
        import torch
        if not torch.cuda.is_available():
            return {'allocated_mb': 0, 'reserved_mb': 0, 'peak_mb': 0}
        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
            'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024 / 1024,
        }

    # ── ROS callbacks ──────────────────────────────────────

    def _pos_cb(self, msg):
        self.pos = msg

    def _status_cb(self, msg):
        self.status = msg

    def _battery_cb(self, msg):
        if self.battery is None:
            self.get_logger().info(
                f'Battery connected: {msg.voltage_v:.1f}V, '
                f'{msg.remaining*100:.0f}%')
        self.battery = msg

    def _rgb_cb(self, msg):
        self.latest_rgb = msg

    def _depth_cb(self, msg):
        self.latest_depth = msg

    # ── PX4 helpers (from orbit_mapper.py) ─────────────────

    def _pub_offboard(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def _pub_setpoint(self, x, y, z, yaw=float('nan'), max_speed=None):
        """Publish position setpoint — raw coordinates, PX4 handles trajectory.

        Matches the projects/ version: no velocity limiting, no per-tick
        position clamping.  PX4's position controller manages the approach.
        """
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.velocity = [float('nan')] * 3
        msg.acceleration = [float('nan')] * 3
        msg.jerk = [float('nan')] * 3
        msg.yaw = yaw
        msg.yawspeed = float('nan')
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def _path_needs_avoidance(self, target_pos):
        """Check if straight-line path to target passes near the rock.

        Two checks:
          1. 3D point-to-segment distance (catches diagonal paths)
          2. Horizontal (XY) point-to-segment distance (catches paths that
             fly directly over the rock at altitude — the 3D check misses
             these because vertical offset inflates the distance)

        If either is too close, returns a climb-first waypoint.
        """
        if self.pos is None:
            return None
        A = np.array([self.pos.x, self.pos.y, self.pos.z])
        B = np.array(target_pos, dtype=np.float64)
        R = np.array(self.rock_ned, dtype=np.float64)

        AB = B - A
        seg_len_sq = np.dot(AB, AB)
        if seg_len_sq < 0.01:
            return None

        # 3D distance check
        t = np.clip(np.dot(R - A, AB) / seg_len_sq, 0.0, 1.0)
        closest = A + t * AB
        dist_3d = np.linalg.norm(closest - R)

        # Horizontal (XY-only) distance check — critical for paths over rock
        AB_xy = AB[:2]
        seg_len_sq_xy = np.dot(AB_xy, AB_xy)
        if seg_len_sq_xy > 0.01:
            t_xy = np.clip(np.dot(R[:2] - A[:2], AB_xy) / seg_len_sq_xy,
                           0.0, 1.0)
            closest_xy = A[:2] + t_xy * AB_xy
            dist_xy = np.linalg.norm(closest_xy - R[:2])
        else:
            dist_xy = np.linalg.norm(A[:2] - R[:2])

        # Horizontal safety margin: half bbox + 1.5m clearance for props/drift
        xy_margin = self.bbox_size / 2.0 + 1.5
        needs_avoidance = dist_3d < self.bbox_size or dist_xy < xy_margin

        if needs_avoidance:
            # Safe alt: bbox_size + 1m above rock center (extra margin)
            safe_alt = self.rock_ned[2] - self.bbox_size - 1.0
            yaw = math.atan2(
                target_pos[1] - self.pos.y,
                target_pos[0] - self.pos.x)
            reason = (f'3D={dist_3d:.1f}m' if dist_3d < self.bbox_size
                      else f'XY={dist_xy:.1f}m<{xy_margin:.1f}m')
            self.get_logger().info(
                f'Path avoidance ({reason}): '
                f'climbing to z={safe_alt:.1f} before transit')
            return (float(A[0]), float(A[1]), float(safe_alt), float(yaw))
        return None

    def _target_above_rock(self, target_pos):
        """Check if target is within the rock's vertical cylinder.

        Returns an approach waypoint at the cylinder edge if the target
        is too close horizontally, or None if the target is clear.
        """
        hs = self.bbox_size / 2.0
        horiz_clearance = 1.5  # meters outside bbox edge
        dx = target_pos[0] - self.rock_ned[0]
        dy = target_pos[1] - self.rock_ned[1]
        horiz_dist = math.sqrt(dx * dx + dy * dy)
        safe_radius = hs + horiz_clearance

        if horiz_dist < safe_radius:
            # Target is above/near rock — approach from outside the cylinder
            # at target altitude, same azimuth
            az = math.atan2(dy, dx)
            approach_x = self.rock_ned[0] + safe_radius * math.cos(az)
            approach_y = self.rock_ned[1] + safe_radius * math.sin(az)
            approach_z = float(target_pos[2])
            yaw = math.atan2(
                self.rock_ned[1] - approach_y,
                self.rock_ned[0] - approach_x)
            self.get_logger().info(
                f'Target above rock (horiz={horiz_dist:.1f}m < {safe_radius:.1f}m): '
                f'approaching from cylinder edge at r={safe_radius:.1f}m')
            return (approach_x, approach_y, approach_z, yaw)
        return None

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
        self._send_cmd(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, p1=1.0)
        self.get_logger().info('ARM')

    def _disarm(self):
        self._send_cmd(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, p1=0.0)
        self.get_logger().info('DISARM')

    def _offboard(self):
        self._send_cmd(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, p1=1.0, p2=6.0)
        self.get_logger().info('OFFBOARD')

    def _land(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('LAND')

    def _yaw_to_rock(self):
        """Compute yaw from current position to rock center."""
        return math.atan2(
            self.rock_ned[1] - self.pos.y,
            self.rock_ned[0] - self.pos.x)

    def _pub_gimbal(self, pitch_rad, yaw_rad=0.0):
        self.gimbal_pitch_pub.publish(Float64(data=pitch_rad))
        self.gimbal_yaw_pub.publish(Float64(data=yaw_rad))

    def _set_gimbal_to_rock(self):
        """Compute exact gimbal pitch from actual position to rock center.

        Called ONCE when settled at a viewpoint.  Aims at the TOP of the
        rock (center + half bbox upward) so the full rock stays in frame.
        Minimum -15° to avoid seeing landing legs.
        """
        dx = self.rock_ned[0] - self.pos.x
        dy = self.rock_ned[1] - self.pos.y
        # Aim at top of rock: center Z minus half bbox (NED: up = negative)
        aim_z = self.rock_ned[2] - self.bbox_size * 0.3
        dz = aim_z - self.pos.z
        horiz = math.sqrt(dx * dx + dy * dy)
        if horiz > 0.1:
            pitch = -math.atan2(dz, horiz)
            self.gimbal_pitch_rad = min(pitch, math.radians(-15))
        self.get_logger().debug(
            f'Gimbal pitch set to {math.degrees(self.gimbal_pitch_rad):.1f}°')

    @staticmethod
    def _pitch_for_elevation(elev_deg):
        """Fallback pitch from elevation tier (used before settling)."""
        if elev_deg < 20:
            return math.radians(-15)
        elif elev_deg < 40:
            return math.radians(-25)
        elif elev_deg < 60:
            return math.radians(-35)
        else:
            return math.radians(-45)

    def _yaw_error_to_rock(self):
        """Absolute yaw error between drone heading and rock direction."""
        if self.pos is None:
            return math.pi
        desired = self._yaw_to_rock()
        err = desired - self.pos.heading
        # Wrap to [-π, π] and return absolute
        return abs(math.atan2(math.sin(err), math.cos(err)))

    # ── Phase timing ───────────────────────────────────────

    def _enter_phase(self, new_phase):
        now = self._now_sec()
        if self._phase_enter_time is not None and self.phase != 'DONE':
            if self.phase not in self.phase_times:
                self.phase_times[self.phase] = {'start': self._phase_enter_time}
            self.phase_times[self.phase]['end'] = now
            self.phase_times[self.phase]['duration_s'] = (
                now - self._phase_enter_time)
        self.phase = new_phase
        self._phase_enter_time = now
        # Log battery at phase transitions
        self._log_battery()

    def _log_battery(self):
        if self.battery is not None:
            self.battery_log.append({
                'time_s': self._now_sec() - (self.mission_start_time or self._now_sec()),
                'phase': self.phase,
                'voltage_v': float(self.battery.voltage_v),
                'remaining_pct': float(self.battery.remaining) * 100.0,
                'current_a': float(self.battery.current_a),
                'discharged_mah': float(self.battery.discharged_mah)
                if hasattr(self.battery, 'discharged_mah') else 0.0,
            })

    # ── Orbit waypoints ────────────────────────────────────

    def _setup_pass(self, idx):
        alt, pitch_deg = self.passes[idx]
        # Altitude above rock center (NED: negative = up)
        # rock_z_ned is already negative; subtract orbit alt to go higher
        self.alt_ned = self.rock_ned[2] - abs(alt)
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

    def _dist_to(self, tx, ty, tz):
        if self.pos is None:
            return 999.0
        return math.sqrt(
            (self.pos.x - tx)**2
            + (self.pos.y - ty)**2
            + (self.pos.z - tz)**2)

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _sim_sec(self):
        """PX4 simulation time in seconds (RTF-independent)."""
        if self.pos is not None:
            return self.pos.timestamp * 1e-6   # µs → s
        return self._now_sec()  # fallback to wall clock

    # ── Keyframe capture ───────────────────────────────────

    def _should_capture_keyframe(self, yaw):
        now = self._now_sec()
        if (now - self.last_kf_time) < self.kf_time_thresh:
            return False
        if self.last_kf_pos is None:
            return True

        px, py, pz = float(self.pos.x), float(self.pos.y), float(self.pos.z)
        dist = math.sqrt(
            (px - self.last_kf_pos[0])**2
            + (py - self.last_kf_pos[1])**2
            + (pz - self.last_kf_pos[2])**2)

        yaw_diff = abs(yaw - self.last_kf_yaw)
        yaw_diff = min(yaw_diff, 2 * math.pi - yaw_diff)

        return dist > self.kf_dist_thresh or yaw_diff > self.kf_rot_thresh

    def _capture_keyframe(self, yaw):
        if self.latest_rgb is None or self.latest_depth is None:
            return

        t_start = time.perf_counter()

        idx_str = f'{self.kf_count:05d}'
        px, py, pz = float(self.pos.x), float(self.pos.y), float(self.pos.z)

        rgb_np = self.bridge.imgmsg_to_cv2(self.latest_rgb, 'bgr8')
        if rgb_np.shape[:2] != (H, W):
            return

        depth_np = self.bridge.imgmsg_to_cv2(self.latest_depth, 'passthrough')
        if depth_np.shape[:2] != (H, W):
            return

        # Save to disk
        cv2.imwrite(os.path.join(self.img_dir, f'{idx_str}.png'), rgb_np)
        depth_clean = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = (depth_clean * 1000.0).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(self.depth_dir, f'{idx_str}.png'), depth_mm)

        t_save = time.perf_counter()

        # Compute viewmat from body heading (gimbal yaw is always 0)
        viewmat = GaussianModel3DGS.compute_viewmat(
            px, py, pz, yaw, self.gimbal_pitch_rad)

        # Backproject depth for 3DGS init
        pts_world, colors = self.gs_model.backproject_depth(
            depth_clean, viewmat, rgb_np)

        t_backproject = time.perf_counter()

        # Update voxel grid
        self.voxels.update_from_depth(depth_clean, viewmat)

        # Store depth keyframe
        with self.depth_keyframes_lock:
            self.depth_keyframes.append({
                'depth_np': depth_clean.copy(),
                'viewmat_np': viewmat.copy(),
            })

        t_voxel = time.perf_counter()

        # Metadata
        self.frames_meta.append({
            'file_path': f'images/{idx_str}.png',
            'depth_path': f'depth/{idx_str}.png',
            'position_ned': [px, py, pz],
            'heading': yaw,
            'gimbal_pitch': self.gimbal_pitch_rad,
        })

        # Queue for optimizer (always — live training throughout)
        rgb_f = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB).astype(
            np.float32) / 255.0
        h_t, w_t = self.gs_model.H_train, self.gs_model.W_train
        if rgb_f.shape[:2] != (h_t, w_t):
            rgb_f = cv2.resize(
                rgb_f, (w_t, h_t), interpolation=cv2.INTER_AREA)
        depth_train = cv2.resize(
            depth_clean, (w_t, h_t), interpolation=cv2.INTER_NEAREST)
        self.kf_queue.put({
            'rgb': rgb_f,
            'depth': depth_train,
            'viewmat': viewmat,
            'pts_world': pts_world,
            'colors': colors,
            'position': np.array([px, py, pz], dtype=np.float32),
        })

        t_end = time.perf_counter()

        self.kf_count += 1
        self.last_kf_pos = (px, py, pz)
        self.last_kf_yaw = yaw
        self.last_kf_time = self._now_sec()

        # KF timing metrics
        self.kf_metrics.append({
            'kf_id': self.kf_count - 1,
            'time_s': self._now_sec() - (self.mission_start_time or 0),
            'phase': self.phase,
            'position_ned': [px, py, pz],
            'yaw_rad': yaw,
            'gimbal_pitch_rad': self.gimbal_pitch_rad,
            'n_points': len(pts_world),
            'save_ms': (t_save - t_start) * 1000,
            'backproject_ms': (t_backproject - t_save) * 1000,
            'voxel_ms': (t_voxel - t_backproject) * 1000,
            'total_ms': (t_end - t_start) * 1000,
            'coverage_pct': self.voxels.get_coverage_pct(),
            'battery_remaining_pct': (
                float(self.battery.remaining) * 100.0
                if self.battery else -1.0),
        })

        # Log battery every keyframe
        self._log_battery()

        # RViz: camera pose + status
        self.rviz.publish_camera_pose(
            self.kf_count - 1, (px, py, pz), yaw)
        coverage = self.voxels.get_coverage_pct()
        self.rviz.publish_status(
            self.phase, coverage,
            self.gs_model.n_gaussians, self.kf_count)

        yaw_err_deg = math.degrees(self._yaw_error_to_rock())
        self.get_logger().info(
            f'KF {self.kf_count}: pos=({px:.1f},{py:.1f},{pz:.1f}) '
            f'yaw={math.degrees(yaw):.1f}° '
            f'body={math.degrees(self.pos.heading):.1f}° '
            f'err={yaw_err_deg:.1f}° '
            f'pitch={math.degrees(self.gimbal_pitch_rad):.1f}° '
            f'pts={len(pts_world)} cov={coverage:.0%} '
            f'[{(t_end-t_start)*1000:.0f}ms]')

    # ── Spatial window selection ────────────────────────────

    def _select_spatial_window(self, all_views, query_pos):
        """Select training views for the current round.

        If window_size == 0: use ALL views (no windowing). This prevents
        catastrophic forgetting when NBV sends the drone to diverse,
        spread-out viewpoints.

        If window_size > 0: pick the N nearest views by camera position
        distance (spatial window). This was designed for structured orbits
        with correlated nearby views but causes forgetting with NBV.

        Args:
            all_views: list of (rgb_t, depth_t, vm_t, position) tuples
            query_pos: np.array [3] — position of the new keyframe

        Returns:
            list of (rgb_t, depth_t, vm_t, position) tuples
        """
        # window_size=0 means use all views (no windowing)
        if self.window_size <= 0 or len(all_views) <= self.window_size:
            return list(all_views)

        # Compute distances from query to all views
        positions = np.array([v[3] for v in all_views], dtype=np.float32)
        dists = np.linalg.norm(positions - query_pos[None, :], axis=1)

        # Pick closest window_size views
        nearest_idx = np.argsort(dists)[:self.window_size]
        return [all_views[i] for i in nearest_idx]

    # ── Optimizer thread (Thread 2) ────────────────────────

    def _optimizer_loop(self):
        import torch
        device = torch.device('cuda')

        # Import skimage BEFORE signaling ready — holds GIL for 1-2s
        from skimage.metrics import structural_similarity as ssim_fn
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn

        self.get_logger().info('Optimizer thread started')

        all_views = []  # All keyframe views: (rgb_t, depth_t, vm_t, pos)
        rounds = 0
        refine_iters = 100  # extra iters when GPU is idle
        last_per_iter_ms = 0.0  # for adaptive iteration budgeting

        while rclpy.ok():
            # Stop after current round when drone has landed
            if self.phase in ('DONE', 'FINISHING') or not self.optimizer_active:
                self.get_logger().info(
                    f'Optimizer finished — final round complete')
                break

            max_kfs_per_round = 4  # stay incremental — don't drain entire queue
            new_kfs = []
            try:
                kf = self.kf_queue.get(timeout=0.5)
                new_kfs.append(kf)
                while not self.kf_queue.empty() and len(new_kfs) < max_kfs_per_round:
                    new_kfs.append(self.kf_queue.get_nowait())
            except queue.Empty:
                # GPU is idle — continue training on existing views
                if all_views and self.gs_model.means is not None:
                    window = self._select_spatial_window(
                        all_views, all_views[-1][3])
                    for it in range(refine_iters):
                        vi = it % len(window)
                        rgb_gt, depth_gt, vm, _ = window[vi]
                        self.gs_model._train_single_view(vm, rgb_gt, depth_gt)
                    # Update viewer with refined model
                    if self.viewer is not None:
                        snap = self.gs_model.get_snapshot()
                        self.viewer.update_snapshot(snap, self.kf_count)
                continue

            t_round_start = time.perf_counter()

            # Process new keyframes
            all_new_pts = []
            all_new_cols = []
            latest_pos = None
            for kf in new_kfs:
                rgb_t = torch.from_numpy(kf['rgb']).to(device)
                depth_t = torch.from_numpy(kf['depth']).to(device)
                vm_t = torch.from_numpy(kf['viewmat']).to(device)
                pos = kf['position']
                all_views.append((rgb_t, depth_t, vm_t, pos))
                self._all_view_positions.append(pos)
                latest_pos = pos

                if len(kf['pts_world']) > 0:
                    all_new_pts.append(kf['pts_world'])
                    all_new_cols.append(kf['colors'])

            # Spatial window: pick nearest views to the latest keyframe
            views = self._select_spatial_window(all_views, latest_pos)

            if not all_new_pts:
                continue

            new_pts = np.concatenate(all_new_pts, axis=0)
            new_cols = np.concatenate(all_new_cols, axis=0)

            # Taper new points as model grows — avoid drowning trained
            # Gaussians with untrained ones. Full rate below 500K,
            # linearly reduce to 25% at max_gaussians.
            n_current = self.gs_model.n_gaussians
            max_gs = int(self.get_parameter('max_gaussians').value)
            if n_current > 500000:
                keep_frac = max(0.25,
                    1.0 - 0.75 * (n_current - 500000) / max(1, max_gs - 500000))
                n_keep = max(1000, int(len(new_pts) * keep_frac))
                if n_keep < len(new_pts):
                    idx = np.random.choice(len(new_pts), n_keep, replace=False)
                    new_pts = new_pts[idx]
                    new_cols = new_cols[idx]

            t_add_start = time.perf_counter()
            self.gs_model.add_points(new_pts, new_cols)
            t_add_end = time.perf_counter()

            n_views = len(views)
            n_gaussians_before = self.gs_model.n_gaussians

            # Scale iters with view count: ~40 passes/view, floor 750, cap 2000.
            effective_iters = min(2500, max(750, n_views * 50))

            self.get_logger().info(
                f'Training {self.gs_model.n_gaussians} gaussians over '
                f'{n_views} views (spatial, {len(all_views)} total) '
                f'for {effective_iters} iters '
                f'(budget from {last_per_iter_ms:.1f} ms/iter)...')

            vram_before = self._get_vram_stats()

            # Shuffle views each epoch — run 072 config (best result).
            iter_losses = []
            t_train_start = time.perf_counter()
            try:
                view_order = list(range(n_views))
                random.shuffle(view_order)

                for it in range(effective_iters):
                    epoch_pos = it % n_views
                    if epoch_pos == 0 and it > 0:
                        random.shuffle(view_order)
                    vi = view_order[epoch_pos]
                    rgb_gt, depth_gt, vm, _ = views[vi]

                    loss = self.gs_model._train_single_view(
                        vm, rgb_gt, depth_gt)
                    iter_losses.append(loss)

                    if (it + 1) % 100 == 0 or it == 0:
                        self.get_logger().info(
                            f'  iter {it+1}/{effective_iters}: '
                            f'loss={loss:.4f}')

            except Exception as e:
                self.get_logger().error(f'Training error: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
                continue

            t_train_end = time.perf_counter()
            # Update per-iter estimate for next round's adaptive budget
            if iter_losses:
                last_per_iter_ms = (
                    (t_train_end - t_train_start) * 1000
                    / len(iter_losses))
            rounds += 1

            # Densify/prune periodically — stop after 70% of expected rounds
            # (Kerbl et al.: densify iters 500–15K out of 30K)
            # With fully adaptive: expect kf_budget/batch_size rounds
            expected_total_rounds = max(
                self.kf_budget // max(1, self.batch_size), 4)
            densify_until_round = int(0.7 * expected_total_rounds)
            t_densify_start = time.perf_counter()
            did_densify = False
            if (rounds % self.densify_every == 0
                    and rounds <= densify_until_round):
                self.gs_model.densify_and_prune()
                did_densify = True
                self.get_logger().info(
                    f'Densify/prune: {self.gs_model.n_gaussians} gaussians')
            t_densify_end = time.perf_counter()

            vram_after = self._get_vram_stats()

            # Compute PSNR/SSIM on most recent view (at training resolution)
            psnr_val = 0.0
            ssim_val = 0.0
            try:
                with torch.no_grad():
                    last_rgb_gt, _, last_vm, _ = views[-1]
                    rendered = self.gs_model.render_train_res(
                        last_vm.cpu().numpy())
                    gt_np = last_rgb_gt.cpu().numpy()
                    if gt_np.max() > 1.0:
                        gt_np = gt_np / 255.0
                    rd_np = rendered.astype(np.float64) / 255.0
                    gt_np = gt_np.astype(np.float64)
                    psnr_val = float(psnr_fn(gt_np, rd_np, data_range=1.0))
                    ssim_val = float(ssim_fn(
                        gt_np, rd_np, data_range=1.0, channel_axis=2))
            except Exception:
                pass

            # ── Forgetting monitor: re-render ALL keyframes ──
            # Produces complete forgetting matrix: KF × Round → metrics.
            # Each render is ~7ms, 48 KFs = ~336ms overhead per round.
            sentinel_metrics = []
            n_total = len(all_views)
            if n_total >= 2 and self.gs_model.means is not None:
                try:
                    # Track which views were in window vs replay
                    window_set = set(id(v) for v in views)
                    for sid in range(n_total):
                        s_rgb_gt, _, s_vm, _ = all_views[sid]
                        s_rendered = self.gs_model.render_train_res(
                            s_vm.cpu().numpy())
                        s_gt_np = s_rgb_gt.cpu().numpy()
                        if s_gt_np.max() > 1.0:
                            s_gt_np = s_gt_np / 255.0
                        s_rd_np = s_rendered.astype(np.float64) / 255.0
                        s_gt_np = s_gt_np.astype(np.float64)
                        s_psnr = float(psnr_fn(
                            s_gt_np, s_rd_np, data_range=1.0))
                        s_ssim = float(ssim_fn(
                            s_gt_np, s_rd_np, data_range=1.0,
                            channel_axis=2))
                        s_l1 = float(np.mean(np.abs(s_gt_np - s_rd_np)))
                        in_window = id(all_views[sid]) in window_set
                        sentinel_metrics.append({
                            'kf_id': sid,
                            'psnr': s_psnr,
                            'ssim': s_ssim,
                            'l1': s_l1,
                            'in_window': in_window,
                        })
                except Exception:
                    pass  # don't break training if monitor fails

            self.forgetting_log.append({
                'round': rounds,
                'n_keyframes_total': n_total,
                'sentinels': sentinel_metrics,
            })

            # Update viewer
            if self.viewer is not None:
                snap = self.gs_model.get_snapshot()
                self.viewer.update_snapshot(snap, self.kf_count)

            # Update voxel grid visualization
            t_rviz_start = time.perf_counter()
            self.rviz.publish_voxel_grid(self.voxels)
            t_rviz_end = time.perf_counter()

            t_round_end = time.perf_counter()

            # Log detailed round metrics
            self.training_log.append({
                'round': rounds,
                'n_keyframes': self.kf_count,
                'n_gaussians_before': n_gaussians_before,
                'n_gaussians_after': self.gs_model.n_gaussians,
                'n_new_points': len(new_pts),
                'n_views_in_window': n_views,
                'loss_final': self.gs_model.last_loss,
                'loss_first': iter_losses[0] if iter_losses else 0.0,
                'loss_mean': float(np.mean(iter_losses)) if iter_losses else 0.0,
                'loss_min': float(np.min(iter_losses)) if iter_losses else 0.0,
                'psnr': psnr_val,
                'ssim': ssim_val,
                'effective_iters': effective_iters,
                'coverage_pct': self.voxels.get_coverage_pct(),
                'did_densify': did_densify,
                # Timing (ms)
                'round_total_ms': (t_round_end - t_round_start) * 1000,
                'add_points_ms': (t_add_end - t_add_start) * 1000,
                'train_ms': (t_train_end - t_train_start) * 1000,
                'train_per_iter_ms': (t_train_end - t_train_start) * 1000 / max(len(iter_losses), 1),
                'densify_ms': (t_densify_end - t_densify_start) * 1000,
                'rviz_publish_ms': (t_rviz_end - t_rviz_start) * 1000,
                # VRAM (MB)
                'vram_allocated_mb': vram_after['allocated_mb'],
                'vram_reserved_mb': vram_after['reserved_mb'],
                'vram_peak_allocated_mb': vram_after['peak_allocated_mb'],
                'vram_peak_reserved_mb': vram_after['peak_reserved_mb'],
                'vram_delta_mb': (
                    vram_after['allocated_mb'] - vram_before['allocated_mb']),
            })

            vram_pct = (vram_after['peak_allocated_mb']
                        / self.gpu_info['total_vram_mb'] * 100
                        if self.gpu_info.get('total_vram_mb', 0) > 0
                        else 0)
            self.get_logger().info(
                f'Round {rounds}: {self.gs_model.n_gaussians} gs, '
                f'loss={self.gs_model.last_loss:.4f}, '
                f'PSNR={psnr_val:.1f}, SSIM={ssim_val:.3f}, '
                f'VRAM={vram_after["peak_allocated_mb"]:.0f}MB '
                f'({vram_pct:.1f}%), '
                f'time={((t_round_end - t_round_start)*1000):.0f}ms')

            # Save PLY checkpoint at pass boundaries and every 3 rounds
            current_pass = self.kf_count // max(1, self.n_wp)
            prev_pass = (self.kf_count - len(new_kfs)) // max(1, self.n_wp)
            if current_pass != prev_pass or rounds % 3 == 0:
                ply_name = (f'round_{rounds:03d}_pass{current_pass}'
                            f'_kf{self.kf_count:03d}'
                            f'_{self.gs_model.n_gaussians // 1000}K.ply')
                self.gs_model.save_ply(
                    os.path.join(self.ply_dir, ply_name))
                self.get_logger().info(f'Saved PLY: {ply_name}')

    # ── Save & evaluation ──────────────────────────────────

    def _save_metadata(self):
        meta = {
            'camera_model': 'PINHOLE',
            'fl_x': FX, 'fl_y': FY, 'cx': CX, 'cy': CY, 'w': W, 'h': H,
            'frames': self.frames_meta,
        }
        with open(os.path.join(self.run_dir, 'transforms.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    def _save_all(self):
        if self.saved:
            return

        mission_end = self._now_sec()
        mission_duration = (
            mission_end - self.mission_start_time
            if self.mission_start_time else 0)

        # Record final phase
        self._log_battery()

        self._save_metadata()
        self.gs_model.save(self.model_dir)
        self.gs_model.save_ply(os.path.join(self.ply_dir, 'final.ply'))
        self.voxels.save(
            os.path.join(self.voxel_dir, 'coverage_final.npz'))

        # Save full metrics JSON
        metrics = {
            'gpu': self.gpu_info,
            'parameters': {
                'rock_ned': self.rock_ned.tolist(),
                'bbox_size': self.bbox_size,
                'voxel_resolution': self.voxel_res,
                'orbit_radius': self.radius,
                'orbit_waypoints': self.n_wp,
                'passes': self.passes,
                'max_gaussians': int(self.get_parameter('max_gaussians').value),
                'iters_per_keyframe': self.iters_per_kf,
                'window_size': self.window_size,
                'pts_per_frame': int(self.get_parameter('pts_per_frame').value),
                'densify_every': self.densify_every,
                'seed_kfs': self.seed_kfs,
                'batch_size': self.batch_size,
                'nbv_k': self.nbv_k,
                'nbv_rounds': self.nbv_round,
                'skip_adaptive': self.skip_adaptive,
                'offline_iters': int(self.get_parameter('offline_iters').value),
                'offline_max_gaussians': int(
                    self.get_parameter('offline_max_gaussians').value),
                'offline_train_scale': int(
                    self.get_parameter('offline_train_scale').value),
            },
            'nbv_analysis': self.planner.last_analysis,
            'mission': {
                'duration_s': mission_duration,
                'n_keyframes': self.kf_count,
                'n_gaussians_final': self.gs_model.n_gaussians,
                'total_train_steps': self.gs_model.total_train_steps,
                'final_loss': self.gs_model.last_loss,
                'kf_rate_hz': (
                    self.kf_count / mission_duration
                    if mission_duration > 0 else 0),
            },
            'coverage': self.voxels.get_stats(),
            'phase_times': self.phase_times,
            'training_rounds': self.training_log,
            'keyframe_metrics': self.kf_metrics,
            'forgetting_monitor': self.forgetting_log,
            'battery_log': self.battery_log,
            'vram_final': self._get_vram_stats(),
        }
        with open(os.path.join(self.eval_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save training log separately for convenience
        with open(os.path.join(
                self.model_dir, 'training_log.json'), 'w') as f:
            json.dump(self.training_log, f, indent=2)

        # Save forgetting monitor separately
        with open(os.path.join(
                self.model_dir, 'forgetting_log.json'), 'w') as f:
            json.dump(self.forgetting_log, f, indent=2)

        # Save NBV planner analysis
        self.planner.save_analysis(self.nbv_dir)

        self.saved = True
        self.get_logger().info(
            f'=== SAVED {self.kf_count} frames to {self.run_dir} ===')
        self.get_logger().info(
            f'  Seed: {self.seed_kfs} KFs, '
            f'Adaptive: {self.kf_count - self.seed_kfs} KFs '
            f'({self.nbv_round} scoring rounds)')
        stats = self.voxels.get_stats()
        self.get_logger().info(
            f'Coverage: {stats["coverage_pct"]:.0%} '
            f'({stats["n_covered_2plus"]}/{stats["n_occupied"]} voxels)')
        self.get_logger().info(
            f'Mission: {mission_duration:.1f}s, '
            f'{self.gs_model.n_gaussians} gaussians, '
            f'loss={self.gs_model.last_loss:.4f}')

        # Generate evaluation + offline reconstruction
        threading.Thread(
            target=self._generate_evaluation, daemon=True).start()

    def _generate_evaluation(self):
        """Generate all evaluation images and plots. Runs in background."""
        try:
            import torch
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            self.get_logger().info('Generating evaluation outputs...')

            # ── 1. Rendered orbit views (multiple elevations) ──
            renders_dir = os.path.join(self.eval_dir, 'renders_orbit')
            os.makedirs(renders_dir, exist_ok=True)
            n_angles = 12
            elevations = [
                ('low', 1.0, -0.15),    # 1m above, slight look-down
                ('mid', 2.0, -0.35),    # 2m above, moderate look-down
                ('top', 3.5, -0.7),     # 3.5m above, steep look-down
            ]
            render_count = 0
            for elev_name, alt, gpitch in elevations:
                for i in range(n_angles):
                    angle = 2.0 * math.pi * i / n_angles
                    dist = 3.0  # close orbit for detail
                    cx = self.rock_ned[0] + dist * math.cos(angle)
                    cy = self.rock_ned[1] + dist * math.sin(angle)
                    cz = self.rock_ned[2] - alt
                    yaw = math.atan2(
                        self.rock_ned[1] - cy, self.rock_ned[0] - cx)
                    vm = GaussianModel3DGS.compute_viewmat(
                        cx, cy, cz, yaw, gpitch)
                    img = self.gs_model.render(vm)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(
                        renders_dir, f'{elev_name}_{i:02d}.png'), img_bgr)
                    render_count += 1
            self.get_logger().info(
                f'Saved {render_count} orbit renders to eval/renders_orbit/')

            # ── 2. GT vs rendered comparisons ──
            compare_dir = os.path.join(self.eval_dir, 'gt_vs_rendered')
            os.makedirs(compare_dir, exist_ok=True)
            # Pick evenly spaced keyframes
            compare_indices = np.linspace(
                0, self.kf_count - 1,
                min(16, self.kf_count), dtype=int)
            for idx in compare_indices:
                frame = self.frames_meta[idx]
                gt_path = os.path.join(self.run_dir, frame['file_path'])
                gt_img = cv2.imread(gt_path)
                if gt_img is None:
                    continue
                vm = GaussianModel3DGS.compute_viewmat(
                    *frame['position_ned'],
                    frame['heading'],
                    frame['gimbal_pitch'])
                rendered = self.gs_model.render(vm)
                rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
                # Side by side
                combined = np.hstack([gt_img, rendered_bgr])
                # Label
                cv2.putText(combined, 'Ground Truth', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(combined, 'Live 3DGS', (W + 10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite(
                    os.path.join(compare_dir, f'kf_{idx:03d}.png'), combined)
            self.get_logger().info(
                f'Saved {len(compare_indices)} GT vs rendered to '
                f'eval/gt_vs_rendered/')

            # ── 3. Loss + PSNR/SSIM curves ──
            # Snapshot training log to avoid race with optimizer thread
            training_log_snap = list(self.training_log)
            if training_log_snap:
                fig, axes = plt.subplots(3, 1, figsize=(12, 12))
                rounds = [r['round'] for r in training_log_snap]
                loss_final = [r['loss_final'] for r in training_log_snap]
                loss_mean = [r['loss_mean'] for r in training_log_snap]
                loss_first = [r['loss_first'] for r in training_log_snap]

                axes[0].plot(rounds, loss_final, 'b-', label='Final loss', linewidth=1.5)
                axes[0].plot(rounds, loss_mean, 'g--', label='Mean loss', alpha=0.7)
                axes[0].plot(rounds, loss_first, 'r:', label='First iter loss', alpha=0.5)
                axes[0].set_xlabel('Training Round')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Training Loss Over Rounds')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                psnr_vals = [r.get('psnr', 0) for r in training_log_snap]
                ssim_vals = [r.get('ssim', 0) for r in training_log_snap]
                ax_psnr = axes[1]
                ax_psnr.plot(rounds, psnr_vals, 'b-', linewidth=1.5)
                ax_psnr.set_xlabel('Training Round')
                ax_psnr.set_ylabel('PSNR (dB)')
                ax_psnr.set_title('PSNR Over Rounds (higher = better)')
                ax_psnr.grid(True, alpha=0.3)
                ax_ssim = ax_psnr.twinx()
                ax_ssim.plot(rounds, ssim_vals, 'r--', linewidth=1.5)
                ax_ssim.set_ylabel('SSIM')

                n_gs = [r['n_gaussians_after'] for r in training_log_snap]
                axes[2].plot(rounds, [g / 1000 for g in n_gs], 'purple', linewidth=1.5)
                axes[2].set_xlabel('Training Round')
                axes[2].set_ylabel('Gaussians (K)')
                axes[2].set_title('Gaussian Count Over Rounds')
                axes[2].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(self.eval_dir, 'loss_curve.png'),
                            dpi=150)
                plt.close()
                self.get_logger().info('Saved eval/loss_curve.png')

            # ── 4. Coverage progression ──
            if self.kf_metrics:
                fig, ax = plt.subplots(figsize=(10, 5))
                kf_ids = [m['kf_id'] for m in self.kf_metrics]
                covs = [m['coverage_pct'] * 100 for m in self.kf_metrics]
                ax.plot(kf_ids, covs, 'g-o', markersize=4, linewidth=1.5)
                ax.set_xlabel('Keyframe')
                ax.set_ylabel('Coverage %')
                ax.set_title('Voxel Coverage Progression (>= 2 views)')
                ax.grid(True, alpha=0.3)
                # Mark orbit transitions
                for m in self.kf_metrics:
                    if m['phase'] == 'SECOND_ORBIT' and (
                            m['kf_id'] == 0 or
                            self.kf_metrics[m['kf_id']-1]['phase']
                            != 'SECOND_ORBIT'):
                        ax.axvline(m['kf_id'], color='orange',
                                   linestyle='--', alpha=0.5,
                                   label='2nd orbit start')
                ax.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.eval_dir, 'coverage_progression.png'),
                    dpi=150)
                plt.close()
                self.get_logger().info('Saved eval/coverage_progression.png')

            # ── 5. VRAM usage (% of total) ──
            if training_log_snap:
                total_vram = self.gpu_info['total_vram_mb']
                fig, (ax_pct, ax_abs) = plt.subplots(
                    2, 1, figsize=(12, 8))
                rounds = [r['round'] for r in training_log_snap]
                vram_alloc = [r['vram_allocated_mb'] for r in training_log_snap]
                vram_peak = [r['vram_peak_allocated_mb']
                             for r in training_log_snap]

                # Top: percentage of total VRAM
                alloc_pct = [v / total_vram * 100 for v in vram_alloc]
                peak_pct = [v / total_vram * 100 for v in vram_peak]
                ax_pct.fill_between(rounds, alloc_pct, alpha=0.3, color='blue')
                ax_pct.plot(rounds, alloc_pct, 'b-', label='Allocated',
                            linewidth=1.5)
                ax_pct.plot(rounds, peak_pct, 'r--', label='Peak',
                            alpha=0.7, linewidth=1.5)
                ax_pct.set_xlabel('Training Round')
                ax_pct.set_ylabel('VRAM Usage (%)')
                ax_pct.set_title(
                    f'GPU Memory Utilization '
                    f'({total_vram/1024:.1f} GB total)')
                ax_pct.set_ylim(0, max(max(peak_pct) * 1.5, 5))
                ax_pct.legend()
                ax_pct.grid(True, alpha=0.3)

                # Bottom: absolute MB (zoomed to actual range)
                ax_abs.plot(rounds, vram_alloc, 'b-', label='Allocated',
                            linewidth=1.5)
                ax_abs.plot(rounds, vram_peak, 'r--', label='Peak',
                            alpha=0.7, linewidth=1.5)
                max_vram = max(vram_peak) if vram_peak else 100
                ax_abs.set_ylim(0, max_vram * 1.3)
                ax_abs.set_xlabel('Training Round')
                ax_abs.set_ylabel('VRAM (MB)')
                ax_abs.set_title('GPU Memory Usage (zoomed)')
                ax_abs.legend()
                ax_abs.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(self.eval_dir, 'vram_usage.png'),
                            dpi=150)
                plt.close()
                self.get_logger().info('Saved eval/vram_usage.png')

            # ── 6. Timing breakdown ──
            if training_log_snap:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

                rounds = [r['round'] for r in training_log_snap]
                total_ms = [r['round_total_ms'] for r in training_log_snap]
                train_ms = [r['train_ms'] for r in training_log_snap]
                densify_ms = [r['densify_ms'] for r in training_log_snap]
                per_iter = [r['train_per_iter_ms'] for r in training_log_snap]

                ax1.plot(rounds, total_ms, 'b-', label='Round total',
                         linewidth=1.5)
                ax1.plot(rounds, train_ms, 'g-', label='Training', alpha=0.8)
                ax1.plot(rounds, densify_ms, 'r-', label='Densify/prune',
                         alpha=0.7)
                ax1.set_xlabel('Training Round')
                ax1.set_ylabel('Time (ms)')
                ax1.set_title('Training Round Timing')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                n_gs = [r['n_gaussians_after'] / 1000
                        for r in training_log_snap]
                ax2.scatter(n_gs, per_iter, c=rounds, cmap='viridis', s=20)
                ax2.set_xlabel('Gaussians (K)')
                ax2.set_ylabel('Per-iteration time (ms)')
                ax2.set_title('Iteration Time vs Gaussian Count')
                ax2.grid(True, alpha=0.3)
                cbar = plt.colorbar(ax2.collections[0], ax=ax2)
                cbar.set_label('Round')

                plt.tight_layout()
                plt.savefig(os.path.join(self.eval_dir, 'timing.png'),
                            dpi=150)
                plt.close()
                self.get_logger().info('Saved eval/timing.png')

            # ── 7. Battery usage ──
            if self.battery_log and len(self.battery_log) > 1:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                times = [b['time_s'] for b in self.battery_log]
                volts = [b['voltage_v'] for b in self.battery_log]
                pcts = [b['remaining_pct'] for b in self.battery_log]

                ax1.plot(times, volts, 'orange', linewidth=1.5)
                ax1.set_ylabel('Voltage (V)')
                ax1.set_title('Battery During Mission')
                ax1.grid(True, alpha=0.3)

                ax2.plot(times, pcts, 'green', linewidth=1.5)
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Remaining %')
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(self.eval_dir, 'battery.png'),
                            dpi=150)
                plt.close()
                self.get_logger().info('Saved eval/battery.png')

            # ── 7b. Forgetting monitor curves ──
            forgetting_snap = list(self.forgetting_log)
            if forgetting_snap and any(
                    len(r['sentinels']) > 0 for r in forgetting_snap):
                # Collect per-sentinel time series
                # Each sentinel is identified by its kf_id at each round
                # Build {kf_id: [(round, psnr, l1), ...]}
                sentinel_traces = {}
                for entry in forgetting_snap:
                    rd = entry['round']
                    for s in entry['sentinels']:
                        sid = s['kf_id']
                        if sid not in sentinel_traces:
                            sentinel_traces[sid] = {
                                'rounds': [], 'psnr': [], 'l1': []}
                        sentinel_traces[sid]['rounds'].append(rd)
                        sentinel_traces[sid]['psnr'].append(s['psnr'])
                        sentinel_traces[sid]['l1'].append(s['l1'])

                if sentinel_traces:
                    fig, (ax1, ax2) = plt.subplots(
                        2, 1, figsize=(12, 8), sharex=True)

                    # Determine pass boundary for vertical line
                    n_wp = self.n_wp
                    pass_boundary_round = None
                    for entry in forgetting_snap:
                        if entry['n_keyframes_total'] > n_wp:
                            pass_boundary_round = entry['round']
                            break

                    colors_map = plt.cm.tab10
                    for i, (sid, trace) in enumerate(
                            sorted(sentinel_traces.items())):
                        color = colors_map(i % 10)
                        label = f'KF {sid}'
                        ax1.plot(trace['rounds'], trace['psnr'],
                                 '-o', color=color, label=label,
                                 markersize=3, linewidth=1.5)
                        ax2.plot(trace['rounds'], trace['l1'],
                                 '-o', color=color, label=label,
                                 markersize=3, linewidth=1.5)

                    # Mark pass transition
                    if pass_boundary_round is not None:
                        for ax in (ax1, ax2):
                            ax.axvline(
                                x=pass_boundary_round, color='red',
                                linestyle='--', linewidth=2,
                                label='Pass 2 starts'
                                if ax == ax1 else None)

                    ax1.set_ylabel('PSNR (dB)')
                    ax1.set_title(
                        'Forgetting Monitor — Sentinel Keyframe Quality '
                        'Over Training Rounds')
                    ax1.legend(fontsize=8, ncol=2)
                    ax1.grid(True, alpha=0.3)

                    ax2.set_xlabel('Training Round')
                    ax2.set_ylabel('L1 Error')
                    ax2.set_title('L1 Error of Sentinel Keyframes Over Time')
                    ax2.legend(fontsize=8, ncol=2)
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(self.eval_dir, 'forgetting_curves.png'),
                        dpi=150)
                    plt.close()
                    self.get_logger().info(
                        'Saved eval/forgetting_curves.png')

                    # Also save a summary: peak PSNR vs final PSNR per sentinel
                    forgetting_summary = []
                    for sid, trace in sorted(sentinel_traces.items()):
                        peak_psnr = max(trace['psnr'])
                        peak_round = trace['rounds'][
                            trace['psnr'].index(peak_psnr)]
                        final_psnr = trace['psnr'][-1]
                        forgetting_summary.append({
                            'kf_id': sid,
                            'peak_psnr': round(peak_psnr, 2),
                            'peak_at_round': peak_round,
                            'final_psnr': round(final_psnr, 2),
                            'psnr_drop': round(peak_psnr - final_psnr, 2),
                            'peak_l1': round(min(trace['l1']), 4),
                            'final_l1': round(trace['l1'][-1], 4),
                        })
                    with open(os.path.join(
                            self.eval_dir, 'forgetting_summary.json'),
                            'w') as f:
                        json.dump(forgetting_summary, f, indent=2)
                    self.get_logger().info(
                        'Saved eval/forgetting_summary.json')

                # ── 7b. Forgetting heatmap (KF × Round → PSNR) ──
                try:
                    # Build matrix from forgetting_log (now has ALL KFs)
                    all_rounds = sorted(sentinel_traces.keys()
                                        if not sentinel_traces
                                        else set().union(
                                            *[set(t['rounds'])
                                              for t in sentinel_traces.values()]))
                    # Actually build from forgetting_log directly
                    if self.forgetting_log:
                        round_nums = [e['round'] for e in self.forgetting_log]
                        all_kf_ids = sorted(set(
                            s['kf_id'] for e in self.forgetting_log
                            for s in e['sentinels']))
                        if all_kf_ids and round_nums:
                            matrix = np.full(
                                (len(all_kf_ids), len(round_nums)),
                                np.nan)
                            kf_idx = {k: i for i, k in enumerate(all_kf_ids)}
                            for ri, entry in enumerate(self.forgetting_log):
                                for s in entry['sentinels']:
                                    if s['kf_id'] in kf_idx:
                                        matrix[kf_idx[s['kf_id']], ri] = \
                                            s['psnr']

                            fig, ax = plt.subplots(figsize=(14, 8))
                            im = ax.imshow(
                                matrix, aspect='auto', cmap='RdYlGn',
                                vmin=8, vmax=35, interpolation='nearest')
                            ax.set_xlabel('Training Round')
                            ax.set_ylabel('Keyframe ID')
                            ax.set_title(
                                'Forgetting Heatmap: Per-KF PSNR Across '
                                'Training Rounds')
                            ax.set_xticks(range(len(round_nums)))
                            ax.set_xticklabels(round_nums, fontsize=7)
                            ax.set_yticks(range(len(all_kf_ids)))
                            ax.set_yticklabels(all_kf_ids, fontsize=7)
                            plt.colorbar(im, label='PSNR (dB)')
                            plt.tight_layout()
                            plt.savefig(
                                os.path.join(self.eval_dir,
                                             'forgetting_heatmap.png'),
                                dpi=150)
                            plt.close()
                            self.get_logger().info(
                                'Saved eval/forgetting_heatmap.png')

                            # Window membership heatmap
                            has_window = any(
                                'in_window' in s
                                for e in self.forgetting_log
                                for s in e['sentinels'])
                            if has_window:
                                win_matrix = np.full(
                                    (len(all_kf_ids), len(round_nums)),
                                    0.0)
                                for ri, entry in enumerate(
                                        self.forgetting_log):
                                    for s in entry['sentinels']:
                                        if s['kf_id'] in kf_idx:
                                            win_matrix[
                                                kf_idx[s['kf_id']], ri] = \
                                                1.0 if s.get(
                                                    'in_window', False) \
                                                else 0.5
                                fig2, ax2 = plt.subplots(figsize=(14, 8))
                                ax2.imshow(
                                    win_matrix, aspect='auto',
                                    cmap='RdYlGn', vmin=0, vmax=1,
                                    interpolation='nearest')
                                ax2.set_xlabel('Training Round')
                                ax2.set_ylabel('Keyframe ID')
                                ax2.set_title(
                                    'Window Membership: Green=In Window, '
                                    'Yellow=Replay/Excluded')
                                ax2.set_xticks(range(len(round_nums)))
                                ax2.set_xticklabels(round_nums, fontsize=7)
                                ax2.set_yticks(range(len(all_kf_ids)))
                                ax2.set_yticklabels(all_kf_ids, fontsize=7)
                                plt.tight_layout()
                                plt.savefig(
                                    os.path.join(self.eval_dir,
                                                 'window_membership.png'),
                                    dpi=150)
                                plt.close()
                                self.get_logger().info(
                                    'Saved eval/window_membership.png')
                except Exception as e:
                    self.get_logger().warning(
                        f'Heatmap generation failed: {e}')

            # ── 8. Reconstruction quality (per-keyframe + novel views) ──
            from skimage.metrics import structural_similarity as ssim_fn
            from skimage.metrics import peak_signal_noise_ratio as psnr_fn

            recon_dir = os.path.join(self.eval_dir, 'reconstruction_quality')
            os.makedirs(recon_dir, exist_ok=True)
            os.makedirs(os.path.join(recon_dir, 'per_keyframe'), exist_ok=True)
            os.makedirs(os.path.join(recon_dir, 'novel_views'), exist_ok=True)
            os.makedirs(os.path.join(recon_dir, 'error_maps'), exist_ok=True)

            # 8a. Per-keyframe PSNR/SSIM/L1/LPIPS on ALL training keyframes
            # LPIPS (Learned Perceptual Image Patch Similarity) measures
            # perceptual distance using VGG features — better correlates
            # with human judgement than pixel-level metrics.
            import torch as _torch
            lpips_model = None
            try:
                import lpips
                lpips_model = lpips.LPIPS(net='vgg').to('cuda')
                lpips_model.eval()
            except Exception:
                self.get_logger().warning(
                    'LPIPS not available, skipping perceptual metric')

            kf_metrics_list = []
            for idx in range(self.kf_count):
                frame = self.frames_meta[idx]
                gt_path = os.path.join(self.run_dir, frame['file_path'])
                gt_img = cv2.imread(gt_path)
                if gt_img is None:
                    continue
                gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                vm = GaussianModel3DGS.compute_viewmat(
                    *frame['position_ned'],
                    frame['heading'],
                    frame['gimbal_pitch'])
                rendered = self.gs_model.render(vm)
                gt_f = gt_rgb.astype(np.float64) / 255.0
                rd_f = rendered.astype(np.float64) / 255.0
                kf_psnr = float(psnr_fn(gt_f, rd_f, data_range=1.0))
                kf_ssim = float(ssim_fn(
                    gt_f, rd_f, data_range=1.0, channel_axis=2))
                kf_l1 = float(np.mean(np.abs(gt_f - rd_f)))
                # LPIPS (lower = better, 0 = identical)
                kf_lpips = -1.0
                if lpips_model is not None:
                    with _torch.no_grad():
                        gt_t = _torch.from_numpy(
                            gt_f.astype(np.float32)).permute(
                            2, 0, 1).unsqueeze(0).to('cuda') * 2 - 1
                        rd_t = _torch.from_numpy(
                            rd_f.astype(np.float32)).permute(
                            2, 0, 1).unsqueeze(0).to('cuda') * 2 - 1
                        kf_lpips = float(lpips_model(gt_t, rd_t).item())
                kf_metrics_list.append({
                    'keyframe': idx,
                    'psnr': kf_psnr,
                    'ssim': kf_ssim,
                    'l1': kf_l1,
                    'lpips': kf_lpips,
                    'pass': idx // self.n_wp if self.n_wp > 0 else 0,
                })

                # Save error map for evenly spaced keyframes
                if idx % max(1, self.kf_count // 20) == 0:
                    err = np.abs(gt_f - rd_f)
                    err_vis = (err * 255 * 3).clip(0, 255).astype(np.uint8)
                    # Side-by-side: GT | Rendered | Error map
                    combined = np.hstack([gt_img,
                                          cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR),
                                          cv2.cvtColor(err_vis, cv2.COLOR_RGB2BGR)])
                    cv2.putText(combined, f'GT', (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(combined, f'Rendered', (W + 10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(combined,
                                f'Error (PSNR={kf_psnr:.1f} SSIM={kf_ssim:.3f})',
                                (2 * W + 10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imwrite(os.path.join(
                        recon_dir, 'error_maps', f'kf_{idx:03d}.png'), combined)

            # Save per-keyframe CSV
            if kf_metrics_list:
                csv_path = os.path.join(recon_dir, 'per_keyframe_metrics.csv')
                with open(csv_path, 'w') as f:
                    f.write('keyframe,psnr,ssim,l1,pass\n')
                    for m in kf_metrics_list:
                        f.write(f'{m["keyframe"]},{m["psnr"]:.4f},'
                                f'{m["ssim"]:.4f},{m["l1"]:.6f},'
                                f'{m["pass"]}\n')

            # 8b. Novel view evaluation — interpolated + random viewpoints
            novel_metrics = []
            novel_renders_dir = os.path.join(recon_dir, 'novel_views')
            n_novel = 36
            for i in range(n_novel):
                angle = 2.0 * math.pi * i / n_novel
                # Vary radius, altitude, pitch randomly per view
                r = self.radius + random.uniform(-0.5, 0.8)
                alt_offset = random.uniform(0.3, 3.0)
                pitch = random.uniform(-0.6, -0.05)
                cx = self.rock_ned[0] + r * math.cos(angle)
                cy = self.rock_ned[1] + r * math.sin(angle)
                cz = self.rock_ned[2] - alt_offset
                yaw = math.atan2(
                    self.rock_ned[1] - cy, self.rock_ned[0] - cx)
                vm = GaussianModel3DGS.compute_viewmat(cx, cy, cz, yaw, pitch)
                rendered = self.gs_model.render(vm)
                rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
                cv2.putText(rendered_bgr,
                            f'r={r:.1f} alt={alt_offset:.1f} p={math.degrees(pitch):.0f}',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)
                cv2.imwrite(os.path.join(
                    novel_renders_dir, f'novel_{i:03d}.png'), rendered_bgr)

                # Find nearest training keyframe for approximate quality check
                min_dist = float('inf')
                nearest_idx = 0
                cam_pos = np.array([cx, cy, cz])
                for idx, frame in enumerate(self.frames_meta):
                    d = np.linalg.norm(
                        np.array(frame['position_ned']) - cam_pos)
                    if d < min_dist:
                        min_dist = d
                        nearest_idx = idx
                novel_metrics.append({
                    'view': i,
                    'radius': r,
                    'altitude': alt_offset,
                    'pitch_deg': math.degrees(pitch),
                    'nearest_kf': nearest_idx,
                    'nearest_kf_dist': float(min_dist),
                })

            # 8c. Per-pass aggregate metrics
            pass_stats = {}
            for m in kf_metrics_list:
                p = m['pass']
                if p not in pass_stats:
                    pass_stats[p] = {'psnr': [], 'ssim': [], 'l1': []}
                pass_stats[p]['psnr'].append(m['psnr'])
                pass_stats[p]['ssim'].append(m['ssim'])
                pass_stats[p]['l1'].append(m['l1'])

            # 8d. Reconstruction quality summary JSON
            if kf_metrics_list:
                all_psnr = [m['psnr'] for m in kf_metrics_list]
                all_ssim = [m['ssim'] for m in kf_metrics_list]
                all_l1 = [m['l1'] for m in kf_metrics_list]
                recon_summary = {
                    'aggregate': {
                        'mean_psnr': float(np.mean(all_psnr)),
                        'median_psnr': float(np.median(all_psnr)),
                        'min_psnr': float(np.min(all_psnr)),
                        'max_psnr': float(np.max(all_psnr)),
                        'std_psnr': float(np.std(all_psnr)),
                        'mean_ssim': float(np.mean(all_ssim)),
                        'median_ssim': float(np.median(all_ssim)),
                        'min_ssim': float(np.min(all_ssim)),
                        'max_ssim': float(np.max(all_ssim)),
                        'std_ssim': float(np.std(all_ssim)),
                        'mean_l1': float(np.mean(all_l1)),
                        'n_keyframes': len(kf_metrics_list),
                    },
                    'per_pass': {},
                    'per_keyframe': kf_metrics_list,
                    'novel_views': novel_metrics,
                }
                for p, stats in pass_stats.items():
                    recon_summary['per_pass'][f'pass_{p}'] = {
                        'mean_psnr': float(np.mean(stats['psnr'])),
                        'mean_ssim': float(np.mean(stats['ssim'])),
                        'mean_l1': float(np.mean(stats['l1'])),
                        'n_keyframes': len(stats['psnr']),
                    }
                with open(os.path.join(
                        recon_dir, 'quality_summary.json'), 'w') as f:
                    json.dump(recon_summary, f, indent=2)

            # 8e. Quality plots
            if kf_metrics_list:
                fig, axes = plt.subplots(3, 1, figsize=(14, 12))
                kf_ids = [m['keyframe'] for m in kf_metrics_list]
                psnr_vals = [m['psnr'] for m in kf_metrics_list]
                ssim_vals = [m['ssim'] for m in kf_metrics_list]
                l1_vals = [m['l1'] for m in kf_metrics_list]

                # Color by pass
                pass_colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
                pass_ids = [m['pass'] for m in kf_metrics_list]
                colors = [pass_colors[min(p, len(pass_colors)-1)]
                          for p in pass_ids]

                axes[0].bar(kf_ids, psnr_vals, color=colors, alpha=0.7,
                            width=1.0)
                axes[0].axhline(y=float(np.mean(psnr_vals)), color='red',
                                linestyle='--', label=f'Mean: {np.mean(psnr_vals):.1f} dB')
                axes[0].set_ylabel('PSNR (dB)')
                axes[0].set_title('Per-Keyframe PSNR (higher = better)')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                axes[1].bar(kf_ids, ssim_vals, color=colors, alpha=0.7,
                            width=1.0)
                axes[1].axhline(y=float(np.mean(ssim_vals)), color='red',
                                linestyle='--', label=f'Mean: {np.mean(ssim_vals):.3f}')
                axes[1].set_ylabel('SSIM')
                axes[1].set_title('Per-Keyframe SSIM (higher = better)')
                axes[1].set_ylim(0, 1)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                axes[2].bar(kf_ids, l1_vals, color=colors, alpha=0.7,
                            width=1.0)
                axes[2].axhline(y=float(np.mean(l1_vals)), color='red',
                                linestyle='--', label=f'Mean: {np.mean(l1_vals):.4f}')
                axes[2].set_xlabel('Keyframe Index')
                axes[2].set_ylabel('L1 Error')
                axes[2].set_title('Per-Keyframe L1 Error (lower = better)')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)

                # Add pass labels
                for p in sorted(pass_stats.keys()):
                    start_kf = p * self.n_wp
                    axes[0].axvline(x=start_kf, color='gray',
                                    linestyle=':', alpha=0.5)
                    axes[0].text(start_kf + 1, axes[0].get_ylim()[1] * 0.95,
                                 f'Pass {p+1}', fontsize=8, color='gray')

                plt.tight_layout()
                plt.savefig(os.path.join(
                    recon_dir, 'per_keyframe_quality.png'), dpi=150)
                plt.close()

                # Per-pass box plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                pass_data_psnr = [pass_stats[p]['psnr']
                                  for p in sorted(pass_stats.keys())]
                pass_data_ssim = [pass_stats[p]['ssim']
                                  for p in sorted(pass_stats.keys())]
                pass_labels = []
                for p in sorted(pass_stats.keys()):
                    if p < len(self.passes):
                        pass_labels.append(
                            f'Pass {p+1}\n({self.passes[p][0]}m, '
                            f'{self.passes[p][1]}°)')
                    else:
                        pass_labels.append(f'Pass {p+1}')
                if pass_data_psnr and len(pass_labels) == len(pass_data_psnr):
                    ax1.boxplot(pass_data_psnr, labels=pass_labels)
                    ax1.set_ylabel('PSNR (dB)')
                    ax1.set_title('PSNR Distribution by Pass')
                    ax1.grid(True, alpha=0.3)
                if pass_data_ssim and len(pass_labels) == len(pass_data_ssim):
                    ax2.boxplot(pass_data_ssim, labels=pass_labels)
                    ax2.set_ylabel('SSIM')
                    ax2.set_title('SSIM Distribution by Pass')
                    ax2.set_ylim(0, 1)
                    ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(
                    recon_dir, 'per_pass_quality.png'), dpi=150)
                plt.close()

            self.get_logger().info(
                f'Saved reconstruction quality eval to '
                f'eval/reconstruction_quality/')

            # ── 9. NBV analysis data ──
            nbv_dir = self.nbv_dir
            os.makedirs(os.path.join(nbv_dir, 'confidence_maps'), exist_ok=True)
            os.makedirs(os.path.join(nbv_dir, 'voxel_slices'), exist_ok=True)

            # 9a. Confidence maps: render alpha from a dense grid of viewpoints
            # Alpha = accumulated opacity — low alpha = sparse/missing Gaussians
            n_az = 12   # azimuth angles
            n_el = 3    # elevation levels
            elevations_nbv = [
                ('low', 0.8),    # ~46° above horizon
                ('mid', 1.5),    # ~86° — nearly level
                ('top', 3.0),    # steep top-down
            ]
            nbv_candidates = []
            for ei, (elev_name, alt) in enumerate(elevations_nbv):
                for ai in range(n_az):
                    angle = 2.0 * math.pi * ai / n_az
                    r = self.radius
                    cx = self.rock_ned[0] + r * math.cos(angle)
                    cy = self.rock_ned[1] + r * math.sin(angle)
                    cz = self.rock_ned[2] - alt
                    yaw = math.atan2(
                        self.rock_ned[1] - cy, self.rock_ned[0] - cx)
                    gpitch = math.atan2(
                        -(self.rock_ned[2] - cz),
                        math.sqrt((self.rock_ned[0]-cx)**2 + (self.rock_ned[1]-cy)**2))
                    vm = GaussianModel3DGS.compute_viewmat(
                        cx, cy, cz, yaw, gpitch)
                    rgb, alpha = self.gs_model.render_with_alpha(vm)

                    # Confidence = mean alpha in center crop (where rock likely is)
                    ch, cw = H // 4, W // 4
                    center_alpha = alpha[ch:H-ch, cw:W-cw]
                    mean_alpha = float(np.mean(center_alpha))
                    low_alpha_frac = float(
                        np.mean(center_alpha < 0.5))  # fraction of uncertain pixels

                    # Save confidence map as heatmap
                    alpha_vis = (alpha * 255).astype(np.uint8)
                    alpha_color = cv2.applyColorMap(alpha_vis, cv2.COLORMAP_JET)
                    # Overlay alpha value
                    cv2.putText(alpha_color,
                                f'alpha={mean_alpha:.2f} low={low_alpha_frac:.0%}',
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)
                    # Side by side: RGB | Alpha heatmap
                    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    combined = np.hstack([rgb_bgr, alpha_color])
                    cv2.imwrite(os.path.join(
                        nbv_dir, 'confidence_maps',
                        f'{elev_name}_{ai:02d}.png'), combined)

                    nbv_candidates.append({
                        'azimuth_deg': float(np.degrees(angle)),
                        'elevation': elev_name,
                        'altitude': alt,
                        'position': [float(cx), float(cy), float(cz)],
                        'mean_alpha': mean_alpha,
                        'low_alpha_fraction': low_alpha_frac,
                        'yaw': float(yaw),
                        'gimbal_pitch': float(gpitch),
                    })

            # 9b. Rank candidates by information gain (low alpha = high info)
            nbv_candidates.sort(key=lambda c: c['mean_alpha'])
            with open(os.path.join(nbv_dir, 'nbv_candidates.json'), 'w') as f:
                json.dump({
                    'description': (
                        'Candidate viewpoints ranked by reconstruction '
                        'confidence. Lower mean_alpha = more uncertain '
                        'region = higher information gain for NBV.'),
                    'candidates': nbv_candidates,
                    'top_5_views': nbv_candidates[:5],
                }, f, indent=2)

            # 9c. Voxel grid slices — per-axis cross sections with view counts
            vg = self.voxels
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            # XY slice (top-down) at middle Z
            mid_z = vg.n // 2
            xy_slice = vg.n_views[:, :, mid_z].T  # transpose for imshow
            im0 = axes[0].imshow(xy_slice, cmap='hot', origin='lower',
                                  vmin=0, vmax=max(5, xy_slice.max()))
            axes[0].set_title(f'Top-down (XY at z={mid_z})')
            axes[0].set_xlabel('X voxel')
            axes[0].set_ylabel('Y voxel')
            plt.colorbar(im0, ax=axes[0], label='View count')

            # XZ slice (front view) at middle Y
            mid_y = vg.n // 2
            xz_slice = vg.n_views[:, mid_y, :].T
            im1 = axes[1].imshow(xz_slice, cmap='hot', origin='lower',
                                  vmin=0, vmax=max(5, xz_slice.max()))
            axes[1].set_title(f'Front (XZ at y={mid_y})')
            axes[1].set_xlabel('X voxel')
            axes[1].set_ylabel('Z voxel')
            plt.colorbar(im1, ax=axes[1], label='View count')

            # YZ slice (side view) at middle X
            mid_x = vg.n // 2
            yz_slice = vg.n_views[mid_x, :, :].T
            im2 = axes[2].imshow(yz_slice, cmap='hot', origin='lower',
                                  vmin=0, vmax=max(5, yz_slice.max()))
            axes[2].set_title(f'Side (YZ at x={mid_x})')
            axes[2].set_xlabel('Y voxel')
            axes[2].set_ylabel('Z voxel')
            plt.colorbar(im2, ax=axes[2], label='View count')

            plt.suptitle('Voxel View Count Slices (hot = many views)')
            plt.tight_layout()
            plt.savefig(os.path.join(nbv_dir, 'voxel_slices.png'), dpi=150)
            plt.close()

            # 9d. View count distribution histogram
            occupied_counts = vg.n_views[vg.occupied]
            if len(occupied_counts) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                max_count = min(int(occupied_counts.max()), 20)
                bins = np.arange(0, max_count + 2) - 0.5
                ax.hist(occupied_counts, bins=bins, color='steelblue',
                        edgecolor='white', alpha=0.8)
                ax.axvline(x=2, color='orange', linestyle='--',
                           label='2-view threshold')
                ax.axvline(x=3, color='green', linestyle='--',
                           label='3-view threshold')
                ax.set_xlabel('Number of Views')
                ax.set_ylabel('Number of Voxels')
                ax.set_title('View Count Distribution (occupied voxels)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(nbv_dir, 'view_count_histogram.png'),
                            dpi=150)
                plt.close()

            # 9e. Save raw voxel grid as npz for external analysis
            np.savez(os.path.join(nbv_dir, 'voxel_grid.npz'),
                     occupied=vg.occupied,
                     n_views=vg.n_views,
                     center=vg.center,
                     bbox_size=vg.bbox_size,
                     resolution=vg.resolution,
                     origin=vg.origin)

            # 9f. Under-observed voxel positions (occupied but < 2 views)
            under_observed = vg.occupied & (vg.n_views < 2)
            if under_observed.any():
                uo_indices = np.argwhere(under_observed)
                uo_world = (vg.origin[None] +
                            (uo_indices + 0.5) * vg.resolution)
                np.save(os.path.join(nbv_dir, 'under_observed_positions.npy'),
                        uo_world.astype(np.float32))

            self.get_logger().info(
                f'Saved NBV analysis to nbv/ '
                f'({len(nbv_candidates)} candidates, '
                f'top={nbv_candidates[0]["mean_alpha"]:.2f})')

            # ── 9f. Per-round convergence analysis + 3D plot ──
            self._generate_nbv_analysis()

            # ── 10. Summary text ──
            vram_final = self._get_vram_stats()
            summary_lines = [
                '=== ACTIVE RECON RUN SUMMARY ===',
                '',
                f'Run directory: {self.run_dir}',
                f'Date: {time.strftime("%Y-%m-%d %H:%M:%S")}',
                '',
                '--- GPU ---',
                f'GPU: {self.gpu_info["name"]}',
                f'VRAM Total: {self.gpu_info["total_vram_mb"]:.0f} MB',
                f'Compute Capability: {self.gpu_info.get("compute_capability", "?")}',
                f'SM Count: {self.gpu_info.get("sm_count", "?")}',
                f'Driver: {self.gpu_info.get("driver_version", "?")}',
                f'CUDA: {self.gpu_info["cuda_version"]}',
                f'PyTorch: {self.gpu_info["torch_version"]}',
                f'cuDNN: {self.gpu_info["cudnn_version"]}',
                '',
                '--- Mission ---',
                f'Duration: {(self._now_sec() - self.mission_start_time):.1f}s'
                if self.mission_start_time else 'Duration: N/A',
                f'Keyframes: {self.kf_count}',
                f'Passes: {self.passes}',
                f'Waypoints/pass: {self.n_wp}',
                '',
                '--- Training ---',
                f'Final Gaussians: {self.gs_model.n_gaussians}',
                f'Total train steps: {self.gs_model.total_train_steps}',
                f'Final loss: {self.gs_model.last_loss:.4f}',
                f'Avg PSNR: {np.mean([r.get("psnr", 0) for r in training_log_snap]):.2f} dB' if training_log_snap else '',
                f'Avg SSIM: {np.mean([r.get("ssim", 0) for r in training_log_snap]):.4f}' if training_log_snap else '',
                f'Iters/keyframe: {self.iters_per_kf}',
                f'Window size: {self.window_size}',
                f'Max Gaussians: {int(self.get_parameter("max_gaussians").value)}',
                '',
                '--- VRAM ---',
                f'Total: {self.gpu_info["total_vram_mb"]/1024:.1f} GB',
                f'Peak Allocated: {vram_final["peak_allocated_mb"]:.0f} MB ({vram_final["peak_allocated_mb"]/self.gpu_info["total_vram_mb"]*100:.1f}%)'
                if self.gpu_info["total_vram_mb"] > 0 else f'Peak Allocated: {vram_final["peak_allocated_mb"]:.0f} MB',
                f'Current Allocated: {vram_final["allocated_mb"]:.0f} MB ({vram_final["allocated_mb"]/self.gpu_info["total_vram_mb"]*100:.1f}%)'
                if self.gpu_info["total_vram_mb"] > 0 else f'Current Allocated: {vram_final["allocated_mb"]:.0f} MB',
                f'Reserved: {vram_final["reserved_mb"]:.0f} MB',
                '',
                '--- Coverage ---',
            ]
            stats = self.voxels.get_stats()
            summary_lines.extend([
                f'Occupied voxels: {stats["n_occupied"]}',
                f'Covered (2+ views): {stats["n_covered_2plus"]}',
                f'Covered (3+ views): {stats["n_covered_3plus"]}',
                f'Coverage %: {stats["coverage_pct"]:.1%}',
                f'Voxel resolution: {self.voxel_res}m',
                f'Bbox size: {self.bbox_size}m',
                '',
                '--- Timing ---',
            ])
            if training_log_snap:
                round_times = [r['round_total_ms'] for r in training_log_snap]
                per_iters = [r['train_per_iter_ms'] for r in training_log_snap]
                summary_lines.extend([
                    f'Avg round time: {np.mean(round_times):.0f} ms',
                    f'Min round time: {np.min(round_times):.0f} ms',
                    f'Max round time: {np.max(round_times):.0f} ms',
                    f'Avg per-iter time: {np.mean(per_iters):.1f} ms',
                ])
            if self.kf_metrics:
                kf_times = [m['total_ms'] for m in self.kf_metrics]
                summary_lines.extend([
                    f'Avg KF capture time: {np.mean(kf_times):.0f} ms',
                ])

            summary_lines.append('')
            summary_lines.append('--- Phase Durations ---')
            for phase, t in self.phase_times.items():
                if 'duration_s' in t:
                    summary_lines.append(
                        f'{phase}: {t["duration_s"]:.1f}s')

            if self.battery_log:
                summary_lines.append('')
                summary_lines.append('--- Battery ---')
                summary_lines.append(
                    f'Start: {self.battery_log[0]["remaining_pct"]:.1f}% '
                    f'({self.battery_log[0]["voltage_v"]:.2f}V)')
                summary_lines.append(
                    f'End: {self.battery_log[-1]["remaining_pct"]:.1f}% '
                    f'({self.battery_log[-1]["voltage_v"]:.2f}V)')
                consumed = (self.battery_log[0]["remaining_pct"]
                            - self.battery_log[-1]["remaining_pct"])
                summary_lines.append(f'Consumed: {consumed:.1f}%')

            # Reconstruction quality section
            if kf_metrics_list:
                all_psnr = [m['psnr'] for m in kf_metrics_list]
                all_ssim = [m['ssim'] for m in kf_metrics_list]
                all_l1 = [m['l1'] for m in kf_metrics_list]
                summary_lines.extend([
                    '',
                    '--- Reconstruction Quality ---',
                    f'Mean PSNR: {np.mean(all_psnr):.2f} dB',
                    f'Median PSNR: {np.median(all_psnr):.2f} dB',
                    f'Min/Max PSNR: {np.min(all_psnr):.2f} / {np.max(all_psnr):.2f} dB',
                    f'Std PSNR: {np.std(all_psnr):.2f} dB',
                    f'Mean SSIM: {np.mean(all_ssim):.4f}',
                    f'Median SSIM: {np.median(all_ssim):.4f}',
                    f'Min/Max SSIM: {np.min(all_ssim):.4f} / {np.max(all_ssim):.4f}',
                    f'Mean L1: {np.mean(all_l1):.6f}',
                ])
                all_lpips = [m['lpips'] for m in kf_metrics_list
                             if m['lpips'] >= 0]
                if all_lpips:
                    summary_lines.extend([
                        f'Mean LPIPS: {np.mean(all_lpips):.4f}',
                        f'Median LPIPS: {np.median(all_lpips):.4f}',
                        f'Min/Max LPIPS: {np.min(all_lpips):.4f} / {np.max(all_lpips):.4f}',
                    ])
                summary_lines.extend([
                    f'Evaluated on: {len(kf_metrics_list)} keyframes',
                ])
                for p in sorted(pass_stats.keys()):
                    if p < len(self.passes):
                        s = pass_stats[p]
                        summary_lines.append(
                            f'  Pass {p+1} ({self.passes[p][0]}m, '
                            f'{self.passes[p][1]}°): '
                            f'PSNR={np.mean(s["psnr"]):.1f} '
                            f'SSIM={np.mean(s["ssim"]):.3f} '
                            f'L1={np.mean(s["l1"]):.4f}')

            summary = '\n'.join(summary_lines)
            with open(os.path.join(self.eval_dir, 'summary.txt'), 'w') as f:
                f.write(summary)
            self.get_logger().info('Saved eval/summary.txt')
            self.get_logger().info(
                '=== Evaluation outputs complete. ===')

            # Offline reconstruction is now run separately via:
            #   bash postprocess.sh <run_dir>
            self.get_logger().info(
                f'Run dir: {self.run_dir}')
            self.get_logger().info(
                'Run offline reconstruction with: '
                'bash postprocess.sh <run_dir>')

            self.get_logger().info(
                '=== All done. Shutting down. ===')
            # Wait for optimizer thread to finish, then force exit
            if hasattr(self, 'opt_thread') and self.opt_thread.is_alive():
                self.opt_thread.join(timeout=15.0)
            os._exit(0)

        except Exception as e:
            self.get_logger().error(f'Evaluation/offline error: {e}')
            import traceback as tb
            self.get_logger().error(tb.format_exc())
            self.get_logger().info('Exiting despite eval error — data is saved')
            os._exit(0)

    # ── Per-round metrics ─────────────────────────────────

    def _save_round_metrics_bg(self, round_num, frames_snapshot, kf_count,
                                n_gaussians, model_snapshot=None):
        """Background: render vs GT images, error maps, PSNR/SSIM/L1 per KF.

        Runs in a daemon thread so it never blocks PX4 offboard heartbeats.
        Uses a snapshot of frames_meta/kf_count and model params taken at
        call time to avoid race conditions with the optimizer thread.
        """
        try:
            from skimage.metrics import structural_similarity as ssim_fn
            from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        except ImportError:
            self.get_logger().warning('skimage not available, skipping round metrics')
            return

        round_dir = os.path.join(self.nbv_dir, 'rounds',
                                 f'round_{round_num:02d}')
        os.makedirs(round_dir, exist_ok=True)
        renders_dir = os.path.join(round_dir, 'renders')
        os.makedirs(renders_dir, exist_ok=True)

        kf_metrics = []
        for idx in range(kf_count):
            frame = frames_snapshot[idx]
            gt_path = os.path.join(self.run_dir, frame['file_path'])
            gt_img = cv2.imread(gt_path)
            if gt_img is None:
                continue
            gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

            vm = GaussianModel3DGS.compute_viewmat(
                *frame['position_ned'],
                frame['heading'],
                frame['gimbal_pitch'])
            if model_snapshot is not None:
                rendered = self.gs_model.render_from_snapshot(vm, model_snapshot)
            else:
                rendered = self.gs_model.render(vm)

            gt_f = gt_rgb.astype(np.float64) / 255.0
            rd_f = rendered.astype(np.float64) / 255.0

            kf_psnr = float(psnr_fn(gt_f, rd_f, data_range=1.0))
            kf_ssim = float(ssim_fn(
                gt_f, rd_f, data_range=1.0, channel_axis=2))
            kf_l1 = float(np.mean(np.abs(gt_f - rd_f)))

            kf_metrics.append({
                'kf': idx, 'psnr': kf_psnr,
                'ssim': kf_ssim, 'l1': kf_l1,
            })

            # Save rendered + error for every keyframe
            W = gt_img.shape[1]
            rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            err = np.abs(gt_f - rd_f)
            err_vis = (err * 255 * 3).clip(0, 255).astype(np.uint8)
            err_bgr = cv2.cvtColor(err_vis, cv2.COLOR_RGB2BGR)

            # GT | Rendered | Error
            combined = np.hstack([gt_img, rendered_bgr, err_bgr])
            cv2.putText(combined, 'GT', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(combined, 'Rendered', (W + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(combined,
                        f'Err PSNR={kf_psnr:.1f} SSIM={kf_ssim:.3f} L1={kf_l1:.4f}',
                        (2 * W + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(renders_dir, f'kf_{idx:03d}.png'), combined)

        # Aggregate metrics for this round
        if kf_metrics:
            all_psnr = [m['psnr'] for m in kf_metrics]
            all_ssim = [m['ssim'] for m in kf_metrics]
            all_l1 = [m['l1'] for m in kf_metrics]
            summary = {
                'round': round_num,
                'n_kfs': kf_count,
                'n_gaussians': n_gaussians,
                'mean_psnr': float(np.mean(all_psnr)),
                'median_psnr': float(np.median(all_psnr)),
                'min_psnr': float(np.min(all_psnr)),
                'max_psnr': float(np.max(all_psnr)),
                'mean_ssim': float(np.mean(all_ssim)),
                'mean_l1': float(np.mean(all_l1)),
                'per_kf': kf_metrics,
            }
            with open(os.path.join(round_dir, 'metrics.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            self.get_logger().info(
                f'Round {round_num} quality: PSNR={np.mean(all_psnr):.1f} '
                f'SSIM={np.mean(all_ssim):.3f} L1={np.mean(all_l1):.4f} '
                f'({len(kf_metrics)} KFs)')

        # Model checkpoint
        ckpt_path = os.path.join(round_dir, 'model')
        self.gs_model.save(ckpt_path)
        self.get_logger().info(f'Saved round {round_num} checkpoint + metrics')

    # ── NBV scoring (bbox-alpha + angular diversity) ─────────

    def _run_nbv_scoring(self):
        """Score candidates and pick next batch using bbox-alpha + angular diversity.

        Called after seed completes and after every batch of captures.
        Each call re-scores all candidates with the latest model,
        picks the next batch_size viewpoints via greedy selection,
        and queues them.  Records comprehensive per-round data.
        """
        self.nbv_round += 1

        # Mark newly captured positions as visited
        n_already_visited = len(self.planner.visited_positions)
        for meta in self.frames_meta[n_already_visited:]:
            self.planner.mark_visited(
                np.array(meta['position_ned'], dtype=np.float32))

        remaining = self.kf_budget - self.kf_count
        batch = min(self.batch_size, remaining)
        if batch <= 0:
            self.get_logger().info('Budget exhausted, no more NBV scoring')
            return

        self.get_logger().info(
            f'NBV round {self.nbv_round}: scoring {self.planner.n_candidates} '
            f'candidates (model has {self.kf_count} KFs, '
            f'{len(self.planner.visited_positions)} visited)...')

        t_start = time.perf_counter()
        start_pos = None
        if self.pos is not None:
            start_pos = np.array([
                self.pos.x, self.pos.y, self.pos.z], dtype=np.float32)
        phase_progress = self.kf_count / self.kf_budget if self.kf_budget > 0 else 0.5
        top_k = self.planner.select_top_k(
            self.gs_model, k=batch, start_pos=start_pos,
            phase_progress=phase_progress)
        t_end = time.perf_counter()
        scoring_ms = (t_end - t_start) * 1000

        if not top_k:
            self.get_logger().warning('NBV scoring returned no candidates')
            return

        # Save raw planner analysis files per round
        self.planner.save_analysis(
            self.nbv_dir, suffix=f'_round{self.nbv_round:02d}')

        analysis = self.planner.last_analysis
        self.get_logger().info(
            f'Geo uncert: mean={analysis["geo_uncertainty_mean"]:.3f}, '
            f'max={analysis["geo_uncertainty_max"]:.3f}, '
            f'angular: mean={analysis["angular_dist_mean"]:.3f}')
        self.get_logger().info(
            f'Score: mean={analysis["score_mean"]:.3f}, '
            f'max={analysis["score_max"]:.3f}, '
            f'phase={phase_progress:.1%}')

        self.get_logger().info(
            f'Scored in {scoring_ms:.0f}ms, '
            f'picked {len(top_k)} for round {self.nbv_round}')

        # ── Save per-round rendered vs GT comparisons + error maps ──
        # Run in background thread to avoid blocking PX4 offboard heartbeats.
        # Take a model snapshot NOW (on the optimizer thread where tensors are
        # consistent) to avoid race conditions with densify/prune resizing.
        frames_snap = [f.copy() for f in self.frames_meta[:self.kf_count]]
        model_snapshot = self.gs_model.get_snapshot()
        threading.Thread(
            target=self._save_round_metrics_bg,
            args=(self.nbv_round, frames_snap, self.kf_count,
                  self.gs_model.n_gaussians, model_snapshot),
            daemon=True,
        ).start()

        # ── Record per-round data for analysis ──
        all_scores = []
        if self.planner.last_scored:
            for c in self.planner.last_scored:
                all_scores.append({
                    'index': c['index'],
                    'azimuth_deg': float(c['azimuth_deg']),
                    'elevation_deg': float(c['elevation_deg']),
                    'altitude': float(c['altitude']),
                    'score': float(c['score']),
                    'geo_uncertainty': float(c['geo_uncertainty']),
                    'score_bbox_uncov': float(c['score_bbox_uncov']),
                    'score_angular': float(c['score_angular']),
                    'bbox_area_frac': float(c['bbox_area_frac']),
                })

        selected_detail = []
        for vp in top_k:
            selected_detail.append({
                'index': vp['index'],
                'azimuth_deg': float(vp['azimuth_deg']),
                'elevation_deg': float(vp['elevation_deg']),
                'altitude': float(vp['altitude']),
                'position': [float(v) for v in vp['position']],
                'yaw': float(vp['yaw']),
                'gimbal_pitch': float(vp['gimbal_pitch']),
                'score': float(vp['score']),
                'geo_uncertainty': float(vp['geo_uncertainty']),
                'score_bbox_uncov': float(vp['score_bbox_uncov']),
                'score_angular': float(vp['score_angular']),
                'bbox_area_frac': float(vp['bbox_area_frac']),
            })

        latest_training = self.training_log[-1] if self.training_log else {}

        round_data = {
            'round': self.nbv_round,
            'n_kfs_at_scoring': self.kf_count,
            'n_kfs_after_batch': self.kf_count + len(top_k),
            'batch_size': len(top_k),
            'timestamp_s': self._now_sec() - (self.mission_start_time or 0),
            'scoring_compute_ms': scoring_ms,
            # Depth-variance geometric uncertainty + angular diversity
            'n_gaussians': analysis.get('n_gaussians_total', 0),
            'geo_uncertainty_mean': analysis.get('geo_uncertainty_mean', 0),
            'geo_uncertainty_max': analysis.get('geo_uncertainty_max', 0),
            'bbox_uncov_mean': analysis.get('bbox_uncov_mean', 0),
            'bbox_uncov_max': analysis.get('bbox_uncov_max', 0),
            'bbox_area_frac_mean': analysis.get('bbox_area_frac_mean', 0),
            'angular_dist_mean': analysis.get('angular_dist_mean', 0),
            'angular_dist_min': analysis.get('angular_dist_min', 0),
            'phase_progress': analysis.get('phase_progress', 0),
            # Score distribution
            'score_max': analysis.get('score_max', 0),
            'score_min': analysis.get('score_min', 0),
            'score_mean': analysis.get('score_mean', 0),
            'score_std': analysis.get('score_std', 0),
            # Coverage
            'coverage_pct': self.voxels.get_coverage_pct(),
            # Training state
            'training_loss': latest_training.get('loss_final', 0),
            'training_psnr': latest_training.get('psnr', 0),
            'training_ssim': latest_training.get('ssim', 0),
            'training_rounds_completed': len(self.training_log),
            # Visited positions
            'n_visited': len(self.planner.visited_positions),
            'visited_positions': [
                [float(v) for v in pos]
                for pos in self.planner.visited_positions],
            # Selected viewpoints
            'selected': selected_detail,
            # All candidate scores
            'all_candidate_scores': all_scores,
        }

        self.round_history.append(round_data)

        rounds_dir = os.path.join(self.nbv_dir, 'rounds')
        os.makedirs(rounds_dir, exist_ok=True)
        with open(os.path.join(
                rounds_dir,
                f'round_{self.nbv_round:02d}.json'), 'w') as f:
            json.dump(round_data, f, indent=2)

        # Queue this batch
        self.nbv_queue = list(top_k)
        self.nbv_batch_count = 0
        for i, vp in enumerate(self.nbv_queue[:6]):
            self.get_logger().info(
                f'  #{i+1}: az={vp["azimuth_deg"]:.0f}° '
                f'el={vp["elevation_deg"]:.0f}° '
                f'score={vp["score"]:.3f} '
                f'(geo={vp["geo_uncertainty"]:.3f} '
                f'ang={vp["score_angular"]:.3f})')

    def _generate_nbv_analysis(self):
        """Generate all NBV convergence plots and 3D visualization.

        Called once after mission completes. Uses self.round_history.
        """
        if not self.round_history:
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        analysis_dir = os.path.join(self.nbv_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)

        rounds = [r['round'] for r in self.round_history]
        n_rounds = len(rounds)

        # ── 1. Round summary CSV ──
        csv_path = os.path.join(self.nbv_dir, 'round_summary.csv')
        with open(csv_path, 'w') as f:
            f.write('round,n_kfs,coverage_pct,n_gaussians,'
                    'geo_uncertainty_mean,geo_uncertainty_max,'
                    'bbox_uncov_mean,bbox_uncov_max,angular_dist_mean,'
                    'score_max,score_min,score_mean,'
                    'loss,psnr,ssim,compute_ms\n')
            for r in self.round_history:
                f.write(f'{r["round"]},{r["n_kfs_at_scoring"]},'
                        f'{r["coverage_pct"]:.4f},{r["n_gaussians"]},'
                        f'{r.get("geo_uncertainty_mean", 0):.4f},'
                        f'{r.get("geo_uncertainty_max", 0):.4f},'
                        f'{r.get("bbox_uncov_mean", 0):.4f},'
                        f'{r.get("bbox_uncov_max", 0):.4f},'
                        f'{r.get("angular_dist_mean", 0):.4f},'
                        f'{r["score_max"]:.4f},{r["score_min"]:.4f},'
                        f'{r["score_mean"]:.4f},'
                        f'{r["training_loss"]:.6f},'
                        f'{r["training_psnr"]:.2f},'
                        f'{r["training_ssim"]:.4f},'
                        f'{r["scoring_compute_ms"]:.1f}\n')
        self.get_logger().info(f'Saved nbv/round_summary.csv')

        # ── 2. NBV signal convergence (bbox-alpha + angular diversity) ──
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1, ax2 = axes

        geo_uncert = [r.get('geo_uncertainty_mean', 0) for r in self.round_history]
        geo_uncert_max = [r.get('geo_uncertainty_max', 0) for r in self.round_history]
        bbox_uncov = [r.get('bbox_uncov_mean', 0) for r in self.round_history]
        angular_mean = [r.get('angular_dist_mean', 0) for r in self.round_history]
        angular_min = [r.get('angular_dist_min', 0) for r in self.round_history]
        score_mean = [r.get('score_mean', 0) for r in self.round_history]
        n_kfs = [r['n_kfs_at_scoring'] for r in self.round_history]

        ax1.plot(rounds, geo_uncert, 'r-o', label='Geo Uncert Mean', markersize=6)
        ax1.plot(rounds, geo_uncert_max, 'r--^', label='Geo Uncert Max',
                 markersize=5, alpha=0.6)
        ax1.plot(rounds, bbox_uncov, '--', color='gray', label='Bbox Uncov Mean (old)',
                 markersize=4, alpha=0.5)
        ax1.set_ylabel('Uncertainty (0=certain, high=uncertain)')
        ax1.set_title('NBV Signal Convergence (should decrease as model improves)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        for i, (rd, kf) in enumerate(zip(rounds, n_kfs)):
            ax1.annotate(f'{kf}kf', (rd, geo_uncert[i]),
                         textcoords='offset points', xytext=(0, 10),
                         fontsize=7, color='gray', ha='center')

        ax2.plot(rounds, angular_mean, 'm-o', label='Angular Dist Mean', markersize=6)
        ax2.plot(rounds, angular_min, 'm--s', label='Angular Dist Min',
                 markersize=5, alpha=0.6)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(rounds, score_mean, 'b-^', label='Combined Score Mean', markersize=6)
        ax2_twin.set_ylabel('Combined Score', color='blue')
        ax2.set_xlabel('Scoring Round')
        ax2.set_ylabel('Angular Distance (0=redundant, 1=novel)')
        ax2.set_title('Angular Diversity + Combined Score')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'nbv_signal_convergence.png'),
                    dpi=150)
        plt.close()

        # ── 3. Score distribution shift (box plot per round) ──
        fig, ax = plt.subplots(figsize=(max(12, n_rounds * 1.5), 6))
        score_data = []
        labels = []
        for r in self.round_history:
            scores = [c['score'] for c in r['all_candidate_scores']]
            if scores:
                score_data.append(scores)
                labels.append(f'R{r["round"]}\n({r["n_kfs_at_scoring"]}kf)')

        if score_data:
            bp = ax.boxplot(score_data, tick_labels=labels, patch_artist=True)
            colors = plt.cm.viridis(np.linspace(0, 1, len(score_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            # Mark selected scores
            for i, r in enumerate(self.round_history):
                sel_scores = [s['score'] for s in r['selected']]
                ax.scatter([i + 1] * len(sel_scores), sel_scores,
                           color='red', zorder=5, s=50, marker='*',
                           label='Selected' if i == 0 else None)
            ax.set_xlabel('Scoring Round (KFs at time of scoring)')
            ax.set_ylabel('Candidate Score')
            ax.set_title('Score Distribution Shift Across Rounds '
                         '(red stars = selected)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'score_distribution.png'),
                    dpi=150)
        plt.close()

        # ── 4. Coverage + training metrics vs round ──
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        coverage = [r['coverage_pct'] * 100 for r in self.round_history]
        axes[0, 0].plot(rounds, coverage, 'g-o', markersize=6)
        axes[0, 0].set_ylabel('Coverage %')
        axes[0, 0].set_title('Voxel Coverage vs Round')
        axes[0, 0].grid(True, alpha=0.3)

        loss = [r['training_loss'] for r in self.round_history]
        axes[0, 1].plot(rounds, loss, 'b-o', markersize=6)
        axes[0, 1].set_ylabel('Training Loss')
        axes[0, 1].set_title('Training Loss at Each Scoring')
        axes[0, 1].grid(True, alpha=0.3)

        psnr = [r['training_psnr'] for r in self.round_history]
        axes[1, 0].plot(rounds, psnr, 'r-o', markersize=6)
        axes[1, 0].set_xlabel('Scoring Round')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].set_title('Live PSNR at Each Scoring')
        axes[1, 0].grid(True, alpha=0.3)

        n_gs = [r['n_gaussians'] / 1000 for r in self.round_history]
        axes[1, 1].plot(rounds, n_gs, 'purple', marker='o', markersize=6)
        axes[1, 1].set_xlabel('Scoring Round')
        axes[1, 1].set_ylabel('Gaussians (K)')
        axes[1, 1].set_title('Gaussian Count at Each Scoring')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'training_convergence.png'),
                    dpi=150)
        plt.close()

        # ── 5. Viewpoint selection polar plot (azimuth x elevation) ──
        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = plt.cm.tab10
        for r in self.round_history:
            rd = r['round']
            color = cmap(rd % 10)
            for s in r['selected']:
                ax.scatter(s['azimuth_deg'], s['elevation_deg'],
                           c=[color], s=120, zorder=5,
                           edgecolors='black', linewidths=0.5,
                           label=f'Round {rd}' if s == r['selected'][0] else None)
        # Plot seed positions
        for i, meta in enumerate(self.frames_meta[:self.seed_kfs]):
            pos = meta['position_ned']
            dx = pos[0] - self.rock_ned[0]
            dy = pos[1] - self.rock_ned[1]
            dz = pos[2] - self.rock_ned[2]
            horiz = math.sqrt(dx * dx + dy * dy)
            el = math.degrees(math.atan2(-dz, horiz))
            az = math.degrees(math.atan2(dy, dx))
            if az < 0:
                az += 360
            ax.scatter(az, el, c='black', s=150, marker='D', zorder=6,
                       label='Seed' if i == 0 else None)

        # Plot all candidates as faint dots
        if self.round_history and self.round_history[0]['all_candidate_scores']:
            for c in self.round_history[0]['all_candidate_scores']:
                ax.scatter(c['azimuth_deg'], c['elevation_deg'],
                           c='lightgray', s=20, zorder=1, alpha=0.5)

        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel('Elevation (degrees)')
        ax.set_title('Viewpoint Selection by Round '
                     '(gray = candidates, black diamonds = seed)')
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 90)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'viewpoint_selection_map.png'),
                    dpi=150)
        plt.close()

        # ── 6. Scoring compute time per round ──
        fig, ax = plt.subplots(figsize=(10, 4))
        compute_ms = [r['scoring_compute_ms'] for r in self.round_history]
        ax.bar(rounds, compute_ms, color='steelblue', alpha=0.8)
        ax.set_xlabel('Scoring Round')
        ax.set_ylabel('Compute Time (ms)')
        ax.set_title('Scoring Compute Time per Round')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'scoring_compute.png'),
                    dpi=150)
        plt.close()

        # ── 7. 3D interactive plot (Plotly HTML) ──
        self._generate_3d_plotly(analysis_dir)

        # Save full round history as single JSON
        with open(os.path.join(self.nbv_dir, 'round_history.json'), 'w') as f:
            json.dump(self.round_history, f, indent=2)

        self.get_logger().info(
            f'Saved NBV analysis: {n_rounds} rounds to nbv/analysis/')

    def _generate_3d_plotly(self, output_dir):
        """Generate interactive 3D Plotly visualization of viewpoints."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            self.get_logger().warning(
                'plotly not installed, skipping 3D visualization. '
                'Install with: pip install plotly')
            return

        fig = go.Figure()

        rock = self.rock_ned
        half = self.bbox_size / 2.0

        # Bounding box wireframe
        corners = np.array([
            [rock[0] - half, rock[1] - half, rock[2] - half],
            [rock[0] + half, rock[1] - half, rock[2] - half],
            [rock[0] + half, rock[1] + half, rock[2] - half],
            [rock[0] - half, rock[1] + half, rock[2] - half],
            [rock[0] - half, rock[1] - half, rock[2] + half],
            [rock[0] + half, rock[1] - half, rock[2] + half],
            [rock[0] + half, rock[1] + half, rock[2] + half],
            [rock[0] - half, rock[1] + half, rock[2] + half],
        ])
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
        ]
        for i, j in edges:
            fig.add_trace(go.Scatter3d(
                x=[corners[i, 0], corners[j, 0]],
                y=[corners[i, 1], corners[j, 1]],
                z=[-corners[i, 2], -corners[j, 2]],  # flip z for display
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip',
            ))

        # Rock center
        fig.add_trace(go.Scatter3d(
            x=[rock[0]], y=[rock[1]], z=[-rock[2]],
            mode='markers',
            marker=dict(size=8, color='black', symbol='diamond'),
            name='Rock center',
            hovertext='Rock center',
        ))

        # Seed viewpoints
        seed_positions = []
        for meta in self.frames_meta[:self.seed_kfs]:
            p = meta['position_ned']
            seed_positions.append(p)
        if seed_positions:
            sp = np.array(seed_positions)
            fig.add_trace(go.Scatter3d(
                x=sp[:, 0], y=sp[:, 1], z=-sp[:, 2],
                mode='markers',
                marker=dict(size=7, color='black', symbol='diamond'),
                name='Seed',
                hovertext=[f'Seed KF {i}' for i in range(len(sp))],
            ))
            # Lines from seed to rock
            for p in seed_positions:
                fig.add_trace(go.Scatter3d(
                    x=[p[0], rock[0]], y=[p[1], rock[1]],
                    z=[-p[2], -rock[2]],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False, hoverinfo='skip',
                ))

        # Round viewpoints (colored by round)
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
        ]
        for r in self.round_history:
            rd = r['round']
            color = colors[rd % len(colors)]
            positions = np.array([s['position'] for s in r['selected']])
            hover = [
                f'Round {rd}, KF {r["n_kfs_at_scoring"]+i}<br>'
                f'Az={s["azimuth_deg"]:.0f}° El={s["elevation_deg"]:.0f}°<br>'
                f'Score={s["score"]:.3f} (geo={s.get("geo_uncertainty", 0):.3f}, ang={s.get("score_angular", 0):.3f})<br>'
                f'BboxArea={s.get("bbox_area_frac", 0):.2f}'
                for i, s in enumerate(r['selected'])
            ]
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0], y=positions[:, 1], z=-positions[:, 2],
                mode='markers',
                marker=dict(size=6, color=color),
                name=f'Round {rd} ({r["n_kfs_at_scoring"]}kf)',
                hovertext=hover,
            ))
            # Lines to rock center
            for p in positions:
                fig.add_trace(go.Scatter3d(
                    x=[p[0], rock[0]], y=[p[1], rock[1]],
                    z=[-p[2], -rock[2]],
                    mode='lines',
                    line=dict(color=color, width=1),
                    opacity=0.3,
                    showlegend=False, hoverinfo='skip',
                ))

        # All candidates as faint dots
        if (self.round_history
                and self.round_history[0]['all_candidate_scores']):
            # Get positions from the planner candidates
            cand_positions = np.array([
                c['position'] for c in self.planner.candidates],
                dtype=np.float32)
            fig.add_trace(go.Scatter3d(
                x=cand_positions[:, 0],
                y=cand_positions[:, 1],
                z=-cand_positions[:, 2],
                mode='markers',
                marker=dict(size=2, color='lightgray', opacity=0.3),
                name='All candidates',
                hoverinfo='skip',
            ))

        fig.update_layout(
            title=f'NBV Viewpoint Selection — {self.kf_count} KFs, '
                  f'{len(self.round_history)} rounds',
            scene=dict(
                xaxis_title='X (NED North)',
                yaxis_title='Y (NED East)',
                zaxis_title='Altitude (m, up)',
                aspectmode='data',
            ),
            width=1200, height=800,
            legend=dict(x=0.02, y=0.98),
        )

        html_path = os.path.join(output_dir, 'viewpoints_3d.html')
        fig.write_html(html_path)
        self.get_logger().info(f'Saved 3D viewpoint plot: {html_path}')

    # ── Flight state machine (main loop) ───────────────────

    def _loop(self):
        # DONE: fully stopped. FINISHING: waiting for optimizer, no offboard needed.
        if self.phase == 'DONE':
            return
        if self.phase == 'FINISHING':
            # Still check if optimizer finished, but don't publish offboard
            if not self.opt_thread.is_alive():
                self._save_all()
                self._enter_phase('DONE')
                self.get_logger().info('Data saved. Done.')
            return

        self._pub_offboard()
        if self.pos is None or self.status is None:
            return

        # Gimbal — publish every tick like mapping package (proven stable).
        self._pub_gimbal(self.gimbal_pitch_rad)

        # ── PREFLIGHT ──
        if self.phase == 'PREFLIGHT':
            # On first tick with valid heading, calibrate NED frame offset.
            # Drone spawns facing Gazebo +X = East → true NED heading = π/2.
            # Any difference means PX4's NED is rotated from the Gazebo world.
            # Rotate rock_ned into PX4's frame so all navigation is consistent.
            if not self._ned_calibrated and self.counter >= 20:
                expected = math.pi / 2.0
                offset = self.pos.heading - expected
                self.get_logger().info(
                    f'NED heading offset {math.degrees(offset):.1f}° '
                    f'(not rotating — gimbal handles centering)')
                self._ned_calibrated = True

            yaw = math.atan2(self.rock_y, self.rock_x)
            takeoff_alt = -1.5
            self._pub_setpoint(0.0, 0.0, takeoff_alt, yaw, max_speed=None)
            self.counter += 1
            if self.counter >= 40:
                self._offboard()
                self._arm()
                self.mission_start_time = self._now_sec()
                self._enter_phase('TAKEOFF')
                self.get_logger().info('TAKEOFF to 1.5m')

        # ── TAKEOFF ──
        elif self.phase == 'TAKEOFF':
            yaw = math.atan2(self.rock_y, self.rock_x)
            takeoff_alt = -1.5
            self._pub_setpoint(0.0, 0.0, takeoff_alt, yaw, max_speed=None)
            if self._dist_to(0.0, 0.0, takeoff_alt) < 0.5:
                # Build seed waypoints: seed_kfs points spread evenly
                self._setup_pass(0)
                self.waypoints = []
                for i in range(self.seed_kfs):
                    angle = 2.0 * math.pi * i / self.seed_kfs
                    x = self.rock_x + self.radius * math.cos(angle)
                    y = self.rock_y + self.radius * math.sin(angle)
                    z = self.alt_ned
                    yw = math.atan2(self.rock_y - y, self.rock_x - x)
                    self.waypoints.append((x, y, z, yw))
                self.n_wp = self.seed_kfs
                self.wp_idx = 0
                # Fly to first seed waypoint at full speed (no speed limit)
                self._enter_phase('FLY_TO_START')
                self.get_logger().info(
                    f'Flying to first seed waypoint at '
                    f'({self.waypoints[0][0]:.1f}, '
                    f'{self.waypoints[0][1]:.1f})')

        # ── FLY_TO_START: get to first seed waypoint fast ──
        elif self.phase == 'FLY_TO_START':
            x, y, z, yaw = self.waypoints[0]
            self._pub_setpoint(x, y, z, yaw, max_speed=8.0)
            if self._dist_to(x, y, z) < 0.5:
                self._enter_phase('SEED')
                self.get_logger().info(
                    f'SEED: {self.seed_kfs} viewpoints at '
                    f'alt={self.orbit_alt}m, radius={self.radius}m')

        # ── SEED: capture initial views to bootstrap 3DGS ──
        elif self.phase == 'SEED':
            if self.wp_idx >= len(self.waypoints):
                if self.skip_adaptive:
                    self.get_logger().info(
                        f'Seed complete ({self.kf_count} KFs), '
                        f'skip_adaptive=True — returning to land')
                    self._enter_phase('RETURN')
                    return
                self.get_logger().info(
                    f'Seed complete ({self.kf_count} KFs), '
                    f'waiting for training before first scoring...')
                self._enter_phase('NBV_WAIT')
                return

            x, y, z, yaw = self.waypoints[self.wp_idx]
            # Fixed setpoint — PX4 handles approach.
            self._pub_setpoint(x, y, z, yaw)

            if self._dist_to(x, y, z) < 0.5:
                yaw_ok = self._yaw_error_to_rock() < math.radians(5)
                if not yaw_ok:
                    self.settle_start = None
                    self._gimbal_set = False
                elif not getattr(self, '_gimbal_set', False):
                    # Arrived & facing rock — compute exact pitch ONCE
                    self._set_gimbal_to_rock()
                    self._gimbal_set = True
                    self.settle_start = self._sim_sec()
                elif (self._sim_sec() - self.settle_start) >= self.settle_time:
                    self._capture_keyframe(yaw)
                    self.settle_start = None
                    self._gimbal_set = False
                    self.wp_idx += 1
                    self.get_logger().info(
                        f'Seed {self.wp_idx}/{len(self.waypoints)}')

        # ── NBV_WAIT: wait for optimizer to process before scoring ──
        elif self.phase == 'NBV_WAIT':
            # Hold position at last waypoint or last NBV target
            if self.nbv_target is not None:
                t = self.nbv_target
                self._pub_setpoint(
                    float(t['position'][0]), float(t['position'][1]),
                    float(t['position'][2]), float(t['yaw']))
            elif self.waypoints:
                x, y, z, yaw = self.waypoints[-1]
                self._pub_setpoint(x, y, z, yaw)

            # Wait for optimizer to have processed all queued KFs
            if (self.kf_queue.empty() and len(self.training_log) > 0):
                self._run_nbv_scoring()

                if self.nbv_queue:
                    self.nbv_target = self.nbv_queue.pop(0)
                    self.gimbal_pitch_rad = self._pitch_for_elevation(
                        self.nbv_target['elevation_deg'])
                    self.avoid_waypoint = self._path_needs_avoidance(
                        self.nbv_target['position'])
                    self.avoid_waypoint_transit = None
                    self.avoid_approach = None
                    self._enter_phase('NBV_FLY')
                    self.get_logger().info(
                        f'NBV round {self.nbv_round}: {len(self.nbv_queue)+1} '
                        f'viewpoints, first: '
                        f'az={self.nbv_target["azimuth_deg"]:.0f}° '
                        f'el={self.nbv_target["elevation_deg"]:.0f}°')
                else:
                    self.get_logger().info('No NBV candidates, returning')
                    self._enter_phase('RETURN')

        # ── NBV_FLY: fly to current NBV target ──
        # 4-phase path to avoid rock collisions:
        #   Phase 1: Climb at current XY to safe altitude
        #   Phase 2: Transit at safe altitude to target XY (or cylinder edge)
        #   Phase 3: If target is above rock, descend at cylinder edge,
        #            then fly inward to target at target altitude
        #   Phase 4: Final approach to target position
        elif self.phase == 'NBV_FLY':
            if self.nbv_target is None:
                self._enter_phase('RETURN')
                return

            # Phase 1: Climb at current XY to safe altitude
            if self.avoid_waypoint is not None:
                ax, ay, az_wp, a_yaw = self.avoid_waypoint
                self._pub_setpoint(ax, ay, az_wp, a_yaw)
                if self._dist_to(ax, ay, az_wp) < 0.8:
                    t = self.nbv_target
                    tx, ty, tz = t['position']
                    # Check if target is above rock — transit to cylinder
                    # edge at safe alt, not directly to target XY
                    above = self._target_above_rock(t['position'])
                    if above is not None:
                        # Transit to cylinder edge at safe alt
                        self.avoid_waypoint_transit = (
                            above[0], above[1], az_wp, float(t['yaw']))
                        self.avoid_approach = above  # descend here, then fly in
                    else:
                        # Target is clear — transit directly to target XY
                        self.avoid_waypoint_transit = (
                            float(tx), float(ty), az_wp, float(t['yaw']))
                        self.avoid_approach = None
                    self.avoid_waypoint = None
                return

            # Phase 2: Fly at safe altitude to target XY (or cylinder edge)
            if self.avoid_waypoint_transit is not None:
                tx, ty, tz_safe, t_yaw = self.avoid_waypoint_transit
                self._pub_setpoint(tx, ty, tz_safe, t_yaw)
                if self._dist_to(tx, ty, tz_safe) < 0.8:
                    self.avoid_waypoint_transit = None
                return

            # Phase 3: If approaching from cylinder edge, descend at edge
            # then fly inward to target at target altitude
            if hasattr(self, 'avoid_approach') and \
                    self.avoid_approach is not None:
                ax, ay, az_app, a_yaw = self.avoid_approach
                self._pub_setpoint(ax, ay, az_app, a_yaw)
                if self._dist_to(ax, ay, az_app) < 0.5:
                    self.avoid_approach = None  # now fly to actual target
                return

            # Phase 4: Final approach to target position
            t = self.nbv_target
            x, y, z = t['position']
            yaw = t['yaw']
            self._pub_setpoint(
                float(x), float(y), float(z), float(yaw))

            if self._dist_to(float(x), float(y), float(z)) < 0.5:
                self.settle_start = self._sim_sec()
                self._enter_phase('NBV_SETTLE')

        # ── NBV_SETTLE: wait, capture, next target or re-score ──
        elif self.phase == 'NBV_SETTLE':
            if self.nbv_target is None:
                self._enter_phase('RETURN')
                return

            t = self.nbv_target
            x, y, z = t['position']
            # Fixed setpoint — no per-tick recomputation
            self._pub_setpoint(
                float(x), float(y), float(z), float(t['yaw']))

            yaw_ok = self._yaw_error_to_rock() < math.radians(5)
            if not yaw_ok:
                self.settle_start = self._sim_sec()
                self._gimbal_set = False
            elif not getattr(self, '_gimbal_set', False):
                # Arrived & facing rock — compute exact pitch ONCE
                self._set_gimbal_to_rock()
                self._gimbal_set = True
                self.settle_start = self._sim_sec()

            # Wait for gimbal to converge after pitch was set
            nbv_settle = max(self.settle_time, 1.5)
            if (yaw_ok and self._gimbal_set and
                    (self._sim_sec() - self.settle_start) >= nbv_settle):
                self._capture_keyframe(float(t['yaw']))
                self.planner.mark_visited(t['position'])
                self.nbv_batch_count += 1

                self.get_logger().info(
                    f'NBV KF {self.kf_count}: '
                    f'az={t["azimuth_deg"]:.0f}° '
                    f'el={t["elevation_deg"]:.0f}° '
                    f'score={t.get("score", 0):.1f} '
                    f'(batch {self.nbv_batch_count}/{self.batch_size}, '
                    f'round {self.nbv_round})')

                # Budget exhausted?
                if self.kf_count >= self.kf_budget:
                    self.get_logger().info(
                        f'Budget reached ({self.kf_count} total KFs)')
                    self._enter_phase('RETURN')
                # Batch complete? Re-score with updated model
                elif self.nbv_batch_count >= self.batch_size:
                    self.get_logger().info(
                        f'Batch complete ({self.nbv_batch_count} KFs), '
                        f're-scoring with {self.kf_count}-image model...')
                    self._enter_phase('NBV_WAIT')
                # More in current batch
                elif self.nbv_queue:
                    self.nbv_target = self.nbv_queue.pop(0)
                    self.gimbal_pitch_rad = self._pitch_for_elevation(
                        self.nbv_target['elevation_deg'])
                    self.avoid_waypoint = self._path_needs_avoidance(
                        self.nbv_target['position'])
                    self.avoid_waypoint_transit = None
                    self.avoid_approach = None
                    self._enter_phase('NBV_FLY')
                else:
                    # Batch queue empty but count not reached — re-score
                    self.get_logger().info('Batch queue empty, re-scoring...')
                    self._enter_phase('NBV_WAIT')

        # ── RETURN ──
        elif self.phase == 'RETURN':
            # Fly back to origin at safe altitude before landing
            self._pub_setpoint(0.0, 0.0, -1.0, 0.0, max_speed=None)
            if self._dist_to(0.0, 0.0, -1.0) < 1.0:
                self._land()
                self._enter_phase('LANDING')
                self.get_logger().info('LANDING')

        # ── LANDING ──
        elif self.phase == 'LANDING':
            if self.pos.z > -0.3:
                self._disarm()
                # Signal optimizer to finish current round then stop
                self._enter_phase('FINISHING')
                self.get_logger().info(
                    'Landed. Waiting for optimizer to finish current round...')

        # ── FINISHING: handled at top of _loop ──


def main():
    rclpy.init()
    node = ActiveMapperNode()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        if node.viewer is not None:
            node.viewer.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
