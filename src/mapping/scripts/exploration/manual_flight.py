#!/usr/bin/env python3
"""Manual flight + gimbal GUI. Take off to 2m, buttons for altitude, gimbal sliders."""

import math
import threading
import tkinter as tk

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
)
from std_msgs.msg import Float64


class ManualFlight(Node):
    def __init__(self):
        super().__init__('manual_flight')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.gimbal_pitch_pub = self.create_publisher(Float64, '/gimbal/pitch', 10)
        self.gimbal_roll_pub = self.create_publisher(Float64, '/gimbal/roll', 10)
        self.gimbal_yaw_pub = self.create_publisher(Float64, '/gimbal/yaw', 10)

        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self._pos_cb, qos)

        self.pos = None
        self.home_x = None
        self.home_y = None
        self.target_alt = -2.0  # NED
        self.armed = False
        self.counter = 0

        self.timer = self.create_timer(0.05, self._loop)

    def _pos_cb(self, msg):
        self.pos = msg
        if self.home_x is None:
            self.home_x = msg.x
            self.home_y = msg.y

    def _pub_offboard(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def _pub_setpoint(self):
        msg = TrajectorySetpoint()
        x = self.home_x if self.home_x is not None else 0.0
        y = self.home_y if self.home_y is not None else 0.0
        msg.position = [x, y, self.target_alt]
        msg.velocity = [float('nan')] * 3
        msg.acceleration = [float('nan')] * 3
        msg.jerk = [float('nan')] * 3
        msg.yaw = 1.5708  # face rock at NED (0, 3) = 90° East
        msg.yawspeed = float('nan')
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def _send_cmd(self, command, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = p1
        msg.param2 = p2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmd_pub.publish(msg)

    def arm_and_offboard(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, p1=1.0, p2=6.0)
        self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, p1=1.0)
        self.armed = True
        self.get_logger().info('ARM + OFFBOARD')

    def land(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('LAND')

    def set_alt(self, alt_m):
        self.target_alt = -abs(alt_m)
        self.get_logger().info(f'Target alt: {abs(self.target_alt):.1f}m')

    def pub_gimbal(self, pitch_rad, roll_rad, yaw_rad):
        self.gimbal_pitch_pub.publish(Float64(data=pitch_rad))
        self.gimbal_roll_pub.publish(Float64(data=roll_rad))
        self.gimbal_yaw_pub.publish(Float64(data=yaw_rad))

    def _loop(self):
        self._pub_offboard()
        self._pub_setpoint()
        self.counter += 1
        if not self.armed and self.counter >= 40 and self.pos is not None:
            self.arm_and_offboard()


def main():
    rclpy.init()
    node = ManualFlight()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    root = tk.Tk()
    root.title('Manual Flight + Gimbal')
    root.geometry('420x500')
    root.configure(bg='#2b2b2b')

    label_style = {'bg': '#2b2b2b', 'fg': '#e0e0e0', 'font': ('monospace', 11)}
    btn_style = {'bg': '#404040', 'fg': '#e0e0e0', 'activebackground': '#505050', 'font': ('monospace', 10)}
    scale_style = {'bg': '#2b2b2b', 'fg': '#e0e0e0', 'troughcolor': '#404040', 'highlightthickness': 0}

    # ── Altitude ──
    tk.Label(root, text='── ALTITUDE ──', **label_style).pack(pady=(10, 5))
    alt_frame = tk.Frame(root, bg='#2b2b2b')
    alt_frame.pack()

    alt_label = tk.Label(root, text='Alt: 2.0m', **label_style)
    alt_label.pack()

    for alt in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        tk.Button(alt_frame, text=f'{alt}m', width=5,
                  command=lambda a=alt: (node.set_alt(a), alt_label.config(text=f'Alt: {a}m')),
                  **btn_style).pack(side='left', padx=2)

    tk.Button(root, text='LAND', command=node.land, bg='#802020', fg='white',
              activebackground='#a03030', font=('monospace', 11, 'bold'), width=10).pack(pady=5)

    # ── Gimbal ──
    tk.Label(root, text='── GIMBAL ──', **label_style).pack(pady=(10, 0))

    pitch_var = tk.DoubleVar(value=0.0)
    roll_var = tk.DoubleVar(value=0.0)
    yaw_var = tk.DoubleVar(value=0.0)

    def on_gimbal(_=None):
        p = math.radians(pitch_var.get())
        r = math.radians(roll_var.get())
        y = math.radians(yaw_var.get())
        node.pub_gimbal(p, r, y)
        pitch_lbl.config(text=f'Pitch: {pitch_var.get():+.0f}°  ({p:+.2f} rad)')
        yaw_lbl.config(text=f'Yaw:   {yaw_var.get():+.0f}°  ({y:+.2f} rad)')

    pitch_lbl = tk.Label(root, text='Pitch: +0°  (+0.00 rad)', **label_style)
    pitch_lbl.pack(anchor='w', padx=10)
    tk.Scale(root, variable=pitch_var, from_=-135, to=45,
             orient='horizontal', length=380, command=on_gimbal,
             resolution=1, **scale_style).pack(padx=10)

    yaw_lbl = tk.Label(root, text='Yaw:   +0°  (+0.00 rad)', **label_style)
    yaw_lbl.pack(anchor='w', padx=10)
    tk.Scale(root, variable=yaw_var, from_=-180, to=180,
             orient='horizontal', length=380, command=on_gimbal,
             resolution=1, **scale_style).pack(padx=10)

    # ── Status ──
    status_label = tk.Label(root, text='', **label_style)
    status_label.pack(pady=5)

    def update_status():
        if node.pos is not None:
            z = -node.pos.z
            status_label.config(text=f'Pos: ({node.pos.x:.1f}, {node.pos.y:.1f})  Alt: {z:.1f}m')
        root.after(200, update_status)
    update_status()

    def on_close():
        root.destroy()
        node.destroy_node()
        rclpy.shutdown()

    root.protocol('WM_DELETE_WINDOW', on_close)
    root.mainloop()


if __name__ == '__main__':
    main()
