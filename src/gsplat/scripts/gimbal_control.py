#!/usr/bin/env python3
"""Gimbal control GUI with sliders for pitch, roll, yaw."""

import math
import threading
import tkinter as tk

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class GimbalControl(Node):
    def __init__(self):
        super().__init__('gimbal_control')
        self.pub_pitch = self.create_publisher(Float64, '/gimbal/pitch', 10)
        self.pub_roll = self.create_publisher(Float64, '/gimbal/roll', 10)
        self.pub_yaw = self.create_publisher(Float64, '/gimbal/yaw', 10)

    def publish(self, pitch, roll, yaw):
        self.pub_pitch.publish(Float64(data=pitch))
        self.pub_roll.publish(Float64(data=roll))
        self.pub_yaw.publish(Float64(data=yaw))


def main():
    rclpy.init()
    node = GimbalControl()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    root = tk.Tk()
    root.title('Gimbal Control')
    root.geometry('400x280')
    root.configure(bg='#2b2b2b')

    label_style = {'bg': '#2b2b2b', 'fg': '#e0e0e0'}
    scale_style = {'bg': '#2b2b2b', 'fg': '#e0e0e0', 'troughcolor': '#404040',
                   'highlightthickness': 0}

    def on_change(_=None):
        pitch = math.radians(pitch_var.get())
        roll = math.radians(roll_var.get())
        yaw = math.radians(yaw_var.get())
        node.publish(pitch, roll, yaw)
        pitch_label.config(text=f'Pitch: {pitch_var.get():.0f} deg')
        roll_label.config(text=f'Roll:  {roll_var.get():.0f} deg')
        yaw_label.config(text=f'Yaw:   {yaw_var.get():.0f} deg')

    def reset():
        pitch_var.set(0.0)
        roll_var.set(0.0)
        yaw_var.set(0.0)
        on_change()

    pitch_var = tk.DoubleVar(value=0.0)
    roll_var = tk.DoubleVar(value=0.0)
    yaw_var = tk.DoubleVar(value=0.0)

    pitch_label = tk.Label(root, text='Pitch: 0 deg', **label_style)
    pitch_label.pack(anchor='w', padx=10, pady=(10, 0))
    tk.Scale(root, variable=pitch_var, from_=-135, to=45,
             orient='horizontal', length=360, command=on_change,
             **scale_style).pack(padx=10)

    roll_label = tk.Label(root, text='Roll:  0 deg', **label_style)
    roll_label.pack(anchor='w', padx=10)
    tk.Scale(root, variable=roll_var, from_=-45, to=45,
             orient='horizontal', length=360, command=on_change,
             **scale_style).pack(padx=10)

    yaw_label = tk.Label(root, text='Yaw:   0 deg', **label_style)
    yaw_label.pack(anchor='w', padx=10)
    tk.Scale(root, variable=yaw_var, from_=-180, to=180,
             orient='horizontal', length=360, command=on_change,
             **scale_style).pack(padx=10)

    tk.Button(root, text='Reset', command=reset, bg='#404040', fg='#e0e0e0',
              activebackground='#505050').pack(pady=10)

    def on_close():
        root.destroy()
        node.destroy_node()
        rclpy.shutdown()

    root.protocol('WM_DELETE_WINDOW', on_close)
    root.mainloop()


if __name__ == '__main__':
    main()
