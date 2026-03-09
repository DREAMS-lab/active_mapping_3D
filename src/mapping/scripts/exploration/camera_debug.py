#!/usr/bin/env python3
"""Debug: show what the camera sees and test gimbal commands.

Usage:
  python3 camera_debug.py              # just view camera
  python3 camera_debug.py --pitch -0.5 # set gimbal pitch to -0.5 rad (~-29°)
"""
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2


class CameraDebug(Node):
    def __init__(self, pitch=None):
        super().__init__('camera_debug')
        self.bridge = CvBridge()
        self.pitch = pitch
        self.frame_count = 0

        self.create_subscription(Image, '/rgbd/image', self._img_cb, 10)

        if pitch is not None:
            self.pitch_pub = self.create_publisher(Float64, '/gimbal/pitch', 10)
            self.yaw_pub = self.create_publisher(Float64, '/gimbal/yaw', 10)
            self.create_timer(0.1, self._gimbal_cmd)
            self.get_logger().info(f'Commanding gimbal pitch={pitch:.2f} rad ({pitch*57.3:.1f}°)')

    def _gimbal_cmd(self):
        self.pitch_pub.publish(Float64(data=self.pitch))
        self.yaw_pub.publish(Float64(data=0.0))

    def _img_cb(self, msg):
        self.frame_count += 1
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imshow('Camera View', img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            raise SystemExit(0)
        if self.frame_count % 30 == 0:
            cv2.imwrite('/tmp/camera_debug.png', img)
            self.get_logger().info(f'Frame {self.frame_count} saved to /tmp/camera_debug.png')


def main():
    rclpy.init()
    pitch = None
    if '--pitch' in sys.argv:
        idx = sys.argv.index('--pitch')
        pitch = float(sys.argv[idx + 1])
    node = CameraDebug(pitch)
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
