#!/usr/bin/env python3
"""RViz2 visualization helpers for active reconstruction.

Publishes voxel grid, bounding box, camera poses, and status text.
"""

import numpy as np

from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, ColorRGBA
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration


class RVizPublisher:
    """Publishes active reconstruction state to RViz2."""

    def __init__(self, node: Node, frame_id='map'):
        self.node = node
        self.frame_id = frame_id

        self.voxel_pub = node.create_publisher(
            MarkerArray, '/active_recon/voxel_grid', 1)
        self.bbox_pub = node.create_publisher(
            Marker, '/active_recon/bounding_box', 1)
        self.poses_pub = node.create_publisher(
            MarkerArray, '/active_recon/camera_poses', 1)
        self.status_pub = node.create_publisher(
            String, '/active_recon/status', 1)

    def publish_status(self, phase, coverage_pct, n_gaussians, n_keyframes):
        msg = String()
        msg.data = (f"Phase: {phase} | Coverage: {coverage_pct:.0%} | "
                    f"Gaussians: {n_gaussians} | KFs: {n_keyframes}")
        self.status_pub.publish(msg)

    def publish_bounding_box(self, center, size):
        """Publish wireframe bounding box around rock."""
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.ns = 'bbox'
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.02  # line width

        m.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)
        m.lifetime = Duration(sec=0, nanosec=0)  # persistent

        cx, cy, cz = center
        hs = size / 2.0

        # 8 corners of the box
        corners = []
        for dx in [-hs, hs]:
            for dy in [-hs, hs]:
                for dz in [-hs, hs]:
                    corners.append((cx + dx, cy + dy, cz + dz))

        # 12 edges
        edges = [
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        ]
        for i, j in edges:
            p1 = Point(x=float(corners[i][0]), y=float(corners[i][1]),
                       z=float(corners[i][2]))
            p2 = Point(x=float(corners[j][0]), y=float(corners[j][1]),
                       z=float(corners[j][2]))
            m.points.append(p1)
            m.points.append(p2)

        self.bbox_pub.publish(m)

    def publish_camera_pose(self, pose_id, position, yaw):
        """Add a camera frustum marker at the given NED position."""
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.ns = 'camera_poses'
        m.id = pose_id
        m.type = Marker.ARROW
        m.action = Marker.ADD

        m.pose.position.x = float(position[0])
        m.pose.position.y = float(position[1])
        m.pose.position.z = float(position[2])

        # Quaternion from yaw (rotation about NED Z axis)
        import math
        half_yaw = yaw / 2.0
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = float(math.sin(half_yaw))
        m.pose.orientation.w = float(math.cos(half_yaw))

        m.scale.x = 0.3  # arrow length
        m.scale.y = 0.05  # arrow width
        m.scale.z = 0.05

        m.color = ColorRGBA(r=0.0, g=0.7, b=1.0, a=0.9)
        m.lifetime = Duration(sec=0, nanosec=0)

        ma.markers.append(m)
        self.poses_pub.publish(ma)

    def publish_voxel_grid(self, voxel_grid):
        """Publish voxel grid as colored cubes."""
        ma = MarkerArray()

        # Delete all previous markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        ma.markers.append(delete_marker)

        marker_id = 0
        occupied_indices = np.argwhere(voxel_grid.occupied)

        # Subsample if too many voxels for RViz performance
        if len(occupied_indices) > 5000:
            idx = np.random.choice(
                len(occupied_indices), 5000, replace=False)
            occupied_indices = occupied_indices[idx]

        # Create a single CUBE_LIST marker for efficiency
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.ns = 'voxels'
        m.id = 0
        m.type = Marker.CUBE_LIST
        m.action = Marker.ADD
        m.scale.x = voxel_grid.resolution * 0.9
        m.scale.y = voxel_grid.resolution * 0.9
        m.scale.z = voxel_grid.resolution * 0.9
        m.lifetime = Duration(sec=0, nanosec=0)

        for ix, iy, iz in occupied_indices:
            # World position
            wx = voxel_grid.origin[0] + (ix + 0.5) * voxel_grid.resolution
            wy = voxel_grid.origin[1] + (iy + 0.5) * voxel_grid.resolution
            wz = voxel_grid.origin[2] + (iz + 0.5) * voxel_grid.resolution
            m.points.append(Point(x=float(wx), y=float(wy), z=float(wz)))

            nv = voxel_grid.n_views[ix, iy, iz]
            if nv >= 3:
                m.colors.append(ColorRGBA(r=0.0, g=0.8, b=0.2, a=0.6))
            elif nv >= 1:
                m.colors.append(ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.4))
            else:
                m.colors.append(ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.2))

        if len(m.points) > 0:
            ma.markers.append(m)

        self.voxel_pub.publish(ma)
