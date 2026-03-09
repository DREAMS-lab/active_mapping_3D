#!/usr/bin/env python3
"""OpenCV window with mouse orbit controls and GPU Gaussian splat rendering.

Renders actual Gaussian splats using gsplat.rasterization() from a virtual
camera that the user can orbit around the rock with mouse drag/scroll.

Thread-safe: reads Gaussian parameter snapshots pushed from the optimizer thread.
"""

import math
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import gsplat

from gaussian_model import FX, FY, CX, CY, W, H


class SplatViewer:
    """Live Gaussian splat viewer with mouse orbit controls."""

    def __init__(self, rock_center, bbox_size=2.0, device=None):
        """
        Args:
            rock_center: (x, y, z) NED world coordinates of rock center.
            bbox_size: bounding box edge length in meters.
            device: torch device for rendering.
        """
        self.rock_center = np.array(rock_center, dtype=np.float32)
        self.bbox_size = bbox_size
        self.device = device or torch.device('cuda')

        # Virtual camera orbit parameters
        self.azimuth = 0.0       # radians, 0 = looking from North
        self.elevation = 0.4     # radians above horizon (positive = looking down)
        self.distance = 4.0      # meters from rock center

        # Mouse state
        self._dragging = False
        self._last_mx = 0
        self._last_my = 0

        # Current Gaussian snapshot (thread-safe via lock)
        self._lock = threading.Lock()
        self._snapshot = None
        self._kf_count = 0
        self._n_gaussians = 0
        self._loss = 0.0

        # Intrinsics on GPU
        self.K = torch.tensor([
            [FX, 0, CX], [0, FY, CY], [0, 0, 1],
        ], dtype=torch.float32, device=self.device)

        self._running = False

    def update_snapshot(self, snapshot, kf_count):
        """Thread-safe update of Gaussian params from optimizer."""
        with self._lock:
            self._snapshot = snapshot
            self._kf_count = kf_count
            if snapshot is not None:
                self._n_gaussians = snapshot['n_gaussians']
                self._loss = snapshot['loss']

    def _build_viewmat(self):
        """Build world-to-camera matrix for virtual orbit camera."""
        # Camera position in NED world frame
        # azimuth=0 → camera at +X (North), looking towards center
        cx = self.rock_center[0] + self.distance * math.cos(
            self.elevation) * math.cos(self.azimuth)
        cy = self.rock_center[1] + self.distance * math.cos(
            self.elevation) * math.sin(self.azimuth)
        cz = self.rock_center[2] - self.distance * math.sin(
            self.elevation)  # NED: up is negative Z

        cam_pos = np.array([cx, cy, cz], dtype=np.float32)
        target = self.rock_center

        # Look-at matrix (camera convention: Z=forward, X=right, Y=down)
        forward = target - cam_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        # World up in NED is (0, 0, -1) = "up"
        world_up = np.array([0, 0, -1], dtype=np.float32)
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            # Degenerate case: looking straight down
            right = np.array([0, 1, 0], dtype=np.float32)
        else:
            right = right / right_norm

        down = np.cross(forward, right)

        # Camera axes: X=right, Y=down, Z=forward
        R = np.stack([right, down, forward], axis=0)  # 3x3
        t = -R @ cam_pos

        W2C = np.eye(4, dtype=np.float32)
        W2C[:3, :3] = R
        W2C[:3, 3] = t
        return W2C

    def _project_point(self, pt_world, viewmat):
        """Project a 3D world point to 2D pixel using viewmat and K."""
        p = viewmat[:3, :3] @ pt_world + viewmat[:3, 3]
        if p[2] <= 0.01:
            return None  # behind camera
        u = FX * p[0] / p[2] + CX
        v = FY * p[1] / p[2] + CY
        return int(u), int(v)

    def _draw_bbox(self, img, viewmat):
        """Draw wireframe bounding box around rock on the image."""
        hs = self.bbox_size / 2.0
        cx, cy, cz = self.rock_center
        corners_3d = []
        for dx in [-hs, hs]:
            for dy in [-hs, hs]:
                for dz in [-hs, hs]:
                    corners_3d.append(
                        np.array([cx + dx, cy + dy, cz + dz],
                                 dtype=np.float32))

        corners_2d = [self._project_point(c, viewmat) for c in corners_3d]

        edges = [
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        ]
        color = (0, 255, 255)  # yellow in BGR
        for i, j in edges:
            p1, p2 = corners_2d[i], corners_2d[j]
            if p1 is not None and p2 is not None:
                cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)

    def _render_frame(self):
        """Render current Gaussian snapshot from virtual camera."""
        with self._lock:
            snap = self._snapshot

        if snap is None:
            img = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.putText(img, "Waiting for data...", (W // 2 - 120, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            return img

        viewmat = self._build_viewmat()
        vm = torch.from_numpy(viewmat).to(self.device)

        with torch.no_grad():
            renders, _, _ = gsplat.rasterization(
                means=snap['means'],
                quats=snap['quats'],
                scales=snap['scales'],
                opacities=snap['opacities'],
                colors=snap['colors'],
                viewmats=vm[None],
                Ks=self.K[None],
                width=W,
                height=H,
                render_mode='RGB',
                packed=True,
                near_plane=0.3,
                far_plane=12.0,
            )
            rgb = renders[0].clamp(0, 1).cpu().numpy()

        img = (rgb * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw bounding box wireframe
        self._draw_bbox(img, viewmat)

        return img

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._last_mx = x
            self._last_my = y
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            dx = x - self._last_mx
            dy = y - self._last_my
            self.azimuth += dx * 0.005
            self.elevation = max(-0.2, min(1.4, self.elevation + dy * 0.005))
            self._last_mx = x
            self._last_my = y
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.distance = max(1.0, self.distance - 0.3)
            else:
                self.distance = min(15.0, self.distance + 0.3)

    def run(self):
        """Main viewer loop — runs on its own thread. Call stop() to exit."""
        self._running = True
        window_name = "Splat Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        while self._running:
            img = self._render_frame()

            # Overlay stats
            with self._lock:
                kf = self._kf_count
                ng = self._n_gaussians
                loss = self._loss
            info = f"KFs: {kf} | Gaussians: {ng // 1000}K | Loss: {loss:.4f}"
            cv2.putText(img, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            controls = "Drag: orbit | Scroll: zoom"
            cv2.putText(img, controls, (10, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            cv2.imshow(window_name, img)
            key = cv2.waitKey(50) & 0xFF  # ~20 FPS
            if key == 27:  # ESC
                break

        cv2.destroyAllWindows()

    def stop(self):
        self._running = False
