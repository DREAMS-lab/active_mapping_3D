#!/usr/bin/env python3
"""Voxel grid for tracking reconstruction coverage.

Centered on rock bounding box. Tracks which voxels have been observed
and from how many distinct keyframes. Provides coverage percentage
and RViz2 MarkerArray visualization.
"""

import numpy as np

from gaussian_model import FX, FY, CX, CY, W, H


class VoxelGrid:
    """3D voxel grid for depth-based coverage tracking."""

    def __init__(self, center, bbox_size=2.0, resolution=0.05):
        """
        Args:
            center: (x, y, z) NED world coordinates of bounding box center.
            bbox_size: extent of bounding box in meters (cube).
            resolution: voxel size in meters.
        """
        self.center = np.array(center, dtype=np.float32)
        self.bbox_size = bbox_size
        self.resolution = resolution

        self.half = bbox_size / 2.0
        self.origin = self.center - self.half  # min corner

        self.n = int(np.ceil(bbox_size / resolution))
        self.shape = (self.n, self.n, self.n)

        # Per-voxel state
        self.occupied = np.zeros(self.shape, dtype=bool)
        self.n_views = np.zeros(self.shape, dtype=np.uint8)

    def update_from_depth(self, depth_np, viewmat_np):
        """Backproject depth and update occupancy + view counts.

        Args:
            depth_np: HxW float32 depth image (meters).
            viewmat_np: 4x4 world-to-camera matrix.
        """
        valid = (np.isfinite(depth_np)
                 & (depth_np > 0.3)
                 & (depth_np < 12.0))
        vs, us = np.where(valid)
        if len(vs) == 0:
            return

        # Subsample for speed
        if len(vs) > 5000:
            idx = np.random.choice(len(vs), 5000, replace=False)
            vs, us = vs[idx], us[idx]

        ds = depth_np[vs, us]

        # Camera-frame 3D points
        x_cam = (us.astype(np.float32) - CX) / FX * ds
        y_cam = (vs.astype(np.float32) - CY) / FY * ds
        z_cam = ds
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

        # Camera-to-world
        R = viewmat_np[:3, :3]
        t = viewmat_np[:3, 3]
        pts_world = (pts_cam - t[None]) @ R

        # Convert to voxel indices
        voxel_coords = ((pts_world - self.origin[None]) / self.resolution
                        ).astype(np.int32)

        # Filter to valid indices
        valid_mask = np.all(
            (voxel_coords >= 0) & (voxel_coords < self.n), axis=1)
        voxel_coords = voxel_coords[valid_mask]

        if len(voxel_coords) == 0:
            return

        # Unique voxels hit by this frame
        unique_voxels = np.unique(voxel_coords, axis=0)
        ix, iy, iz = unique_voxels[:, 0], unique_voxels[:, 1], unique_voxels[:, 2]

        self.occupied[ix, iy, iz] = True
        # Saturate at 255
        current = self.n_views[ix, iy, iz].astype(np.uint16)
        self.n_views[ix, iy, iz] = np.minimum(current + 1, 255).astype(
            np.uint8)

    def get_coverage_pct(self, min_views=2):
        """Fraction of occupied voxels seen from >= min_views viewpoints."""
        n_occupied = self.occupied.sum()
        if n_occupied == 0:
            return 0.0
        n_covered = ((self.n_views >= min_views) & self.occupied).sum()
        return float(n_covered) / float(n_occupied)

    def get_stats(self):
        """Return dict with coverage statistics."""
        n_occupied = int(self.occupied.sum())
        return {
            'n_occupied': n_occupied,
            'n_covered_2plus': int(
                ((self.n_views >= 2) & self.occupied).sum()),
            'n_covered_3plus': int(
                ((self.n_views >= 3) & self.occupied).sum()),
            'coverage_pct': self.get_coverage_pct(),
            'total_voxels': int(np.prod(self.shape)),
        }

    def save(self, path):
        """Save voxel grid to npz."""
        np.savez(path,
                 occupied=self.occupied,
                 n_views=self.n_views,
                 center=self.center,
                 bbox_size=self.bbox_size,
                 resolution=self.resolution)
