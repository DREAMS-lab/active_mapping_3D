#!/usr/bin/env python3
"""Depth-variance geometric uncertainty + angular diversity NBV planner.

Scores candidate viewpoints using two complementary signals:

1. Area-weighted geometric uncertainty (60% weight):
   Render per-pixel depth variance from each candidate viewpoint via
   alpha-blended depth second moments: Var(D) = E[D²] - E[D]².
   High variance = Gaussians at inconsistent depths = uncertain geometry.
   Transparent pixels (alpha < threshold) are flagged as max uncertainty.
   Measured within the projected rock bounding box, multiplied by
   bbox_area_frac to favour side views where the rock fills the frame.
   Cannot be fooled by hallucination (unlike alpha-based uncoverage).

2. Angular diversity (40% weight):
   Minimum angle between candidate view direction (candidate → rock)
   and all existing view directions, normalised to [0, 1] by the
   candidate grid angular resolution (sqrt(2π/N)).  One grid cell
   away = 1.0; exact revisit = 0.0.  Promotes novel viewpoints.

Selection uses multiplicative batch diversity: initial scores are
multiplied by a proximity penalty to already-picked viewpoints in
this batch.  0° from pick → ×0, 90°+ → ×1.
"""

import json
import math
import os
import time

import numpy as np
import torch

from gaussian_model_nbv import (
    GaussianModel3DGS,
    FX_NBV, FY_NBV, CX_NBV, CY_NBV, W_NBV, H_NBV,
)

ALPHA_THRESH = 0.1
DEFAULT_MAX_VAR = 4.0  # bbox_size², max depth uncertainty


def _fibonacci_sphere(n):
    """Generate n approximately uniform points on a unit sphere.

    Uses the Fibonacci spiral method for near-uniform distribution.
    Returns (n, 3) array of unit vectors.
    """
    points = np.zeros((n, 3), dtype=np.float32)
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    for i in range(n):
        z = 1.0 - (2.0 * i + 1.0) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = 2.0 * math.pi * i / golden_ratio
        points[i] = [r * math.cos(theta), r * math.sin(theta), z]
    return points


class NBVPlanner:
    """Depth-variance + angular diversity acquisition function for NBV."""

    def __init__(self, rock_ned, radius=2.0, n_azimuth=12,
                 altitudes=(0.5, 1.0, 2.0, 3.5), min_altitude=0.3,
                 bbox_size=2.0, device=None):
        """
        Args:
            rock_ned:  [3] rock center in NED.
            radius:    orbit radius in meters.
            n_azimuth: azimuthal candidate count.
            altitudes: altitude offsets above rock (meters).
            min_altitude: safety floor (NED: z < -min_altitude).
            bbox_size: bounding box edge length.
            device: torch device.
        """
        self.rock_ned = np.array(rock_ned, dtype=np.float32)
        self.radius = radius
        self.n_azimuth = n_azimuth
        self.altitudes = altitudes
        self.min_altitude = min_altitude
        self.bbox_size = bbox_size
        self.device = device or torch.device('cuda')

        # Sphere radius: max rock dimension * 2 + small offset
        self.sphere_radius = bbox_size + 0.5

        # 8 corners of the rock bounding box in world (NED) coordinates
        self.bbox_corners = self._compute_bbox_corners()

        self.candidates = self._build_candidates()
        self.visited_positions = []
        self.existing_view_dirs = []   # unit vectors: visited → rock
        # Angular scale derived from candidate grid resolution:
        # on a hemisphere (area=2π) with N candidates, each cell has
        # angular radius ≈ sqrt(2π/N).  Used to normalize angular_dist
        # so that "one grid cell away" maps to 1.0.
        self._grid_scale = math.sqrt(
            2.0 * math.pi / max(len(self.candidates), 1))

        # Filled by score_candidates()
        self.last_scored = None
        self.last_analysis = None
        self.scoring_history = []

        # Rock center on GPU for view direction computation
        self.rock_ned_t = torch.tensor(
            self.rock_ned, dtype=torch.float32, device=self.device)

    # ── Bbox helpers ───────────────────────────────────────

    def _compute_bbox_corners(self):
        """Return (8, 3) float32 array of bounding box corners in NED."""
        cx, cy, cz = self.rock_ned
        hs = self.bbox_size / 2.0
        corners = np.array([
            [cx - hs, cy - hs, cz - hs],
            [cx - hs, cy - hs, cz + hs],
            [cx - hs, cy + hs, cz - hs],
            [cx - hs, cy + hs, cz + hs],
            [cx + hs, cy - hs, cz - hs],
            [cx + hs, cy - hs, cz + hs],
            [cx + hs, cy + hs, cz - hs],
            [cx + hs, cy + hs, cz + hs],
        ], dtype=np.float32)
        return corners

    def _project_bbox_rect(self, viewmat_np):
        """Project 8 bbox corners into image and return clipped rect.

        Returns:
            (u0, v0, u1, v1) pixel rect clipped to [0, W_NBV) x [0, H_NBV),
            or None if bbox is entirely behind camera or off-screen.
        """
        R = viewmat_np[:3, :3]
        t = viewmat_np[:3, 3]

        # Transform to camera space
        pts_cam = (R @ self.bbox_corners.T + t[:, None]).T  # (8, 3)

        # At least some corners must be in front of camera
        in_front = pts_cam[:, 2] > 0.1
        if not in_front.any():
            return None

        # Project only forward-facing corners
        pts_fwd = pts_cam[in_front]
        us = FX_NBV * pts_fwd[:, 0] / pts_fwd[:, 2] + CX_NBV
        vs = FY_NBV * pts_fwd[:, 1] / pts_fwd[:, 2] + CY_NBV

        u0 = max(0, int(np.floor(us.min())))
        v0 = max(0, int(np.floor(vs.min())))
        u1 = min(W_NBV, int(np.ceil(us.max())))
        v1 = min(H_NBV, int(np.ceil(vs.max())))

        if u1 <= u0 or v1 <= v0:
            return None

        return u0, v0, u1, v1

    # ── Candidate generation ──────────────────────────────

    def _build_candidates(self):
        """Generate candidate viewpoints on a sphere around the rock.

        Uses Fibonacci spiral for near-uniform coverage of the upper
        hemisphere (NED: z < rock_z, i.e. above the rock).
        """
        rx, ry, rz = self.rock_ned
        hs = self.bbox_size / 2.0
        bbox_margin = 0.5

        n_sphere = 200
        unit_pts = _fibonacci_sphere(n_sphere)

        candidates = []
        for pt in unit_pts:
            if pt[2] < 0.05:
                continue

            cx = rx + self.sphere_radius * pt[0]
            cy = ry + self.sphere_radius * pt[1]
            cz = rz - self.sphere_radius * pt[2]

            if cz > -self.min_altitude:
                continue

            dx = max(0.0, abs(cx - rx) - hs)
            dy = max(0.0, abs(cy - ry) - hs)
            dz = max(0.0, abs(cz - rz) - hs)
            dist_to_bbox = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist_to_bbox < bbox_margin:
                continue

            yaw = math.atan2(ry - cy, rx - cx)
            horiz = math.sqrt((rx - cx)**2 + (ry - cy)**2)
            vert = rz - cz
            gimbal_pitch = -abs(math.atan2(vert, horiz))

            elev_deg = math.degrees(math.atan2(
                self.sphere_radius * pt[2], horiz))
            azimuth_deg = math.degrees(math.atan2(pt[1], pt[0]))
            if azimuth_deg < 0:
                azimuth_deg += 360.0

            altitude = self.sphere_radius * pt[2]

            # Pre-compute view direction (candidate → rock), unit vector
            view_vec = self.rock_ned - np.array([cx, cy, cz], dtype=np.float32)
            view_dir = view_vec / (np.linalg.norm(view_vec) + 1e-8)

            candidates.append({
                'position': np.array([cx, cy, cz], dtype=np.float32),
                'yaw': yaw,
                'gimbal_pitch': gimbal_pitch,
                'azimuth_deg': azimuth_deg,
                'elevation_deg': elev_deg,
                'altitude': altitude,
                'view_dir': view_dir,
                'index': len(candidates),
            })

        return candidates

    @property
    def n_candidates(self):
        return len(self.candidates)

    # ── Acquisition function ──────────────────────────────

    def score_candidates(self, gs_model, phase_progress=0.5):
        """Score all candidates using depth-variance + angular diversity.

        Args:
            gs_model: GaussianModel3DGS with trained Gaussians
            phase_progress: kf_count / kf_budget [0..1]
        Returns:
            list of candidate dicts with scores, sorted descending.
        """
        t_start = time.perf_counter()

        params = gs_model.prepare_nbv_params()
        if params is None:
            return []

        C = len(self.candidates)

        # Pre-compute angular distances to existing views
        existing_dirs = np.array(self.existing_view_dirs, dtype=np.float32) \
            if self.existing_view_dirs else np.zeros((0, 3), dtype=np.float32)

        geo_uncertainty = np.zeros(C, dtype=np.float32)
        bbox_uncov = np.zeros(C, dtype=np.float32)  # kept for comparison
        angular_dist = np.zeros(C, dtype=np.float32)
        bbox_area_frac = np.zeros(C, dtype=np.float32)

        t_render_start = time.perf_counter()

        for i, cand in enumerate(self.candidates):
            pos = cand['position']
            viewmat = GaussianModel3DGS.compute_viewmat(
                pos[0], pos[1], pos[2], cand['yaw'],
                cand['gimbal_pitch'])

            # ── 1. Depth-variance geometric uncertainty ──
            rect = self._project_bbox_rect(viewmat)
            if rect is None:
                geo_uncertainty[i] = 0.0
                bbox_uncov[i] = 0.0
                bbox_area_frac[i] = 0.0
            else:
                u0, v0, u1, v1 = rect
                depth_var, alpha = gs_model.render_depth_variance_nbv(
                    viewmat, params)
                if depth_var is None:
                    geo_uncertainty[i] = DEFAULT_MAX_VAR
                    bbox_uncov[i] = 1.0
                else:
                    roi_dvar = depth_var[v0:v1, u0:u1]
                    roi_alpha = alpha[v0:v1, u0:u1]

                    # Comparison logging: old-style bbox uncoverage
                    bbox_uncov[i] = float((1.0 - roi_alpha).mean().item())

                    # Classify pixels: opaque vs transparent
                    opaque_mask = roi_alpha > ALPHA_THRESH
                    if opaque_mask.any():
                        opaque_var = roi_dvar[opaque_mask]
                        max_var = float(opaque_var.max().item())
                    else:
                        max_var = DEFAULT_MAX_VAR

                    # Transparent pixels get max_var (maximum uncertainty)
                    pixel_uncert = torch.where(
                        opaque_mask, roi_dvar,
                        torch.full_like(roi_dvar, max_var))
                    geo_uncertainty[i] = float(pixel_uncert.mean().item())

                bbox_area_frac[i] = (
                    (u1 - u0) * (v1 - v0)) / (W_NBV * H_NBV)

            # ── 2. Angular diversity ──
            view_dir = cand['view_dir']
            if len(existing_dirs) == 0:
                angular_dist[i] = 1.0  # max diversity when no views yet
            else:
                dots = existing_dirs @ view_dir
                dots = np.clip(dots, -1.0, 1.0)
                min_angle = np.arccos(dots).min()
                # Normalize by grid resolution so "one cell away" = 1.0.
                # With π normalization, angles 0.01-0.15 rad all mapped to
                # ~0, making angular contribution negligible after 20+ views.
                angular_dist[i] = min(min_angle / self._grid_scale, 1.0)

        t_render_end = time.perf_counter()

        # ── Combined score ──
        # Area-weight geo_uncertainty: multiply by bbox_area_frac so views
        # where the rock fills more of the frame (side views) score higher.
        W_ANGULAR = 0.4
        W_GEO = 0.6
        uncert_score = geo_uncertainty * bbox_area_frac
        # Robust p95 normalization
        p95 = np.percentile(uncert_score, 95)
        if p95 > 1e-6:
            uncert_score = np.clip(uncert_score / p95, 0.0, 1.0)
        final_scores = W_ANGULAR * angular_dist + W_GEO * uncert_score

        # Build results
        scored = []
        for i, c in enumerate(self.candidates):
            entry = dict(c)
            entry['score'] = float(final_scores[i])
            entry['geo_uncertainty'] = float(geo_uncertainty[i])
            entry['score_bbox_uncov'] = float(bbox_uncov[i])
            entry['score_angular'] = float(angular_dist[i])
            entry['bbox_area_frac'] = float(bbox_area_frac[i])
            scored.append(entry)

        scored.sort(key=lambda x: x['score'], reverse=True)
        for rank, entry in enumerate(scored):
            entry['rank'] = rank

        t_end = time.perf_counter()

        # Analysis summary
        analysis = {
            'n_gaussians_total': gs_model.n_gaussians,
            'n_candidates': C,
            'sphere_radius': float(self.sphere_radius),
            'phase_progress': float(phase_progress),
            # Depth-variance geometric uncertainty stats
            'geo_uncertainty_mean': float(geo_uncertainty.mean()),
            'geo_uncertainty_max': float(geo_uncertainty.max()),
            'geo_uncertainty_min': float(geo_uncertainty.min()),
            # Bbox-alpha stats (kept for comparison)
            'bbox_uncov_mean': float(bbox_uncov.mean()),
            'bbox_uncov_max': float(bbox_uncov.max()),
            'bbox_uncov_min': float(bbox_uncov.min()),
            'bbox_area_frac_mean': float(bbox_area_frac.mean()),
            # Angular diversity stats
            'angular_dist_mean': float(angular_dist.mean()),
            'angular_dist_min': float(angular_dist.min()),
            'angular_dist_max': float(angular_dist.max()),
            # Combined score stats
            'score_max': float(final_scores.max()),
            'score_min': float(final_scores.min()),
            'score_mean': float(final_scores.mean()),
            'score_std': float(final_scores.std()),
            'score_top5_mean': float(
                final_scores[np.argsort(-final_scores)[:5]].mean()),
            # Timing
            'render_time_ms': (t_render_end - t_render_start) * 1000,
            'compute_time_ms': (t_end - t_start) * 1000,
            'n_visited': len(self.visited_positions),
            'n_existing_views': len(self.existing_view_dirs),
            # Weights
            'w_angular': W_ANGULAR,
            'w_geo': W_GEO,
        }

        self.last_scored = scored
        self.last_analysis = analysis
        self.scoring_history.append({
            'n_visited': len(self.visited_positions),
            'analysis': dict(analysis),
            'top_10': [
                {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in s.items()}
                for s in scored[:10]
            ],
        })

        return scored

    def select_top_k(self, gs_model, k=24, start_pos=None,
                     phase_progress=0.5):
        """Score all candidates and greedily select top-k.

        Uses multiplicative batch diversity: each candidate's initial
        score (from score_candidates) is multiplied by a proximity
        penalty to already-picked viewpoints in this batch.  Candidates
        near a pick get score × ~0; candidates ≥90° away keep full
        score.  This naturally spreads viewpoints without hard-coded
        angular thresholds.

        Args:
            gs_model: live Gaussian model
            k: number of viewpoints to select
            start_pos: [3] starting position (unused, kept for API compat)
            phase_progress: kf_count / kf_budget [0..1]
        Returns:
            List of candidate dicts in flight order.
        """
        scored = self.score_candidates(
            gs_model, phase_progress=phase_progress)
        if not scored:
            return []

        C = len(scored)
        initial_scores = np.array(
            [s['score'] for s in scored], dtype=np.float32)
        cand_dirs = np.array(
            [s['view_dir'] for s in scored], dtype=np.float32)

        MIN_SCORE = 1e-3

        selected = []
        selected_set = set()

        for _ in range(min(k, C)):
            # Batch diversity: multiplicative penalty for proximity
            # to viewpoints already picked in THIS batch.
            # 0° from pick → 0, 45° → 0.5, 90°+ → 1.0.
            if selected_set:
                sel_dirs = np.array(
                    [cand_dirs[s] for s in selected_set], dtype=np.float32)
                dots = cand_dirs @ sel_dirs.T
                batch_min_ang = np.arccos(
                    np.clip(dots, -1.0, 1.0)).min(axis=1)
                batch_div = np.clip(
                    batch_min_ang / (math.pi / 2), 0.0, 1.0)
            else:
                batch_div = np.ones(C, dtype=np.float32)

            scores = initial_scores * batch_div

            for idx in selected_set:
                scores[idx] = -1.0

            best_idx = int(np.argmax(scores))
            if scores[best_idx] < MIN_SCORE:
                break

            selected.append(scored[best_idx])
            selected_set.add(best_idx)

        # Order by azimuth for efficient flight path
        selected.sort(key=lambda c: (c['azimuth_deg'], c['elevation_deg']))

        return selected

    def mark_visited(self, position):
        """Record a visited position and its view direction."""
        pos = np.array(position, dtype=np.float32)
        self.visited_positions.append(pos)
        # Compute and store view direction
        view_vec = self.rock_ned - pos
        view_dir = view_vec / (np.linalg.norm(view_vec) + 1e-8)
        self.existing_view_dirs.append(view_dir)

    # ── Persistence ───────────────────────────────────────

    def save_analysis(self, path, suffix=''):
        """Save all scoring data to disk."""
        os.makedirs(path, exist_ok=True)

        if self.last_scored is not None:
            scores_json = []
            for s in self.last_scored:
                entry = {}
                for k, v in s.items():
                    entry[k] = v.tolist() if isinstance(v, np.ndarray) else v
                scores_json.append(entry)
            with open(os.path.join(
                    path, f'candidate_scores{suffix}.json'), 'w') as f:
                json.dump(scores_json, f, indent=2)

        if self.last_analysis is not None:
            with open(os.path.join(
                    path, f'nbv_analysis{suffix}.json'), 'w') as f:
                json.dump(self.last_analysis, f, indent=2)

        if self.scoring_history:
            with open(os.path.join(
                    path, f'scoring_history{suffix}.json'), 'w') as f:
                json.dump(self.scoring_history, f, indent=2,
                          default=lambda o: o.tolist()
                          if isinstance(o, np.ndarray) else o)

    def get_starting_viewpoint(self):
        """Return starting viewpoint (mid altitude, azimuth=0)."""
        mid_alt = sorted(set(
            c['altitude'] for c in self.candidates))[
                len(set(c['altitude'] for c in self.candidates)) // 2]
        best = None
        best_dist = float('inf')
        for c in self.candidates:
            d = abs(c['altitude'] - mid_alt) + abs(c['azimuth_deg'])
            if d < best_dist:
                best_dist = d
                best = c
        return best
