#!/usr/bin/env python3
"""3DGS live model using gsplat for incremental online training.

Manages Gaussian parameters (means, quats, scales, opacities, colors) on GPU.
Supports incremental point addition, training with sliding window of views,
densification/pruning, rendering, and checkpointing.

All coordinates in NED world frame. Camera convention: X=Right, Y=Down, Z=Forward.
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import gsplat


# ── SSIM (ported from Kerbl et al. 3DGS loss_utils.py) ──────────────

def _gaussian_window(window_size, sigma):
    gauss = torch.Tensor([
        math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
        for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window(window_size, channel, device):
    w1d = _gaussian_window(window_size, 1.5).unsqueeze(1)
    w2d = w1d.mm(w1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = w2d.expand(channel, 1, window_size, window_size).contiguous()
    return window.to(device)


def compute_ssim(img1, img2, window_size=11):
    """Differentiable SSIM for [1,C,H,W] tensors. Returns scalar mean SSIM."""
    channel = img1.size(1)
    window = _create_window(window_size, channel, img1.device)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# ── Exponential LR decay (ported from Plenoxels / 3DGS general_utils.py) ──

def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0,
                      lr_delay_mult=1.0, max_steps=1000000):
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    return helper

# Camera intrinsics (640x480, HFOV=1.204 rad)
# fx = fy = 640 / (2 * tan(1.204/2)) = 465.7412
FX = 465.7412
FY = 465.7412
CX = 320.0
CY = 240.0
W = 640
H = 480

# NBV candidate scoring resolution (scale=2, matches training res)
NBV_SCALE = 2
W_NBV = W // NBV_SCALE    # 320
H_NBV = H // NBV_SCALE    # 240
FX_NBV = FX / NBV_SCALE
FY_NBV = FY / NBV_SCALE
CX_NBV = CX / NBV_SCALE
CY_NBV = CY / NBV_SCALE

# Depth rendering for NBV error comparison (scale=4 from full res)
DEPTH_NBV_SCALE = 4
W_DEPTH_NBV = W // DEPTH_NBV_SCALE   # 320
H_DEPTH_NBV = H // DEPTH_NBV_SCALE   # 240
FX_DEPTH_NBV = FX / DEPTH_NBV_SCALE
FY_DEPTH_NBV = FY / DEPTH_NBV_SCALE
CX_DEPTH_NBV = CX / DEPTH_NBV_SCALE
CY_DEPTH_NBV = CY / DEPTH_NBV_SCALE


class GaussianModel3DGS:
    """Incremental 3D Gaussian Splatting model backed by gsplat."""

    def __init__(self, max_gaussians=200000, pts_per_frame=2000,
                 min_depth=0.3, max_depth=12.0, depth_loss_weight=0.5,
                 iso_loss_weight=0.1, bbox_center=None, bbox_size=None,
                 train_scale=4, device=None):
        self.max_gaussians = max_gaussians
        self.pts_per_frame = pts_per_frame
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_loss_w = depth_loss_weight
        self.iso_loss_w = iso_loss_weight
        self.train_scale = train_scale
        self.device = device or torch.device('cuda')

        # Bounding box for clipping Gaussians to rock region
        if bbox_center is not None and bbox_size is not None:
            hs = bbox_size / 2.0
            self.bbox_min = torch.tensor(
                [bbox_center[0] - hs, bbox_center[1] - hs,
                 bbox_center[2] - hs],
                dtype=torch.float32, device=self.device)
            self.bbox_max = torch.tensor(
                [bbox_center[0] + hs, bbox_center[1] + hs,
                 bbox_center[2] + hs],
                dtype=torch.float32, device=self.device)
        else:
            self.bbox_min = None
            self.bbox_max = None

        # Full-res intrinsics (for rendering/eval)
        self.K = torch.tensor([
            [FX, 0, CX], [0, FY, CY], [0, 0, 1],
        ], dtype=torch.float32, device=self.device)

        # Training-res intrinsics (downscaled for faster rasterization)
        s = train_scale
        self.W_train = W // s
        self.H_train = H // s
        self.K_train = torch.tensor([
            [FX / s, 0, CX / s], [0, FY / s, CY / s], [0, 0, 1],
        ], dtype=torch.float32, device=self.device)

        # Ultra-low-res intrinsics for NBV alpha rendering
        self.K_nbv = torch.tensor([
            [FX_NBV, 0.0, CX_NBV],
            [0.0, FY_NBV, CY_NBV],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32, device=self.device)

        # Depth-res intrinsics for NBV depth error comparison
        self.K_depth_nbv = torch.tensor([
            [FX_DEPTH_NBV, 0.0, CX_DEPTH_NBV],
            [0.0, FY_DEPTH_NBV, CY_DEPTH_NBV],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32, device=self.device)

        # Gaussian parameters (None until first points added)
        self.means = None
        self.scales = None
        self.quats = None
        self.opacities = None
        self.colors = None
        self.optimizer = None

        # Gradient accumulator for densification
        self._grad_accum = None
        self._grad_count = None

        # Training state
        self.n_gaussians = 0
        self.total_train_steps = 0
        self.last_loss = 0.0
        self._steps_since_opacity_reset = float('inf')  # safe to prune initially

        # Position LR schedule: exp decay from 0.00016 → 0.0000016 over 15K steps
        self.pos_lr_scheduler = get_expon_lr_func(
            lr_init=0.00016, lr_final=0.0000016, max_steps=15000)

        # Warmup gsplat CUDA kernels
        self._warmup()

    def _warmup(self):
        _m = torch.randn(4, 3, device=self.device, requires_grad=True)
        _q = F.normalize(torch.randn(4, 4, device=self.device), dim=-1)
        _s = torch.ones(4, 3, device=self.device) * 0.01
        _o = torch.ones(4, device=self.device) * 0.5
        _c = torch.rand(4, 3, device=self.device)
        _vm = torch.eye(4, device=self.device, dtype=torch.float32)[None]
        gsplat.rasterization(
            _m, _q, _s, _o, _c, _vm, self.K_train[None],
            self.W_train, self.H_train, render_mode='RGB+D', packed=True)
        del _m, _q, _s, _o, _c, _vm
        torch.cuda.empty_cache()

    @staticmethod
    def compute_viewmat(px, py, pz, yaw, gimbal_pitch):
        """Build 4x4 world-to-camera matrix from PX4 NED pose + gimbal pitch.

        World: NED (X=North, Y=East, Z=Down)
        Body:  FRD (X=Forward, Y=Right, Z=Down) at heading=yaw
        Camera: X=Right, Y=Down, Z=Forward (OpenCV/COLMAP convention)

        Gimbal pitch: negative = looking down from horizontal.
        """
        alpha = -gimbal_pitch  # alpha > 0 means looking down
        ca, sa = math.cos(alpha), math.sin(alpha)
        cy, sy = math.cos(yaw), math.sin(yaw)

        R_cam_body = np.array([
            [0,   1, 0],
            [-sa, 0, ca],
            [ca,  0, sa],
        ], dtype=np.float32)

        R_body_world = np.array([
            [cy,  sy, 0],
            [-sy, cy, 0],
            [0,   0,  1],
        ], dtype=np.float32)

        R = R_cam_body @ R_body_world
        p = np.array([px, py, pz], dtype=np.float32)
        t = -R @ p

        W2C = np.eye(4, dtype=np.float32)
        W2C[:3, :3] = R
        W2C[:3, 3] = t
        return W2C

    def backproject_depth(self, depth_np, viewmat_np, rgb_np):
        """Backproject depth image to world 3D points with colors.

        Returns (pts_world, colors) as float32 numpy arrays.
        """
        valid = (np.isfinite(depth_np)
                 & (depth_np > self.min_depth)
                 & (depth_np < self.max_depth))
        vs, us = np.where(valid)
        if len(vs) == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

        if len(vs) > self.pts_per_frame:
            idx = np.random.choice(len(vs), self.pts_per_frame, replace=False)
            vs, us = vs[idx], us[idx]

        ds = depth_np[vs, us]

        # Backproject using unified intrinsics (RGB and depth share resolution)
        x_cam = (us.astype(np.float32) - CX) / FX * ds
        y_cam = (vs.astype(np.float32) - CY) / FY * ds
        z_cam = ds
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

        R = viewmat_np[:3, :3]
        t = viewmat_np[:3, 3]
        pts_world = (pts_cam - t[None]) @ R  # R^T @ (p_cam - t)

        # Direct pixel correspondence (same resolution)
        colors = rgb_np[vs, us, ::-1].astype(np.float32) / 255.0

        return pts_world, colors

    def _inside_bbox(self, means):
        """Return boolean mask of Gaussians inside the bounding box."""
        if self.bbox_min is None:
            return torch.ones(len(means), dtype=torch.bool, device=self.device)
        return ((means >= self.bbox_min) & (means <= self.bbox_max)).all(dim=-1)

    def add_points(self, pts_world, colors_rgb):
        """Add new Gaussians from world-space points and RGB colors [0,1].

        Points outside the bounding box are discarded.
        """
        n_new = len(pts_world)
        if n_new == 0:
            return

        new_means = torch.from_numpy(pts_world).to(self.device)

        # Clip to bounding box
        inside = self._inside_bbox(new_means)
        if inside.sum() == 0:
            return
        if inside.sum() < len(new_means):
            new_means = new_means[inside]
            colors_rgb = colors_rgb[inside.cpu().numpy()]
            n_new = len(new_means)
        new_scales = torch.full((n_new, 3), -4.5, device=self.device)
        new_quats = torch.zeros(n_new, 4, device=self.device)
        new_quats[:, 0] = 1.0
        new_opacities = torch.full((n_new,), 0.0, device=self.device)
        new_colors = torch.logit(
            torch.from_numpy(colors_rgb).to(self.device).clamp(0.01, 0.99))

        if self.means is None:
            self.means = new_means
            self.scales = new_scales
            self.quats = new_quats
            self.opacities = new_opacities
            self.colors = new_colors
        else:
            self.means = torch.cat([self.means.detach(), new_means], dim=0)
            self.scales = torch.cat([self.scales.detach(), new_scales], dim=0)
            self.quats = torch.cat([self.quats.detach(), new_quats], dim=0)
            self.opacities = torch.cat(
                [self.opacities.detach(), new_opacities], dim=0)
            self.colors = torch.cat([self.colors.detach(), new_colors], dim=0)

        # Enforce budget — keep highest-opacity Gaussians (trained > untrained)
        if len(self.means) > self.max_gaussians:
            _, top_idx = torch.sigmoid(self.opacities).topk(
                self.max_gaussians)
            self.means = self.means[top_idx]
            self.scales = self.scales[top_idx]
            self.quats = self.quats[top_idx]
            self.opacities = self.opacities[top_idx]
            self.colors = self.colors[top_idx]

        self._rebuild_optimizer()
        self.n_gaussians = len(self.means)

    def _rebuild_optimizer(self):
        self.means = self.means.requires_grad_(True)
        self.scales = self.scales.requires_grad_(True)
        self.quats = self.quats.requires_grad_(True)
        self.opacities = self.opacities.requires_grad_(True)
        self.colors = self.colors.requires_grad_(True)

        self.optimizer = torch.optim.Adam([
            {'params': [self.means], 'lr': 0.00016},
            {'params': [self.scales], 'lr': 0.005},
            {'params': [self.quats], 'lr': 0.001},
            {'params': [self.opacities], 'lr': 0.025},
            {'params': [self.colors], 'lr': 0.0025},
        ])

        self._grad_accum = torch.zeros(
            len(self.means), device=self.device)
        self._grad_count = torch.zeros(
            len(self.means), device=self.device, dtype=torch.int32)

    def train_step(self, viewmat_np, rgb_gt_np, depth_gt_np):
        """One forward+backward pass. Returns loss float.

        Legacy single-view interface (accepts numpy). Use train_step_multi
        for batched multi-view training.
        """
        if self.means is None or len(self.means) == 0:
            return 0.0

        vm = torch.from_numpy(viewmat_np).to(self.device)
        rgb_gt = torch.from_numpy(rgb_gt_np).to(self.device)
        depth_gt = torch.from_numpy(depth_gt_np).to(self.device)

        return self._train_single_view(vm, rgb_gt, depth_gt)

    def _train_single_view(self, vm, rgb_gt, depth_gt):
        """Single-view forward+backward+step. All args are GPU tensors."""
        if self.optimizer is None:
            return 0.0
        self.optimizer.zero_grad()

        act_scales = torch.exp(self.scales)
        act_quats = F.normalize(self.quats, dim=-1)
        act_opac = torch.sigmoid(self.opacities)
        act_colors = torch.sigmoid(self.colors)

        renders, alphas, meta = gsplat.rasterization(
            means=self.means,
            quats=act_quats,
            scales=act_scales,
            opacities=act_opac,
            colors=act_colors,
            viewmats=vm[None],
            Ks=self.K_train[None],
            width=self.W_train,
            height=self.H_train,
            render_mode='RGB+D',
            packed=True,
            near_plane=self.min_depth,
            far_plane=self.max_depth,
        )

        rendered = renders[0]
        rgb_pred = rendered[:, :, :3]
        depth_pred = rendered[:, :, 3]

        l1_loss = F.l1_loss(rgb_pred, rgb_gt)
        rgb_pred_bchw = rgb_pred.permute(2, 0, 1).unsqueeze(0)
        rgb_gt_bchw = rgb_gt.permute(2, 0, 1).unsqueeze(0)
        ssim_val = compute_ssim(rgb_pred_bchw, rgb_gt_bchw)
        loss_color = 0.8 * l1_loss + 0.2 * (1.0 - ssim_val)

        depth_mask = (
            (depth_gt > self.min_depth)
            & (depth_gt < self.max_depth)
            & torch.isfinite(depth_gt))
        if depth_mask.any():
            loss_depth = F.l1_loss(
                depth_pred[depth_mask], depth_gt[depth_mask])
        else:
            loss_depth = torch.tensor(0.0, device=self.device)

        scale_mean = act_scales.mean(dim=-1, keepdim=True)
        loss_iso = (act_scales - scale_mean).abs().mean()

        loss = (loss_color
                + self.depth_loss_w * loss_depth
                + self.iso_loss_w * loss_iso)
        loss.backward()

        new_pos_lr = self.pos_lr_scheduler(self.total_train_steps)
        self.optimizer.param_groups[0]['lr'] = new_pos_lr

        self.optimizer.step()

        if self.means.grad is not None:
            grad_norm = self.means.grad.detach().norm(dim=-1)
            self._grad_accum += grad_norm
            self._grad_count += 1

        self.total_train_steps += 1
        self._steps_since_opacity_reset += 1

        if self.total_train_steps % 5000 == 0:
            self.reset_opacity()
        self.last_loss = loss.item()
        return self.last_loss

    def train_step_multi(self, views):
        """Multi-view training: accumulate loss across all views, backward once.

        Args:
            views: list of (rgb_gt, depth_gt, viewmat) GPU tensors.
        Returns:
            mean loss float across views.
        """
        if self.means is None or len(self.means) == 0:
            return 0.0

        self.optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device)
        n = len(views)

        for rgb_gt, depth_gt, vm in views:
            act_scales = torch.exp(self.scales)
            act_quats = F.normalize(self.quats, dim=-1)
            act_opac = torch.sigmoid(self.opacities)
            act_colors = torch.sigmoid(self.colors)

            renders, alphas, meta = gsplat.rasterization(
                means=self.means,
                quats=act_quats,
                scales=act_scales,
                opacities=act_opac,
                colors=act_colors,
                viewmats=vm[None],
                Ks=self.K_train[None],
                width=self.W_train,
                height=self.H_train,
                render_mode='RGB+D',
                packed=True,
                near_plane=self.min_depth,
                far_plane=self.max_depth,
            )

            rendered = renders[0]
            rgb_pred = rendered[:, :, :3]
            depth_pred = rendered[:, :, 3]

            l1_loss = F.l1_loss(rgb_pred, rgb_gt)
            rgb_pred_bchw = rgb_pred.permute(2, 0, 1).unsqueeze(0)
            rgb_gt_bchw = rgb_gt.permute(2, 0, 1).unsqueeze(0)
            ssim_val = compute_ssim(rgb_pred_bchw, rgb_gt_bchw)
            loss_color = 0.8 * l1_loss + 0.2 * (1.0 - ssim_val)

            depth_mask = (
                (depth_gt > self.min_depth)
                & (depth_gt < self.max_depth)
                & torch.isfinite(depth_gt))
            if depth_mask.any():
                loss_depth = F.l1_loss(
                    depth_pred[depth_mask], depth_gt[depth_mask])
            else:
                loss_depth = torch.tensor(0.0, device=self.device)

            # Isotropic regularization per view
            scale_mean = act_scales.mean(dim=-1, keepdim=True)
            loss_iso = self.iso_loss_w * (act_scales - scale_mean).abs().mean()

            view_loss = (loss_color
                         + self.depth_loss_w * loss_depth
                         + loss_iso) / n
            total_loss = total_loss + view_loss

        total_loss.backward()

        new_pos_lr = self.pos_lr_scheduler(self.total_train_steps)
        self.optimizer.param_groups[0]['lr'] = new_pos_lr

        self.optimizer.step()

        if self.means.grad is not None:
            grad_norm = self.means.grad.detach().norm(dim=-1)
            self._grad_accum += grad_norm
            self._grad_count += 1

        self.total_train_steps += 1
        self._steps_since_opacity_reset += 1

        if self.total_train_steps % 5000 == 0:
            self.reset_opacity()
        self.last_loss = total_loss.item()
        return self.last_loss

    def reset_opacity(self):
        """Reset all opacities to near-transparent (logit(0.01)).

        Only Gaussians needed by the loss will be driven back to high opacity.
        Dead/redundant ones stay transparent and get pruned. (Kerbl et al.)
        """
        with torch.no_grad():
            self.opacities.fill_(torch.logit(torch.tensor(0.01)).item())
        self._steps_since_opacity_reset = 0
        self._rebuild_optimizer()

    def render(self, viewmat_np):
        """Forward-only render for viewer. Returns RGB HxWx3 numpy uint8."""
        if self.means is None or len(self.means) == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)

        with torch.no_grad():
            vm = torch.from_numpy(viewmat_np).to(self.device)
            act_scales = torch.exp(self.scales)
            act_quats = F.normalize(self.quats, dim=-1)
            act_opac = torch.sigmoid(self.opacities)
            act_colors = torch.sigmoid(self.colors)

            renders, _, _ = gsplat.rasterization(
                means=self.means,
                quats=act_quats,
                scales=act_scales,
                opacities=act_opac,
                colors=act_colors,
                viewmats=vm[None],
                Ks=self.K[None],
                width=W,
                height=H,
                render_mode='RGB',
                packed=True,
                near_plane=self.min_depth,
                far_plane=self.max_depth,
            )

            rgb = renders[0].clamp(0, 1).cpu().numpy()
            return (rgb * 255).astype(np.uint8)

    def render_from_snapshot(self, viewmat_np, snapshot):
        """Render using a pre-taken snapshot (thread-safe for bg threads).

        Args:
            viewmat_np: 4x4 float32 numpy array
            snapshot: dict from get_snapshot() with activated tensors
        Returns:
            RGB HxWx3 numpy uint8
        """
        if snapshot is None or snapshot['n_gaussians'] == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)

        with torch.no_grad():
            vm = torch.from_numpy(viewmat_np).to(self.device)
            renders, _, _ = gsplat.rasterization(
                means=snapshot['means'],
                quats=snapshot['quats'],
                scales=snapshot['scales'],
                opacities=snapshot['opacities'],
                colors=snapshot['colors'],
                viewmats=vm[None],
                Ks=self.K[None],
                width=W,
                height=H,
                render_mode='RGB',
                packed=True,
                near_plane=self.min_depth,
                far_plane=self.max_depth,
            )
            rgb = renders[0].clamp(0, 1).cpu().numpy()
            return (rgb * 255).astype(np.uint8)

    def render_train_res(self, viewmat_np):
        """Forward-only render at training resolution. Returns RGB HxWx3 uint8."""
        if self.means is None or len(self.means) == 0:
            return np.zeros((self.H_train, self.W_train, 3), dtype=np.uint8)

        with torch.no_grad():
            vm = torch.from_numpy(viewmat_np).to(self.device)
            act_scales = torch.exp(self.scales)
            act_quats = F.normalize(self.quats, dim=-1)
            act_opac = torch.sigmoid(self.opacities)
            act_colors = torch.sigmoid(self.colors)

            renders, _, _ = gsplat.rasterization(
                means=self.means,
                quats=act_quats,
                scales=act_scales,
                opacities=act_opac,
                colors=act_colors,
                viewmats=vm[None],
                Ks=self.K_train[None],
                width=self.W_train,
                height=self.H_train,
                render_mode='RGB',
                packed=True,
                near_plane=self.min_depth,
                far_plane=self.max_depth,
            )

            rgb = renders[0].clamp(0, 1).cpu().numpy()
            return (rgb * 255).astype(np.uint8)

    def render_with_alpha(self, viewmat_np):
        """Render RGB + per-pixel alpha (accumulated opacity).

        Returns:
            rgb: HxWx3 numpy uint8
            alpha: HxW numpy float32 in [0, 1]  (1 = fully opaque, 0 = empty)
        """
        if self.means is None or len(self.means) == 0:
            return (np.zeros((H, W, 3), dtype=np.uint8),
                    np.zeros((H, W), dtype=np.float32))

        with torch.no_grad():
            vm = torch.from_numpy(viewmat_np).to(self.device)
            act_scales = torch.exp(self.scales)
            act_quats = F.normalize(self.quats, dim=-1)
            act_opac = torch.sigmoid(self.opacities)
            act_colors = torch.sigmoid(self.colors)

            renders, alphas, _ = gsplat.rasterization(
                means=self.means,
                quats=act_quats,
                scales=act_scales,
                opacities=act_opac,
                colors=act_colors,
                viewmats=vm[None],
                Ks=self.K[None],
                width=W,
                height=H,
                render_mode='RGB',
                packed=True,
                near_plane=self.min_depth,
                far_plane=self.max_depth,
            )

            rgb = renders[0].clamp(0, 1).cpu().numpy()
            alpha = alphas[0, :, :, 0].cpu().numpy()
            return (rgb * 255).astype(np.uint8), alpha

    def render_with_alpha_lowres(self, viewmat_np):
        """Render alpha at training resolution (fast, for NBV scoring)."""
        if self.means is None or len(self.means) == 0:
            return (np.zeros((self.H_train, self.W_train, 3), dtype=np.uint8),
                    np.zeros((self.H_train, self.W_train), dtype=np.float32))

        with torch.no_grad():
            vm = torch.from_numpy(viewmat_np).to(self.device)
            act_scales = torch.exp(self.scales)
            act_quats = F.normalize(self.quats, dim=-1)
            act_opac = torch.sigmoid(self.opacities)
            act_colors = torch.sigmoid(self.colors)

            renders, alphas, _ = gsplat.rasterization(
                means=self.means,
                quats=act_quats,
                scales=act_scales,
                opacities=act_opac,
                colors=act_colors,
                viewmats=vm[None],
                Ks=self.K_train[None],
                width=self.W_train,
                height=self.H_train,
                render_mode='RGB',
                packed=True,
                near_plane=self.min_depth,
                far_plane=self.max_depth,
            )

            rgb = renders[0].clamp(0, 1).cpu().numpy()
            alpha = alphas[0, :, :, 0].cpu().numpy()
            return (rgb * 255).astype(np.uint8), alpha

    def densify_and_prune(self, grad_threshold=0.0002, opacity_threshold=0.005,
                          scale_threshold=0.05):
        """Split/clone high-gradient Gaussians, prune low-opacity ones."""
        if self.means is None or self._grad_count is None:
            return

        with torch.no_grad():
            # Average gradients
            mask_counted = self._grad_count > 0
            avg_grad = torch.zeros_like(self._grad_accum)
            avg_grad[mask_counted] = (
                self._grad_accum[mask_counted]
                / self._grad_count[mask_counted].float())

            act_scales = torch.exp(self.scales)
            act_opac = torch.sigmoid(self.opacities)

            # High-gradient Gaussians that are large → split
            big_mask = act_scales.max(dim=-1).values > scale_threshold
            split_mask = (avg_grad > grad_threshold) & big_mask
            n_split = split_mask.sum().item()

            # High-gradient Gaussians that are small → clone
            small_mask = ~big_mask
            clone_mask = (avg_grad > grad_threshold) & small_mask
            n_clone = clone_mask.sum().item()

            new_tensors = []
            if n_split > 0:
                # Split: replace with two smaller Gaussians
                split_means = self.means[split_mask].detach()
                split_scales = self.scales[split_mask].detach() - math.log(1.6)
                noise = torch.randn_like(split_means) * torch.exp(
                    self.scales[split_mask].detach())
                new_tensors.append({
                    'means': torch.cat([split_means + noise,
                                        split_means - noise]),
                    'scales': split_scales.repeat(2, 1),
                    'quats': self.quats[split_mask].detach().repeat(2, 1),
                    'opacities': self.opacities[split_mask].detach().repeat(2),
                    'colors': self.colors[split_mask].detach().repeat(2, 1),
                })

            if n_clone > 0:
                new_tensors.append({
                    'means': self.means[clone_mask].detach().clone(),
                    'scales': self.scales[clone_mask].detach().clone(),
                    'quats': self.quats[clone_mask].detach().clone(),
                    'opacities': self.opacities[clone_mask].detach().clone(),
                    'colors': self.colors[clone_mask].detach().clone(),
                })

            # Prune split originals and out-of-bbox; only prune low-opacity
            # if enough steps have passed since last opacity reset (Kerbl resets
            # to 0.01 and prunes at 0.005 — need training time to separate them)
            outside_bbox = ~self._inside_bbox(self.means)
            if self._steps_since_opacity_reset >= 200:
                prune_mask = (act_opac < opacity_threshold) | outside_bbox
            else:
                prune_mask = outside_bbox
            keep_mask = ~(prune_mask | split_mask)

            # Never prune ALL Gaussians — keep at least the highest-opacity ones
            if keep_mask.sum() == 0 and len(self.means) > 0:
                n_keep = max(1, len(self.means) // 10)
                top_idx = act_opac.topk(n_keep).indices
                keep_mask[top_idx] = True

            n_pruned = (~keep_mask).sum().item()

            self.means = self.means[keep_mask].detach()
            self.scales = self.scales[keep_mask].detach()
            self.quats = self.quats[keep_mask].detach()
            self.opacities = self.opacities[keep_mask].detach()
            self.colors = self.colors[keep_mask].detach()

            # Add new Gaussians from splits/clones
            for nt in new_tensors:
                self.means = torch.cat([self.means, nt['means']])
                self.scales = torch.cat([self.scales, nt['scales']])
                self.quats = torch.cat([self.quats, nt['quats']])
                self.opacities = torch.cat([self.opacities, nt['opacities']])
                self.colors = torch.cat([self.colors, nt['colors']])

            # Enforce budget — keep highest-opacity Gaussians
            if len(self.means) > self.max_gaussians:
                _, top_idx = torch.sigmoid(self.opacities).topk(
                    self.max_gaussians)
                self.means = self.means[top_idx]
                self.scales = self.scales[top_idx]
                self.quats = self.quats[top_idx]
                self.opacities = self.opacities[top_idx]
                self.colors = self.colors[top_idx]

        self._rebuild_optimizer()
        self.n_gaussians = len(self.means)

    def get_snapshot(self):
        """Return detached copies of Gaussian params for viewer thread.

        Only includes Gaussians inside the bounding box.
        Returns dict with means, quats, scales, opacities, colors as tensors,
        or None if no Gaussians exist.
        """
        if self.means is None:
            return None
        with torch.no_grad():
            mask = self._inside_bbox(self.means)
            return {
                'means': self.means[mask].detach().clone(),
                'quats': F.normalize(self.quats[mask].detach(), dim=-1),
                'scales': torch.exp(self.scales[mask].detach()),
                'opacities': torch.sigmoid(self.opacities[mask].detach()),
                'colors': torch.sigmoid(self.colors[mask].detach()),
                'n_gaussians': int(mask.sum().item()),
                'loss': self.last_loss,
            }

    def prepare_nbv_params(self):
        """Snapshot and activate params once for multi-render NBV scoring.

        Returns dict of GPU tensors with activations already applied,
        including gradient info for uncertainty estimation.
        Or None if no Gaussians exist.

        Thread-safe: clones all params then truncates to consistent size,
        since the optimizer thread may resize params via densify/prune
        between individual clone() calls.
        """
        if self.means is None or len(self.means) == 0:
            return None
        with torch.no_grad():
            # Clone each param independently (training thread may resize
            # between clones, so sizes can differ by a few thousand)
            means = self.means.detach().clone()
            quats = self.quats.detach().clone()
            scales = self.scales.detach().clone()
            opacs = self.opacities.detach().clone()
            colors = self.colors.detach().clone()

            # Snapshot gradient accumulators for uncertainty scoring
            if (self._grad_accum is not None
                    and self._grad_count is not None):
                ga = self._grad_accum.detach().clone()
                gc = self._grad_count.detach().clone()
            else:
                ga = None
                gc = None

            # Truncate to minimum consistent size
            n = min(len(means), len(quats), len(scales),
                    len(opacs), len(colors))
            if n == 0:
                return None

            means = means[:n]
            quats = quats[:n]
            scales = scales[:n]
            opacs = opacs[:n]
            colors = colors[:n]

            # Filter to Gaussians inside rock bbox — critical for
            # depth variance to measure rock surface uncertainty,
            # not ground/sky noise.
            bbox_mask = self._inside_bbox(means)
            if bbox_mask.sum() == 0:
                return None
            means = means[bbox_mask]
            quats = quats[bbox_mask]
            scales = scales[bbox_mask]
            opacs = opacs[bbox_mask]
            colors = colors[bbox_mask]

            # Compute average gradient per Gaussian
            if ga is not None and gc is not None:
                n_g = min(n, len(ga), len(gc))
                avg_grad_full = torch.zeros(n, device=self.device)
                counted = gc[:n_g] > 0
                avg_grad_full[:n_g][counted] = (
                    ga[:n_g][counted] / gc[:n_g][counted].float())
                avg_grad = avg_grad_full[bbox_mask]
            else:
                avg_grad = torch.zeros(len(means), device=self.device)

            return {
                'means': means,
                'quats': F.normalize(quats, dim=-1),
                'scales': torch.exp(scales),
                'opacities': torch.sigmoid(opacs),
                'colors': torch.sigmoid(colors),
                'avg_grad': avg_grad,
            }

    def render_alpha_nbv_fast(self, viewmat_np, params):
        """Render alpha at NBV res (320x240) using pre-activated params.

        Args:
            viewmat_np: 4x4 world-to-camera matrix (numpy float32).
            params: dict from prepare_nbv_params().
        Returns:
            alpha: (H_NBV, W_NBV) float32 GPU tensor, or None if no params.
        """
        if params is None:
            return None
        with torch.no_grad():
            vm = torch.from_numpy(viewmat_np).to(self.device)
            _, alphas, _ = gsplat.rasterization(
                means=params['means'],
                quats=params['quats'],
                scales=params['scales'],
                opacities=params['opacities'],
                colors=params['colors'],
                viewmats=vm[None],
                Ks=self.K_nbv[None],
                width=W_NBV,
                height=H_NBV,
                render_mode='RGB',
                packed=True,
                near_plane=self.min_depth,
                far_plane=self.max_depth,
            )
            return alphas[0, :, :, 0]  # (H_NBV, W_NBV), keep on GPU

    def render_depth_variance_nbv(self, viewmat_np, params):
        """Render per-pixel depth variance and alpha at NBV resolution.

        Computes depth and depth² as 2-channel colors, renders via alpha
        blending, then derives Var(D) = E[D²] - E[D]² per pixel.

        Args:
            viewmat_np: 4x4 world-to-camera matrix (numpy float32).
            params: dict from prepare_nbv_params().
        Returns:
            depth_var: (H_NBV, W_NBV) float32 GPU tensor.
            alpha: (H_NBV, W_NBV) float32 GPU tensor.
            Returns (None, None) if params is None.
        """
        if params is None:
            return None, None
        with torch.no_grad():
            vm = torch.from_numpy(viewmat_np).float().to(self.device)

            # Per-Gaussian depth in camera frame
            R = vm[:3, :3]
            t = vm[:3, 3]
            means_cam = params['means'] @ R.T + t[None, :]  # (N, 3)
            cam_depths = means_cam[:, 2:3]                    # (N, 1)

            # 2-channel colors: [depth, depth²]
            colors_dvar = torch.cat([cam_depths, cam_depths ** 2], dim=1)

            renders, alphas, _ = gsplat.rasterization(
                means=params['means'],
                quats=params['quats'],
                scales=params['scales'],
                opacities=params['opacities'],
                colors=colors_dvar,
                viewmats=vm[None],
                Ks=self.K_nbv[None],
                width=W_NBV,
                height=H_NBV,
                render_mode='RGB',
                packed=True,
                near_plane=self.min_depth,
                far_plane=self.max_depth,
            )

            # renders: (1, H, W, 2), alphas: (1, H, W, 1)
            d_blend = renders[0, :, :, 0]
            d2_blend = renders[0, :, :, 1]
            alpha = alphas[0, :, :, 0]

            safe_alpha = alpha.clamp(min=1e-6)
            E_D = d_blend / safe_alpha
            E_D2 = d2_blend / safe_alpha
            depth_var = (E_D2 - E_D ** 2).clamp(min=0.0)

            return depth_var, alpha

    def render_depth_nbv(self, viewmat_np, params):
        """Render expected depth at 320x240 using pre-activated params.

        Args:
            viewmat_np: 4x4 world-to-camera matrix (numpy float32).
            params: dict from prepare_nbv_params().
        Returns:
            depth: (H_DEPTH_NBV, W_DEPTH_NBV) float32 GPU tensor, or None.
        """
        if params is None:
            return None
        with torch.no_grad():
            vm = torch.from_numpy(viewmat_np).to(self.device)
            renders, _, _ = gsplat.rasterization(
                means=params['means'],
                quats=params['quats'],
                scales=params['scales'],
                opacities=params['opacities'],
                colors=params['colors'],
                viewmats=vm[None],
                Ks=self.K_depth_nbv[None],
                width=W_DEPTH_NBV,
                height=H_DEPTH_NBV,
                render_mode='RGB+D',
                packed=True,
                near_plane=self.min_depth,
                far_plane=self.max_depth,
            )
            return renders[0, :, :, 3]  # depth channel, keep on GPU

    def save(self, path):
        """Save Gaussian parameters to disk."""
        os.makedirs(path, exist_ok=True)
        if self.means is None:
            return
        torch.save({
            'means': self.means.detach().cpu(),
            'scales': self.scales.detach().cpu(),
            'quats': self.quats.detach().cpu(),
            'opacities': self.opacities.detach().cpu(),
            'colors': self.colors.detach().cpu(),
            'n_gaussians': self.n_gaussians,
            'total_train_steps': self.total_train_steps,
        }, os.path.join(path, 'gaussians.pt'))

    def save_ply(self, filepath):
        """Export Gaussians in standard 3DGS PLY format (Kerbl et al.).

        Stores all parameters in raw (pre-activation) space:
        - Colors as SH DC coefficients (f_dc_0/1/2)
        - Opacity in logit space
        - Scales in log space
        Compatible with supersplat, antimatter15, and the reference viewer.
        """
        if self.means is None or len(self.means) == 0:
            return

        SH_C0 = 0.28209479177387814  # Y_0^0 normalization constant

        with torch.no_grad():
            pos = self.means.detach().cpu().numpy()
            normals = np.zeros_like(pos)
            # Convert sigmoid colors [0,1] -> SH DC coefficients
            rgb = torch.sigmoid(self.colors.detach()).cpu().numpy()
            f_dc = (rgb - 0.5) / SH_C0
            # Raw (pre-activation) opacity and scales
            opacities = self.opacities.detach().cpu().numpy()[:, None]
            scales = self.scales.detach().cpu().numpy()
            quats = F.normalize(self.quats.detach(), dim=-1).cpu().numpy()

        n = len(pos)
        # Build structured numpy array matching Kerbl reference format
        attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                 'f_dc_0', 'f_dc_1', 'f_dc_2',
                 'opacity',
                 'scale_0', 'scale_1', 'scale_2',
                 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        dtype = [(a, 'f4') for a in attrs]
        elements = np.empty(n, dtype=dtype)
        data = np.concatenate(
            [pos, normals, f_dc, opacities, scales, quats], axis=1)
        elements[:] = list(map(tuple, data))

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        # Write binary PLY without plyfile dependency
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
        )
        for name, fmt in dtype:
            header += f"property float {name}\n"
        header += "end_header\n"

        with open(filepath, 'wb') as f:
            f.write(header.encode('ascii'))
            f.write(elements.tobytes())

    def load(self, path):
        """Load Gaussian parameters from disk."""
        ckpt = torch.load(
            os.path.join(path, 'gaussians.pt'),
            map_location=self.device, weights_only=True)
        self.means = ckpt['means'].to(self.device)
        self.scales = ckpt['scales'].to(self.device)
        self.quats = ckpt['quats'].to(self.device)
        self.opacities = ckpt['opacities'].to(self.device)
        self.colors = ckpt['colors'].to(self.device)
        self.n_gaussians = len(self.means)
        self.total_train_steps = ckpt.get('total_train_steps', 0)
        self._rebuild_optimizer()
