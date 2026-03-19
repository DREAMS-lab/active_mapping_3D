#!/usr/bin/env python3
"""Convert transforms.json (PX4 known poses) to COLMAP text format.

Generates sparse/0/{cameras.txt, images.txt, points3D.txt} so that
the gaussian-splatting train.py can load the scene directly without
running COLMAP SfM.

Usage:
    python poses_to_colmap.py <run_dir>
"""

import json
import math
import os
import sys

import numpy as np


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return w, x, y, z


def compute_viewmat(px, py, pz, yaw, gimbal_pitch):
    """Replicate GaussianModel3DGS.compute_viewmat."""
    alpha = -gimbal_pitch
    ca, sa = math.cos(alpha), math.sin(alpha)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R_cam_body = np.array([
        [0,   1, 0],
        [-sa, 0, ca],
        [ca,  0, sa],
    ], dtype=np.float64)

    R_body_world = np.array([
        [cy,  sy, 0],
        [-sy, cy, 0],
        [0,   0,  1],
    ], dtype=np.float64)

    R = R_cam_body @ R_body_world
    p = np.array([px, py, pz], dtype=np.float64)
    t = -R @ p
    return R, t


def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <run_dir>')
        sys.exit(1)

    run_dir = sys.argv[1]
    transforms_path = os.path.join(run_dir, 'transforms.json')

    if not os.path.exists(transforms_path):
        print(f'Error: {transforms_path} not found')
        sys.exit(1)

    with open(transforms_path) as f:
        meta = json.load(f)

    fx = meta['fl_x']
    fy = meta['fl_y']
    cx = meta['cx']
    cy = meta['cy']
    w = meta['w']
    h = meta['h']
    frames = meta['frames']

    # Create output directory
    sparse_dir = os.path.join(run_dir, 'sparse', '0')
    os.makedirs(sparse_dir, exist_ok=True)

    # cameras.txt
    with open(os.path.join(sparse_dir, 'cameras.txt'), 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write(f'1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n')

    # images.txt
    with open(os.path.join(sparse_dir, 'images.txt'), 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, '
                'CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')

        for i, frame in enumerate(frames):
            pos = frame['position_ned']
            yaw = frame['heading']
            gpitch = frame['gimbal_pitch']

            R, t = compute_viewmat(pos[0], pos[1], pos[2], yaw, gpitch)
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R)

            # Image filename (strip path prefix)
            img_name = os.path.basename(frame['file_path'])

            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            f.write(f'{i+1} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} '
                    f'{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} 1 {img_name}\n')
            # Empty POINTS2D line
            f.write('\n')

    # points3D.txt — empty (train.py handles random init if no SfM points,
    # or we can provide depth-backprojected points)
    # For now, use depth backprojection to seed initial points
    depth_dir = os.path.join(run_dir, 'depth')
    img_dir = os.path.join(run_dir, 'images')

    points = []
    colors = []

    # Depth intrinsics (same as RGB: 640x480)
    fx_d, fy_d, cx_d, cy_d = 465.7412, 465.7412, 320.0, 240.0

    try:
        import cv2
        for i, frame in enumerate(frames):
            depth_path = os.path.join(
                run_dir, frame.get('depth_path', f'depth/{i:05d}.png'))
            rgb_path = os.path.join(run_dir, frame['file_path'])

            if not os.path.exists(depth_path):
                continue

            depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_mm is None:
                continue
            depth_m = depth_mm.astype(np.float64) / 1000.0

            rgb = cv2.imread(rgb_path)
            if rgb is None:
                continue

            pos = frame['position_ned']
            yaw = frame['heading']
            gpitch = frame['gimbal_pitch']
            R, t = compute_viewmat(pos[0], pos[1], pos[2], yaw, gpitch)

            valid = (depth_m > 0.3) & (depth_m < 12.0)
            vs, us = np.where(valid)
            if len(vs) == 0:
                continue

            # Subsample
            if len(vs) > 2000:
                idx = np.random.choice(len(vs), 2000, replace=False)
                vs, us = vs[idx], us[idx]

            ds = depth_m[vs, us]
            # Depth camera coords
            x_cam = (us.astype(np.float64) - cx_d) / fx_d * ds
            y_cam = (vs.astype(np.float64) - cy_d) / fy_d * ds
            z_cam = ds
            pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

            # Camera to world
            pts_world = (pts_cam - t[None]) @ R

            # Colors from RGB (map depth coords to RGB coords)
            h_rgb, w_rgb = rgb.shape[:2]
            h_d, w_d = depth_m.shape[:2]
            rgb_us = (us.astype(np.float64) * w_rgb / w_d).astype(
                int).clip(0, w_rgb - 1)
            rgb_vs = (vs.astype(np.float64) * h_rgb / h_d).astype(
                int).clip(0, h_rgb - 1)
            cols = rgb[rgb_vs, rgb_us, ::-1]  # BGR to RGB

            points.append(pts_world)
            colors.append(cols)

    except ImportError:
        print('Warning: cv2 not available, writing empty points3D.txt')

    with open(os.path.join(sparse_dir, 'points3D.txt'), 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, '
                'TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')

        if points:
            all_pts = np.concatenate(points)
            all_cols = np.concatenate(colors)
            for j in range(len(all_pts)):
                x, y, z = all_pts[j]
                r, g, b = int(all_cols[j, 0]), int(all_cols[j, 1]), \
                    int(all_cols[j, 2])
                f.write(f'{j+1} {x:.6f} {y:.6f} {z:.6f} '
                        f'{r} {g} {b} 0.0\n')

    # ── Inverse-depth images for depth supervision ──────────
    # gaussian-splatting train.py expects:
    #   depth_inv/<image_name>.png  (uint16, value = (1/depth_m) * 65536)
    #   sparse/0/depth_params.json  (per-image scale/offset, identity for GT)
    inv_dir = os.path.join(run_dir, 'depth_inv')
    os.makedirs(inv_dir, exist_ok=True)
    n_depth = 0

    try:
        if 'cv2' not in dir():
            import cv2

        depth_params = {}
        for i, frame in enumerate(frames):
            depth_path = os.path.join(
                run_dir, frame.get('depth_path', f'depth/{i:05d}.png'))
            if not os.path.exists(depth_path):
                continue

            depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_mm is None:
                continue
            depth_m = depth_mm.astype(np.float64) / 1000.0

            # Inverse depth: uint16 = (1/depth_m) * 65536
            valid = depth_m > 0.01
            inv_depth = np.zeros_like(depth_m)
            inv_depth[valid] = 1.0 / depth_m[valid]
            inv_uint16 = np.clip(inv_depth * 65536.0, 0, 65535).astype(
                np.uint16)

            # Output name matches image basename (without extension)
            img_name = os.path.basename(frame['file_path'])
            inv_name = os.path.splitext(img_name)[0] + '.png'
            cv2.imwrite(os.path.join(inv_dir, inv_name), inv_uint16)

            # Identity transform — our depth is ground truth
            depth_params[f'{i:05d}'] = {'scale': 1.0, 'offset': 0.0}
            n_depth += 1

        # Write depth_params.json
        params_path = os.path.join(sparse_dir, 'depth_params.json')
        with open(params_path, 'w') as f:
            json.dump(depth_params, f, indent=2)

    except ImportError:
        print('Warning: cv2 not available, skipping inverse depth generation')

    n_images = len(frames)
    n_points = len(np.concatenate(points)) if points else 0
    print(f'=== Poses to COLMAP ===')
    print(f'  Images:  {n_images}')
    print(f'  Points:  {n_points}')
    print(f'  Depth:   {n_depth} inverse-depth images in depth_inv/')
    print(f'  Output:  {sparse_dir}/')
    print(f'  Camera:  PINHOLE {w}x{h} fx={fx} fy={fy}')


if __name__ == '__main__':
    main()
