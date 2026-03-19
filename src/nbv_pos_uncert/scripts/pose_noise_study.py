#!/usr/bin/env python3
"""Pose noise degradation study for 3DGS.

Perturbs COLMAP camera translations with per-keyframe Gaussian noise
matching the real PX4 EKF variance pattern from aquatic missions.
Scales from 0x to 20x the real sigma (~0.107m) to find the crossover
where positional uncertainty starts degrading reconstruction.

Uses existing captured data — no simulation needed.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


# Real PX4 EKF sigma from aquatic-mapping pose_aware trials
# Per-sample variance is ~0.01143 m^2, sigma ~0.107 m
# Very stable: range [0.1061, 0.1086] across 500 samples
REAL_EKF_SIGMA = 0.107  # meters


def parse_colmap_images(path):
    images = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 10 and parts[0].isdigit():
                images.append({
                    'id': int(parts[0]),
                    'qw': float(parts[1]), 'qx': float(parts[2]),
                    'qy': float(parts[3]), 'qz': float(parts[4]),
                    'tx': float(parts[5]), 'ty': float(parts[6]),
                    'tz': float(parts[7]),
                    'cam_id': int(parts[8]), 'name': parts[9],
                })
    return images


def write_colmap_images(images, path):
    with open(path, 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        for img in images:
            f.write(f"{img['id']} {img['qw']:.10f} {img['qx']:.10f} "
                    f"{img['qy']:.10f} {img['qz']:.10f} "
                    f"{img['tx']:.10f} {img['ty']:.10f} {img['tz']:.10f} "
                    f"{img['cam_id']} {img['name']}\n")
            f.write('\n')


def perturb_per_keyframe(images, scale, rng):
    """Per-keyframe noise matching real EKF pattern.

    Each keyframe gets its own sigma drawn from the real distribution:
    sigma_i ~ U(0.1061, 0.1086) * scale
    Then translation noise ~ N(0, sigma_i^2) per axis.
    """
    perturbed = []
    for img in images:
        # Real EKF variance varies slightly per sample
        sigma_i = rng.uniform(0.1061, 0.1086) * scale
        noise = rng.normal(0, sigma_i, size=3)
        perturbed.append({
            **img,
            'tx': img['tx'] + noise[0],
            'ty': img['ty'] + noise[1],
            'tz': img['tz'] + noise[2],
        })
    return perturbed


def run_3dgs(source_dir, model_dir, iterations, test_at):
    """Run Kerbl's train.py and parse PSNR from stdout."""
    # Derive workspace root from script install path
    ws_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    train_script = os.path.join(
        ws_root, 'repos', 'gaussian-splatting', 'train.py')

    cmd = [
        sys.executable, train_script,
        '-s', str(source_dir),
        '-m', str(model_dir),
        '--iterations', str(iterations),
        '--test_iterations', *[str(t) for t in test_at],
        '-r', '1',
        '--quiet',
        '--disable_viewer',
    ]

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=900,
        cwd=os.path.join(ws_root, 'repos', 'gaussian-splatting'))

    # Parse: [ITER N] Evaluating train: L1 X.XXXX PSNR XX.XXXX
    # train.py mixes stdout/stderr (tqdm), so we merge them above
    metrics = {}
    for line in result.stdout.split('\n'):
        m = re.search(
            r'\[ITER\s+(\d+)\]\s+Evaluating\s+(\w+):\s+L1\s+([\d.]+)\s+PSNR\s+([\d.]+)',
            line)
        if m:
            it, split = int(m.group(1)), m.group(2)
            metrics[f'{split}_psnr_{it}'] = float(m.group(4))
            metrics[f'{split}_l1_{it}'] = float(m.group(3))

    return metrics


def main():
    # Default run directory — override via command line if needed
    ws_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    run_dir = Path(os.path.join(ws_root, 'data', 'nbv', 'run_072'))
    sparse_dir = run_dir / 'sparse' / '0'

    original_images = parse_colmap_images(sparse_dir / 'images.txt')
    print(f'Loaded {len(original_images)} poses')

    output_base = run_dir / 'pose_noise_study'
    output_base.mkdir(exist_ok=True)

    # Scale factors: 0 (baseline), 0.5x, 1x, 2x, 5x, 10x, 20x real EKF sigma
    scales = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    n_trials = 3
    iterations = 7000
    test_at = [7000]

    rng = np.random.default_rng(42)
    results = []

    total = len(scales) * n_trials
    done = 0

    for scale in scales:
        sigma = REAL_EKF_SIGMA * scale
        print(f'\n{"="*50}')
        print(f'Scale {scale}x | sigma = {sigma:.4f} m | {sigma*100:.1f} cm')
        print(f'{"="*50}')

        for trial in range(n_trials):
            done += 1
            print(f'\n  Trial {trial+1}/{n_trials} [{done}/{total}]')

            trial_dir = output_base / f'scale_{scale:.1f}x' / f'trial_{trial:02d}'
            trial_sparse = trial_dir / 'sparse' / '0'
            trial_sparse.mkdir(parents=True, exist_ok=True)

            # Perturb
            if scale > 0:
                perturbed = perturb_per_keyframe(original_images, scale, rng)
            else:
                perturbed = original_images

            write_colmap_images(perturbed, trial_sparse / 'images.txt')
            shutil.copy2(sparse_dir / 'cameras.txt', trial_sparse / 'cameras.txt')
            shutil.copy2(sparse_dir / 'points3D.txt', trial_sparse / 'points3D.txt')

            # Symlink images
            img_link = trial_dir / 'images'
            if not img_link.exists():
                img_link.symlink_to(run_dir / 'images')

            # Train
            model_dir = trial_dir / 'model'
            print(f'  Training 7K iters ...')
            metrics = run_3dgs(trial_dir, model_dir, iterations, test_at)

            psnr = metrics.get('train_psnr_7000', None)
            l1 = metrics.get('train_l1_7000', None)

            entry = {
                'scale': scale,
                'sigma_m': sigma,
                'trial': trial,
                'psnr': psnr,
                'l1': l1,
            }
            results.append(entry)

            if psnr is not None:
                print(f'  PSNR = {psnr:.2f} dB, L1 = {l1:.4f}')
            else:
                print(f'  FAILED to parse metrics')
                print(f'  Keys found: {list(metrics.keys())}')

    # Save
    with open(output_base / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Summary table
    print(f'\n\n{"="*60}')
    print(f'POSE NOISE DEGRADATION STUDY — NBV run_072 (48 views)')
    print(f'{"="*60}')
    print(f'{"Scale":<8} {"Sigma (cm)":<12} {"Mean PSNR":<12} {"Std":<8} {"Drop":<8}')
    print(f'{"-"*48}')

    baseline_psnr = None
    for scale in scales:
        trials = [r for r in results if r['scale'] == scale and r['psnr'] is not None]
        if trials:
            psnrs = [t['psnr'] for t in trials]
            mean_p = np.mean(psnrs)
            std_p = np.std(psnrs)
            if baseline_psnr is None:
                baseline_psnr = mean_p
                drop = 0.0
            else:
                drop = baseline_psnr - mean_p
            sigma_cm = scale * REAL_EKF_SIGMA * 100
            print(f'{scale:<8.1f} {sigma_cm:<12.1f} {mean_p:<12.2f} {std_p:<8.2f} {drop:>+7.2f}')


if __name__ == '__main__':
    main()
