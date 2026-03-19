#!/usr/bin/env python3
"""Interactive 3D visualization of NBV planning results.

Generates a plotly HTML figure showing:
  - Rock bounding box (wireframe cube)
  - Orbit camera positions (blue dots)
  - All NBV candidate viewpoints (gray dots, opacity ~ score)
  - Selected top-K viewpoints (red dots)
  - TSP path connecting selected viewpoints (red dashed line)
  - Candidate sphere wireframe (transparent)

Coordinate convention:
  NED (input) -> display as (East=y_ned, North=x_ned, Up=-z_ned)
  so z-axis points up in the visualization.

Usage:
    python3 visualize_nbv.py <run_dir>
    python3 visualize_nbv.py              # defaults to latest run_NNN

Output:
    {run_dir}/nbv/nbv_viewpoints_3d.html  (interactive plotly)
    {run_dir}/nbv/nbv_viewpoints_3d.png   (static matplotlib fallback)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROCK_CENTER_NED = np.array([0.0, 8.0, -0.8])   # North, East, Down
BBOX_SIZE = 2.0
DATA_ROOT = Path.home() / "workspaces" / "gs_ws" / "data" / "nbv"
MAPPING_ROOT = Path.home() / "workspaces" / "gs_ws" / "data" / "mapping"
TOP_K = 24  # number of selected NBV viewpoints


def ned_to_display(pts_ned):
    """Convert NED array (N,3) to display coords (East, North, Up)."""
    pts = np.asarray(pts_ned, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    return np.column_stack([pts[:, 1], pts[:, 0], -pts[:, 2]])


def find_latest_run(data_root):
    """Find the latest run_NNN directory."""
    runs = sorted(
        [d for d in Path(data_root).iterdir()
         if d.is_dir() and d.name.startswith("run_")],
        key=lambda p: int(p.name.split("_")[1])
    )
    if not runs:
        raise FileNotFoundError(f"No run directories found in {data_root}")
    return runs[-1]


def load_transforms(run_dir):
    """Load orbit camera positions from transforms.json.

    Looks in run_dir first, then falls back to data/mapping/run_NNN/.
    """
    # Try run_dir itself
    tf_path = Path(run_dir) / "transforms.json"
    if not tf_path.exists():
        # Fall back to mapping directory with same run number
        run_name = Path(run_dir).name
        tf_path = MAPPING_ROOT / run_name / "transforms.json"
    if not tf_path.exists():
        print(f"WARNING: transforms.json not found for {run_dir}")
        return []

    with open(tf_path) as f:
        data = json.load(f)

    positions = []
    for frame in data.get("frames", []):
        pos = frame.get("position_ned")
        if pos is not None:
            positions.append(pos)
    return positions


def load_candidate_scores(run_dir):
    """Load all scored NBV candidates."""
    path = Path(run_dir) / "nbv" / "candidate_scores.json"
    if not path.exists():
        raise FileNotFoundError(f"candidate_scores.json not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_scoring_history(run_dir):
    """Load scoring history (contains top-K selections)."""
    path = Path(run_dir) / "nbv" / "scoring_history.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_sphere_wireframe(center_ned, radius, n_lines=12, pts_per_line=40):
    """Generate wireframe sphere lines in display coords."""
    center_disp = ned_to_display(center_ned).flatten()
    lines = []

    # Latitude lines
    for i in range(n_lines):
        phi = np.pi * (i + 1) / (n_lines + 1) - np.pi / 2
        theta = np.linspace(0, 2 * np.pi, pts_per_line)
        x = center_disp[0] + radius * np.cos(phi) * np.cos(theta)
        y = center_disp[1] + radius * np.cos(phi) * np.sin(theta)
        z = np.full_like(theta, center_disp[2] + radius * np.sin(phi))
        lines.append((x, y, z))

    # Longitude lines
    for i in range(n_lines):
        theta = 2 * np.pi * i / n_lines
        phi = np.linspace(-np.pi / 2, np.pi / 2, pts_per_line)
        x = center_disp[0] + radius * np.cos(phi) * np.cos(theta)
        y = center_disp[1] + radius * np.cos(phi) * np.sin(theta)
        z = center_disp[2] + radius * np.sin(phi)
        lines.append((x, y, z))

    return lines


def build_cube_wireframe(center_disp, size):
    """Build cube wireframe edges in display coords.

    Returns lists of (x, y, z) line segments with None separators.
    """
    h = size / 2.0
    cx, cy, cz = center_disp

    # 8 corners
    corners = np.array([
        [cx - h, cy - h, cz - h],
        [cx + h, cy - h, cz - h],
        [cx + h, cy + h, cz - h],
        [cx - h, cy + h, cz - h],
        [cx - h, cy - h, cz + h],
        [cx + h, cy - h, cz + h],
        [cx + h, cy + h, cz + h],
        [cx - h, cy + h, cz + h],
    ])

    # 12 edges (pairs of corner indices)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
    ]

    xs, ys, zs = [], [], []
    for i, j in edges:
        xs.extend([corners[i, 0], corners[j, 0], None])
        ys.extend([corners[i, 1], corners[j, 1], None])
        zs.extend([corners[i, 2], corners[j, 2], None])

    return xs, ys, zs


def create_plotly_figure(orbit_positions, candidates, selected, sphere_radius):
    """Build the interactive 3D plotly figure."""
    import plotly.graph_objects as go

    fig = go.Figure()

    rock_disp = ned_to_display(ROCK_CENTER_NED).flatten()

    # --- Rock bounding box (wireframe cube) ---
    bx, by, bz = build_cube_wireframe(rock_disp, BBOX_SIZE)
    fig.add_trace(go.Scatter3d(
        x=bx, y=by, z=bz,
        mode="lines",
        line=dict(color="green", width=3),
        name="Rock BBox",
        hoverinfo="skip",
    ))

    # --- Candidate sphere wireframe ---
    sphere_lines = get_sphere_wireframe(ROCK_CENTER_NED, sphere_radius,
                                         n_lines=8, pts_per_line=30)
    for i, (lx, ly, lz) in enumerate(sphere_lines):
        fig.add_trace(go.Scatter3d(
            x=lx.tolist(), y=ly.tolist(), z=lz.tolist(),
            mode="lines",
            line=dict(color="lightgray", width=1),
            opacity=0.2,
            name="Candidate Sphere" if i == 0 else None,
            showlegend=(i == 0),
            hoverinfo="skip",
        ))

    # --- Orbit positions (blue dots) ---
    if orbit_positions:
        orbit_ned = np.array(orbit_positions)
        orbit_disp = ned_to_display(orbit_ned)
        hover_orbit = [
            f"Orbit KF {i}<br>"
            f"NED: ({orbit_ned[i, 0]:.2f}, {orbit_ned[i, 1]:.2f}, {orbit_ned[i, 2]:.2f})"
            for i in range(len(orbit_ned))
        ]
        fig.add_trace(go.Scatter3d(
            x=orbit_disp[:, 0].tolist(), y=orbit_disp[:, 1].tolist(),
            z=orbit_disp[:, 2].tolist(),
            mode="markers",
            marker=dict(size=5, color="dodgerblue", symbol="circle"),
            name=f"Orbit KFs ({len(orbit_positions)})",
            text=hover_orbit,
            hoverinfo="text",
        ))

    # --- All candidates (gray, opacity ~ score) ---
    if candidates:
        scores = np.array([c["score"] for c in candidates])
        score_min, score_max = scores.min(), scores.max()
        score_range = max(score_max - score_min, 1e-6)
        norm_scores = (scores - score_min) / score_range  # 0..1

        cand_ned = np.array([c["position"] for c in candidates])
        cand_disp = ned_to_display(cand_ned)

        hover_cand = []
        for i, c in enumerate(candidates):
            hover_cand.append(
                f"Candidate {c.get('index', i)}<br>"
                f"NED: ({c['position'][0]:.2f}, {c['position'][1]:.2f}, {c['position'][2]:.2f})<br>"
                f"Azimuth: {c.get('azimuth_deg', 0):.1f} deg<br>"
                f"Altitude: {c.get('altitude', 0):.2f} m<br>"
                f"Score: {c['score']:.3f}<br>"
                f"Geo Uncert: {c.get('geo_uncertainty', 0):.3f}<br>"
                f"Angular: {c.get('score_angular', 0):.3f}<br>"
                f"Rank: {c.get('rank', '?')}"
            )

        # Use opacity proportional to normalized score (min 0.1, max 0.7)
        opacities = 0.1 + 0.6 * norm_scores

        fig.add_trace(go.Scatter3d(
            x=cand_disp[:, 0].tolist(), y=cand_disp[:, 1].tolist(),
            z=cand_disp[:, 2].tolist(),
            mode="markers",
            marker=dict(
                size=3,
                color=scores.tolist(),
                colorscale="Greys",
                opacity=float(np.median(opacities)),
                colorbar=dict(title="Score", x=1.05),
            ),
            name=f"All Candidates ({len(candidates)})",
            text=hover_cand,
            hoverinfo="text",
        ))

    # --- Selected top-K viewpoints (red dots) ---
    if selected:
        sel_ned = np.array([s["position"] for s in selected])
        sel_disp = ned_to_display(sel_ned)

        hover_sel = []
        for i, s in enumerate(selected):
            hover_sel.append(
                f"Selected #{i} (rank {s.get('rank', '?')})<br>"
                f"NED: ({s['position'][0]:.2f}, {s['position'][1]:.2f}, {s['position'][2]:.2f})<br>"
                f"Azimuth: {s.get('azimuth_deg', 0):.1f} deg<br>"
                f"Altitude: {s.get('altitude', 0):.2f} m<br>"
                f"Score: {s['score']:.3f}<br>"
                f"Geo Uncert: {s.get('geo_uncertainty', 0):.3f}<br>"
                f"Angular: {s.get('score_angular', 0):.3f}"
            )

        fig.add_trace(go.Scatter3d(
            x=sel_disp[:, 0].tolist(), y=sel_disp[:, 1].tolist(),
            z=sel_disp[:, 2].tolist(),
            mode="markers",
            marker=dict(size=7, color="red", symbol="circle"),
            name=f"Selected Top-{len(selected)}",
            text=hover_sel,
            hoverinfo="text",
        ))

        # --- TSP path line (red dashed) ---
        # Close the loop for visualization
        path_disp = np.vstack([sel_disp, sel_disp[:1]])
        fig.add_trace(go.Scatter3d(
            x=path_disp[:, 0].tolist(), y=path_disp[:, 1].tolist(),
            z=path_disp[:, 2].tolist(),
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="TSP Path",
            hoverinfo="skip",
        ))

    # --- Layout ---
    fig.update_layout(
        title="NBV Viewpoint Planning — 3D Visualization",
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Up (m)",
            aspectmode="data",
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=40, b=0),
        width=1200,
        height=800,
    )

    return fig


def create_matplotlib_figure(orbit_positions, candidates, selected, sphere_radius):
    """Static 3D plot using matplotlib (fallback)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    rock_disp = ned_to_display(ROCK_CENTER_NED).flatten()

    # Rock bbox
    h = BBOX_SIZE / 2.0
    cx, cy, cz = rock_disp
    for s in [-h, h]:
        for t in [-h, h]:
            ax.plot([cx - h, cx + h], [cy + s, cy + s], [cz + t, cz + t],
                    "g-", linewidth=1)
            ax.plot([cx + s, cx + s], [cy - h, cy + h], [cz + t, cz + t],
                    "g-", linewidth=1)
            ax.plot([cx + s, cx + s], [cy + t, cy + t], [cz - h, cz + h],
                    "g-", linewidth=1)

    # Orbit positions
    if orbit_positions:
        orbit_disp = ned_to_display(np.array(orbit_positions))
        ax.scatter(orbit_disp[:, 0], orbit_disp[:, 1], orbit_disp[:, 2],
                   c="dodgerblue", s=20, label=f"Orbit KFs ({len(orbit_positions)})",
                   depthshade=True)

    # All candidates
    if candidates:
        scores = np.array([c["score"] for c in candidates])
        score_min, score_max = scores.min(), scores.max()
        norm = (scores - score_min) / max(score_max - score_min, 1e-6)

        cand_disp = ned_to_display(np.array([c["position"] for c in candidates]))
        ax.scatter(cand_disp[:, 0], cand_disp[:, 1], cand_disp[:, 2],
                   c="gray", s=8, alpha=0.3,
                   label=f"All Candidates ({len(candidates)})",
                   depthshade=True)

    # Selected viewpoints
    if selected:
        sel_disp = ned_to_display(np.array([s["position"] for s in selected]))
        ax.scatter(sel_disp[:, 0], sel_disp[:, 1], sel_disp[:, 2],
                   c="red", s=50, label=f"Selected Top-{len(selected)}",
                   depthshade=True, zorder=5)

        # TSP path
        path_disp = np.vstack([sel_disp, sel_disp[:1]])
        ax.plot(path_disp[:, 0], path_disp[:, 1], path_disp[:, 2],
                "r--", linewidth=1.5, label="TSP Path")

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title("NBV Viewpoint Planning")
    ax.legend(loc="upper left", fontsize=8)

    return fig


def main():
    # Determine run directory
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1]).expanduser().resolve()
    else:
        run_dir = find_latest_run(DATA_ROOT)

    print(f"Run directory: {run_dir}")

    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    # Load data
    orbit_positions = load_transforms(run_dir)
    print(f"  Orbit positions: {len(orbit_positions)}")

    candidates = load_candidate_scores(run_dir)
    print(f"  Total candidates: {len(candidates)}")

    # Selected = top-K by rank (already sorted by score descending in file)
    selected = [c for c in candidates if c.get("rank", 999) < TOP_K]
    selected.sort(key=lambda c: c["rank"])
    print(f"  Selected viewpoints: {len(selected)}")

    # Get sphere radius from scoring history if available
    sphere_radius = 2.5  # default
    history = load_scoring_history(run_dir)
    if history and len(history) > 0:
        analysis = history[0].get("analysis", {})
        sphere_radius = analysis.get("sphere_radius", 2.5)
    print(f"  Sphere radius: {sphere_radius}")

    # Output directory
    out_dir = run_dir / "nbv"
    os.makedirs(out_dir, exist_ok=True)

    # --- Plotly interactive HTML ---
    try:
        import plotly  # noqa: F401

        fig = create_plotly_figure(orbit_positions, candidates, selected,
                                   sphere_radius)
        html_path = out_dir / "nbv_viewpoints_3d.html"
        fig.write_html(str(html_path), include_plotlyjs=True)
        print(f"  Plotly HTML saved: {html_path}")
    except ImportError:
        print("  WARNING: plotly not installed, skipping HTML output")
        print("  Install with: <workspace>/venv/bin/pip install plotly")

    # --- Matplotlib static PNG ---
    try:
        mpl_fig = create_matplotlib_figure(orbit_positions, candidates,
                                            selected, sphere_radius)
        png_path = out_dir / "nbv_viewpoints_3d.png"
        mpl_fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
        print(f"  Matplotlib PNG saved: {png_path}")
    except Exception as e:
        print(f"  WARNING: matplotlib PNG failed: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
