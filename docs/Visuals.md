Visuals Guide

Overview
This guide lists the available visualization scripts and what they render. All outputs are saved under `results/` and are optional; matplotlib must be installed for plotting functions. ffmpeg needed for MP4 Visuals. 

Plan-View Layouts (E1) FIXED
- Script: `python -m experiments.plot_layout`
- Output: `results/figures/*.png`
- What it shows: plan-view ship centers colored by class or value, with footprint outlines.

Historical vs Optimized Layouts (E1) FIXED
- Script: `python -m experiments.plot_historical_vs_optimized`
- Output:
  - `results/figures/historical_vs_optimized_overlay.png`
  - `results/figures/historical_vs_optimized_grid.png`
- What it shows: overlay and side-by-side comparison of two layouts, plus summary text.

Static Attack Overlay (E2)
- Script: `python -m experiments.plot_attack_once`
- Output:
  - `results/figures/attack_once.png`
  - `results/debug/attack_once.json`
- What it shows: static ships (t=0), torpedo rays, hit markers, and miss annotations.

Temporal Attack Frames (E2)
- Script: `python -m experiments.render_attack_animation`
- Output:
  - `results/frames/demo_attack/frame_*.png`
  - `results/frames/demo_attack.mp4` (optional)
- What it shows: time-indexed ship motion (if dynamics provided) and torpedo trails.
- Notes:
  - MP4 export depends on matplotlib animation support.
  - If animation fails, the PNG frames still render.

Coverage Heatmap (E3)
- Script: `python -m experiments.plot_coverage`
- Output: `results/figures/coverage_heatmap.png`
- What it shows: density map of torpedo paths over space, with optional ship overlay.

Before/After Diagnostics (E3)
- Script: `python -m experiments.run_diagnostics_before_after`
- Output:
  - `results/figures/diag_layout_before_after.png`
  - `results/figures/diag_attack_overlay.png`
  - `results/figures/diag_coverage_compare.png`
  - `results/diag/diagnostics_summary.json`
- What it shows: side-by-side layouts, attack overlays, coverage comparisons, and a JSON summary.

Static vs Dynamic Notes
- Static plots (`plot_attack_once`, plan-view layouts) assume ships are stationary at t=0.
- Dynamic plots (`render_attack_animation`, `render_attack_frame`) move ships according to convoy kinematics when provided.

Prerequisites
- Matplotlib is required for all plotting scripts.
- Core simulation and analysis run without matplotlib installed.
