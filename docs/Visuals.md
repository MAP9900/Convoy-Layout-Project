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
-Currently Optimized layout not based on any RL/ML. Right now Optimized is just an example

Static Attack Overlay (E2) FIXED
- Script: `python -m experiments.plot_attack_once`
- Output:
  - `results/figures/attack_once.png`
  - `results/debug/attack_once.json`
- What it shows: static ships (t=0), torpedo rays, hit markers, and miss annotations.

Temporal Attack Frames (E2) FIXED (Can Edit Depending on What Example One is Looking At)
- Script: `python -m experiments.render_attack_animation`
- Output:
  - `results/frames/demo_attack/frame_*.png`
  - `results/frames/demo_attack.mp4` (optional)
- What it shows: time-indexed ship motion (if dynamics provided) and torpedo trails.
- Notes:
  -Uses the same logic for the simulation for continuity and proper verification.
  - MP4 export depends on matplotlib animation support.
  - If animation fails, the PNG frames still render. 
  -Debug Version added which adds directional arrows to each ship. Point in the direction of the ship's heading. Used to verify ship movements. #TODO Make arrow a vector which arrow size decreasing when speed decreases. 

Before/After Diagnostics (E3) FIXED
- Script: `python -m experiments.run_diagnostics_before_after`
- Output:
  - `results/figures/diag_attack_overlay.png`
  - `results/diag/diagnostics_summary.json`
- What it shows: attack overlays and a JSON summary.

Static vs Dynamic Notes
- Static plots (`plot_attack_once`, plan-view layouts) assume ships are stationary at t=0. Used for initial testing and verification. Static plots used for testing, creating visuals, and simple verifications. Not used in validation prior to RL & ML sections - see dynamic plots for this. 
- Dynamic plots (`render_attack_animation`, `render_attack_frame`) move ships according to convoy kinematics when provided (kinematics same as in the simulation). Primary tool for verification before moving onto the RL & ML portions. 

Prerequisites
- Matplotlib is required for all plotting scripts.
- Core simulation and analysis run without matplotlib installed.
