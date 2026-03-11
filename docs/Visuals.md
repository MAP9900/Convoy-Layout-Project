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

Attack Profile Tests (Notebook)
- Notebook: `docs/notebooks/attack_profile_tests.ipynb`
- Output:
  - `results/frames/attack_profile_previews/<PROFILE_ID>/frame_0001.png`
  - `results/frames/attack_profile_previews/<PROFILE_ID>/frame_1500.png` (middle at 600s, `fps=5`)
  - `results/frames/attack_profile_previews/<PROFILE_ID>/frame_3000.png`
- What it shows:
  - First, middle, and last frame for each attack profile over a 600-second run at `fps=5`.
  - Includes U-boat marker at each profile's `u_pos`.
- Notes:
  - Designed as plug-and-play for when `P01..P25` are updated in `convoy_sim/attack_profiles.py`.
  - Set `SELECT_PROFILE_IDS` in the notebook to preview a subset.
  - Notebook now runs a fast geometry plausibility audit first and writes:
    - `results/diag/attack_profile_geometry_audit.csv`
    - `results/diag/attack_profile_geometry_audit.json`
  - Runtime modes:
    - Fast mode: `RUN_MODE='fast'` (300s horizon, coarse hit dt, profile limit) for iteration.
    - Verify mode: `RUN_MODE='verify'` (600s horizon, fine hit dt, full set) for final validation.

Static vs Dynamic Notes
- Static plots (`plot_attack_once`, plan-view layouts) assume ships are stationary at t=0. Used for initial testing and verification. Static plots used for testing, creating visuals, and simple verifications. Not used in validation prior to RL & ML sections - see dynamic plots for this. 
- Dynamic plots (`render_attack_animation`, `render_attack_frame`) move ships according to convoy kinematics when provided (kinematics same as in the simulation). Primary tool for verification before moving onto the RL & ML portions. 

Prerequisites
- Matplotlib is required for all plotting scripts.
- Core simulation and analysis run without matplotlib installed.

Runtime Expectations (attack_profile_tests notebook)
- Fast mode (`RUN_MODE='fast'`):
  - Intended use: quick iteration while editing/fixing profiles.
  - Typical settings: shorter horizon, coarser hit tracking dt, limited profile count.
  - Expected runtime: minutes (depends on machine and profile count).
- Verify mode (`RUN_MODE='verify'`):
  - Intended use: final validation outputs and review artifacts.
  - Typical settings: 600s horizon, fine hit tracking dt, full profile set.
  - Expected runtime: can be long (often 1-3+ hours depending on hardware and settings).
- Main runtime drivers:
  - Number of profiles rendered.
  - Hit-tracking dt (`HIT_DT`) and total sim duration.
  - Number of saved frames per profile.
  - Matplotlib rendering + image write overhead.

Attack Profile Editing Workflow (Source of Truth)
- Profile definitions are edited in:
  - `convoy_sim/attack_profiles.py` (`build_scaffolded_attack_profile_library()`).
- Recommended process:
  1. Edit `P01..P25` fields (`u_pos`, `base_bearing_rad`, `spread_rad`, timings, weights).
  2. Run geometry plausibility audit:
     - `python -m experiments.audit_attack_profiles --convoy-profile convoy_layout_1`
  3. Review audit outputs:
     - `results/diag/attack_profile_geometry_audit.csv`
     - `results/diag/attack_profile_geometry_audit.json`
  4. Run notebook in fast mode to visually inspect flagged profiles.
  5. Re-edit bad/implausible profiles.
  6. Run notebook in verify mode for full first/middle/last frame outputs.
- Acceptance guidance before RL runs:
  - Keep any misses intentional and plausible.
  - Avoid close-range profiles with bearings that clearly shoot away from target area unless explicitly modeling fire-control error.
  - Treat `implausible_geometry` audit labels as fix-first unless intentional.
