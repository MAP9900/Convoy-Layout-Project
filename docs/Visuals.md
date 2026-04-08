Visuals Guide

Overview
This guide lists the available visualization scripts and what they render. All outputs are saved under `results/` and are optional; matplotlib must be installed for plotting functions. ffmpeg needed for MP4 Visuals. 

Plan-View Layouts (E1) FIXED
- Script: `python -m experiments.plot_layout`
- Output: `results/figures/*.png`
- What it shows: plan-view ship centers colored by class or value, with footprint outlines.

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

Attack Profile Tests (Notebook)
- Notebook: `notebooks/attack_profile_tests.ipynb`
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
  - Rendering now runs through `python -m experiments.render_attack_profile_previews` so notebook and CLI share one path.
  - Notebook now runs a fast geometry plausibility audit first and writes:
    - `results/diag/attack_profile_geometry_audit.csv`
    - `results/diag/attack_profile_geometry_audit.json`
    - `results/diag/attack_profile_hit_report.csv` (per-profile hits, ships hit, torpedoes that hit)
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
  - Process worker count (`PARALLEL_WORKERS`) and trail render settings.

Parallel + Trail Tuning
- For speed with strong verification:
  - Keep `SIM_DURATION_S=600` and `HIT_DT=1/FPS` in verify mode.
  - Increase `PARALLEL_WORKERS` to use CPU cores.
  - Keep trails enabled but reduce cost:
    - shorter trail horizon (`TRAIL_LENGTH_S`, default verify: 20s)
    - thinner/lower-alpha lines (`TRAIL_LINEWIDTH`, `TRAIL_ALPHA`)
    - disable anti-aliasing (`TRAIL_ANTIALIASED=False`)
- CLI equivalent:
  - `python -m experiments.render_attack_profile_previews --convoy-profile convoy_layout_1 --run-mode verify --workers 8 --trail-length-s 20 --trail-linewidth 0.8 --trail-alpha 0.6`

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

Torpedo Firing Doctrine Comparison (Notebook)
- Notebook: `notebooks/torpedo_firing_doctrine_comparison.ipynb`
- What it shows:
  - one large plot per doctrine/motion case instead of a batch grid
  - end-of-firing-cycle snapshots for:
    - longitudinal timing spread
    - uniform divergent spread
    - explicit divergent spread
  - static and moving U-boat variants side by side in notebook order
  - moving U-boat dashed path, red torpedo tracks, and a compact heading table under each plot
- Output behavior:
  - `show_case(..., save=True)` and `show_summary_grid(save=True)` save PNGs by default
  - output directory: `notebooks/results/torpedo_firing_doctrine_comparison/`
- Notes:
  - Intended for doctrine interpretation and visual validation, not convoy-scale run review.
  - Uses focused submarine-centric plotting helpers in `convoy_sim/viz_attack.py` while preserving the repo's existing visual style.
  - The current longitudinal panels are intentionally stylized visualization aids; simulator doctrine still assumes a steady firing course during the salvo for all doctrines unless the stability guardrails are explicitly disabled.
