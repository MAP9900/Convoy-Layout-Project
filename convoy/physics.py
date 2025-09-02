from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

__all__ = [
    "ShipState",
    "Torpedo",
    "hit_time",
    "closest_approach_time",
    "distance_at_time",]

EPS = 1e-12


@dataclass(frozen=True)
class ShipState:
    """
    Constant-velocity circular ship.

    Attributes
    ----------
    x0, y0 : float
        Ship position at absolute time t=0.
    vx, vy : float
        Ship velocity components (same units as torpedo speed).
    radius : float
        Collision radius of the ship.
    """
    x0: float
    y0: float
    vx: float
    vy: float
    radius: float


@dataclass(frozen=True)
class Torpedo:
    """
    Straight-running torpedo along a ray from launch time.

    Attributes
    ----------
    x0, y0 : float
        Launch position (torpedo is at this position at t = t_launch).
    ux, uy : float
        Direction vector (need not be normalized; normalization is handled).
    v_t : float
        Torpedo speed (> 0).
    t_launch : float
        Absolute launch time.
    max_run : float
        Maximum travel distance before the torpedo becomes inactive.
    """
    x0: float
    y0: float
    ux: float
    uy: float
    v_t: float
    t_launch: float
    max_run: float


def _unit(vec: np.ndarray) -> np.ndarray:
    """Return the unit vector; if zero, returns zero vector."""
    n = float(np.hypot(vec[0], vec[1]))
    if n < EPS:
        return np.array([0.0, 0.0], dtype=float)
    return vec / n


def _ship_pos(ship: ShipState, t: float) -> np.ndarray:
    return np.array([ship.x0 + ship.vx * t, ship.y0 + ship.vy * t], dtype=float)


def _torpedo_pos(torp: Torpedo, t: float) -> np.ndarray:
    """
    Piecewise torpedo position:
    - For t < t_launch: sits at launch point (not yet moving).
    - For t in [t_launch, t_end]: moves along direction at speed v_t.
    - For t > t_end: clamped to end-of-run position.
    """
    u_hat = _unit(np.array([torp.ux, torp.uy], dtype=float))
    t0 = torp.t_launch
    if torp.v_t <= 0.0 or torp.max_run <= 0.0 or u_hat.dot(u_hat) < EPS:
        # No movement (degenerate) -> always at launch point.
        return np.array([torp.x0, torp.y0], dtype=float)
    t1 = t0 + torp.max_run / torp.v_t

    if t <= t0 + EPS:
        dt = 0.0
    elif t >= t1 - EPS:
        dt = t1 - t0
    else:
        dt = t - t0

    return np.array([torp.x0, torp.y0], dtype=float) + u_hat * torp.v_t * dt


def distance_at_time(ship_state: ShipState, torpedo: Torpedo, t: float) -> float:
    """
    Euclidean separation between ship and torpedo at time t (with torpedo
    clamped before launch and after max run).
    """
    s = _ship_pos(ship_state, t)
    p = _torpedo_pos(torpedo, t)
    return float(np.hypot(*(s - p)))


def closest_approach_time(ship_state: ShipState, torpedo: Torpedo) -> float:
    """
    Time of closest approach between ship and torpedo, clamped to the
    active interval [t_launch, t_launch + max_run / v_t].

    Returns
    -------
    float
        Time in absolute units, always within the clamped interval.
    """
    u_hat = _unit(np.array([torpedo.ux, torpedo.uy], dtype=float))
    t0 = torpedo.t_launch
    t1 = t0 if torpedo.v_t <= 0.0 or u_hat.dot(u_hat) < EPS else t0 + max(torpedo.max_run, 0.0) / max(torpedo.v_t, EPS)

    # Relative motion representation: Δ(t) = B + A t
    v_s = np.array([ship_state.vx, ship_state.vy], dtype=float)
    A = v_s - torpedo.v_t * u_hat
    B = np.array([ship_state.x0 - torpedo.x0, ship_state.y0 - torpedo.y0], dtype=float) + torpedo.v_t * u_hat * t0

    a = float(A.dot(A))
    if a < EPS:
        # Relative velocity ~ 0 -> distance ~ constant; closest anywhere in window.
        return float(t0)

    t_star = -float(A.dot(B)) / a  # Unconstrained minimizer of ||B + A t||^2
    # Clamp to active window
    if t_star < t0:
        return float(t0)
    if t_star > t1:
        return float(t1)
    return float(t_star)


def hit_time(ship_state: ShipState, torpedo: Torpedo) -> Optional[float]:
    """
    First collision time (ship treated as circle), or None if no hit within
    [t_launch, t_launch + max_run / v_t].

    Notes
    -----
    Solves ||B + A t||^2 = R^2 for absolute time t and selects the earliest
    root within the active interval. Handles degeneracies robustly.
    """
    R = float(ship_state.radius)
    if R < 0.0:
        return None  # Nonsensical radius

    u_hat = _unit(np.array([torpedo.ux, torpedo.uy], dtype=float))
    t0 = torpedo.t_launch
    # Handle degenerate run window safely
    if torpedo.v_t <= 0.0 or torpedo.max_run <= 0.0 or u_hat.dot(u_hat) < EPS:
        # Only possible "hit" is if already within radius at launch.
        d0 = distance_at_time(ship_state, torpedo, t0)
        return float(t0) if d0 <= R + 1e-12 else None

    t1 = t0 + torpedo.max_run / torpedo.v_t

    # Quick check: already inside at launch.
    if distance_at_time(ship_state, torpedo, t0) <= R + 1e-12:
        return float(t0)

    # Build quadratic for ||B + A t||^2 = R^2
    v_s = np.array([ship_state.vx, ship_state.vy], dtype=float)
    A = v_s - torpedo.v_t * u_hat
    B = np.array([ship_state.x0 - torpedo.x0, ship_state.y0 - torpedo.y0], dtype=float) + torpedo.v_t * u_hat * t0

    a = float(A.dot(A))
    b = 2.0 * float(A.dot(B))
    c = float(B.dot(B)) - R * R

    if a < EPS:
        # Relative velocity ~ 0 -> distance constant; no new intersections.
        return None

    disc = b * b - 4.0 * a * c
    if disc < -1e-10:
        return None
    if disc < 0.0:
        disc = 0.0  # grazing due to round-off

    sqrt_disc = float(np.sqrt(disc))
    # Numerically stable quadratic roots
    # Compute both and pick earliest within [t0, t1].
    t_candidates = []
    inv_2a = 0.5 / a
    t_minus = (-b - sqrt_disc) * inv_2a
    t_plus = (-b + sqrt_disc) * inv_2a

    # We want the first time >= t0.
    for t in (t_minus, t_plus):
        if t0 - 1e-12 <= t <= t1 + 1e-12:
            t_candidates.append(float(t))

    if not t_candidates:
        return None

    t_first = min(t_candidates)
    if t_first < t0:
        t_first = t0
    if t_first > t1:
        return None

    # Final guard: ensure actual distance is within radius at the chosen time.
    if distance_at_time(ship_state, torpedo, t_first) <= R + 1e-9:
        return float(t_first)
    return None