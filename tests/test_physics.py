import math
import numpy as np
import pytest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__))) #Enables Code Runner to work

from convoy.physics import (
    ShipState,
    Torpedo,
    hit_time,
    closest_approach_time,
    distance_at_time,
)

RTOL = 1e-9
ATOL = 1e-9


def test_static_line_closest_distance_matches_analytic():
    """
    1) Static line: stationary ship at (0,0),
       torpedo along +x from x=-1000 at y=+50.
       Closest approach distance should equal |y0| = 50.
    """
    ship = ShipState(x0=0.0, y0=0.0, vx=0.0, vy=0.0, radius=10.0)
    torp = Torpedo(
        x0=-1000.0, y0=50.0,
        ux=1.0, uy=0.0,
        v_t=40.0,
        t_launch=0.0,
        max_run=5000.0,
    )

    t_star = closest_approach_time(ship, torp)
    d_min = distance_at_time(ship, torp, t_star)

    assert d_min == pytest.approx(50.0, rel=RTOL, abs=ATOL)
    # No hit since radius=10 < 50
    assert hit_time(ship, torp) is None


def test_time_gating_closest_before_launch_no_hit():
    """
    2) Time gating: closest approach occurs before t_launch -> no hit.
       Place ship directly behind torpedo direction so closest time is < t_launch.
    """
    ship = ShipState(x0=-50.0, y0=0.0, vx=0.0, vy=0.0, radius=5.0)
    torp = Torpedo(
        x0=0.0, y0=0.0,
        ux=1.0, uy=0.0,   # torpedo looks +x; ship is at x=-50 behind it
        v_t=20.0,
        t_launch=10.0,
        max_run=1000.0,
    )

    t_star = closest_approach_time(ship, torp)
    # Since unconstrained t* < t_launch, it must clamp to t_launch
    assert t_star == pytest.approx(torp.t_launch, rel=RTOL, abs=ATOL)

    ht = hit_time(ship, torp)
    assert ht is None  # no collision after launch


def test_max_run_not_enough_range_no_hit():
    """
    3) Max run: closest approach beyond max_run -> no hit.
       Ship is far ahead along the ray; torpedo can't reach it.
    """
    ship = ShipState(x0=1500.0, y0=0.0, vx=0.0, vy=0.0, radius=10.0)
    torp = Torpedo(
        x0=0.0, y0=0.0,
        ux=1.0, uy=0.0,
        v_t=10.0,
        t_launch=0.0,
        max_run=1000.0,  # can only travel 1000 units, needs 1500
    )

    ht = hit_time(ship, torp)
    assert ht is None

    # Closest approach is at end of run since target lies beyond
    t_star = closest_approach_time(ship, torp)
    assert t_star == pytest.approx(torp.t_launch + torp.max_run / torp.v_t, rel=RTOL, abs=ATOL)


def test_symmetry_mirror_y_axis_same_results():
    """
    4) Symmetry: mirror across y-axis yields same distance/hit result.
       Reflect x positions, x velocities, and torpedo x components.
    """
    ship = ShipState(x0=100.0, y0=20.0, vx=-2.0, vy=1.0, radius=5.0)
    torp = Torpedo(
        x0=-200.0, y0=30.0,
        ux=0.6, uy=0.8,     # already unit length (3-4-5)
        v_t=30.0,
        t_launch=2.5,
        max_run=5000.0,
    )

    # Mirrored across y-axis: x -> -x; vx -> -vx; ux -> -ux
    ship_m = ShipState(x0=-ship.x0, y0=ship.y0, vx=-ship.vx, vy=ship.vy, radius=ship.radius)
    torp_m = Torpedo(
        x0=-torp.x0, y0=torp.y0,
        ux=-torp.ux, uy=torp.uy,
        v_t=torp.v_t,
        t_launch=torp.t_launch,
        max_run=torp.max_run,
    )

    # Compare hit times (both None or equal)
    ht = hit_time(ship, torp)
    ht_m = hit_time(ship_m, torp_m)
    if ht is None or ht_m is None:
        assert (ht is None) == (ht_m is None)
    else:
        assert ht_m == pytest.approx(ht, rel=RTOL, abs=1e-7)

    # Compare closest-approach distances
    t_star = closest_approach_time(ship, torp)
    t_star_m = closest_approach_time(ship_m, torp_m)
    d1 = distance_at_time(ship, torp, t_star)
    d2 = distance_at_time(ship_m, torp_m, t_star_m)
    assert d2 == pytest.approx(d1, rel=RTOL, abs=1e-7)


if __name__ == "__main__":
    # Tiny usage snippet: construct one ship & one torpedo and print hit_time
    ship = ShipState(x0=0.0, y0=0.0, vx=0.0, vy=0.0, radius=20.0)
    torp = Torpedo(
        x0=-1000.0, y0=10.0,
        ux=1.0, uy=0.0,
        v_t=40.0,
        t_launch=0.0,
        max_run=5000.0,
    )
    print("hit_time =", hit_time(ship, torp))