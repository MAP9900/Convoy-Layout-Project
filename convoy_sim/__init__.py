"""Core package for the Optimal Convoy Layout Project.

The package surfaces a geometry-based simulator for WWII-style convoys and
straight-running torpedoes on a 2D plane measured in meters.
"""

from .geometry import (
    Point2D,
    Vector2D,
    Vec2,
    as_vec,
    bearing_between,
    closest_approach_time,
    distance,
    min_distance_over_interval,
    rotate_point,
    step_position,
    translate_point,
)
from .entities import Convoy, Ship, ShipClass, Torpedo, torpedo_hits_ship
from .layouts import (
    apply_jitter,
    make_hexagonal_convoy,
    make_rectangular_convoy,
    make_staggered_convoy,
)
from .noise import NoiseModel
from .risk import empirical_cvar, empirical_var
from .attackers import fan_spread, parallel_spread
from .simulation import (
    run_monte_carlo_attack,
    sample_fan_spread,
    sample_parallel_spread,
    sample_parallel_torpedoes,
    sample_torpedo_spread_fixed_origin,
    simulate_attack,
    simulate_attack_once,
)
from .minmax_loop import run_minmax_loop
from .ship_catalog import SHIP_CATALOG, make_ship

__all__ = [
    "Vec2",
    "Point2D",
    "Vector2D",
    "as_vec",
    "bearing_between",
    "distance",
    "step_position",
    "closest_approach_time",
    "min_distance_over_interval",
    "rotate_point",
    "translate_point",
    "Ship",
    "ShipClass",
    "Torpedo",
    "Convoy",
    "torpedo_hits_ship",
    "make_rectangular_convoy",
    "make_staggered_convoy",
    "make_hexagonal_convoy",
    "apply_jitter",
    "NoiseModel",
    "empirical_var",
    "empirical_cvar",
    "simulate_attack_once",
    "simulate_attack",
    "run_monte_carlo_attack",
    "fan_spread",
    "parallel_spread",
    "sample_fan_spread",
    "sample_parallel_spread",
    "sample_torpedo_spread_fixed_origin",
    "sample_parallel_torpedoes",
    "run_minmax_loop",
    "SHIP_CATALOG",
    "make_ship",
]
