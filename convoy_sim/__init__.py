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
from .entities import Convoy, Ship, Torpedo, torpedo_hits_ship
from .layouts import (
    hexagonal_layout,
    jittered_layout,
    rectangular_layout,
    staggered_layout,
)
from .simulation import MonteCarloResult, simulate_attack, run_monte_carlo_attack

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
    "Torpedo",
    "Convoy",
    "torpedo_hits_ship",
    "rectangular_layout",
    "staggered_layout",
    "hexagonal_layout",
    "jittered_layout",
    "simulate_attack",
    "run_monte_carlo_attack",
    "MonteCarloResult",
]
