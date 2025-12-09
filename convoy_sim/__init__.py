"""Core package for the Optimal Convoy Layout Project.

The package surfaces a geometry-based simulator for WWII-style convoys and
straight-running torpedoes on a 2D plane measured in meters.
"""

from .geometry import (
    Point2D,
    Vector2D,
    bearing_between,
    rotate_point,
    translate_point,
)
from .entities import Convoy, Ship, Torpedo
from .layouts import (
    hexagonal_layout,
    jittered_layout,
    rectangular_layout,
    staggered_layout,
)
from .simulation import MonteCarloResult, simulate_attack, run_monte_carlo_attack

__all__ = [
    "Point2D",
    "Vector2D",
    "bearing_between",
    "rotate_point",
    "translate_point",
    "Ship",
    "Torpedo",
    "Convoy",
    "rectangular_layout",
    "staggered_layout",
    "hexagonal_layout",
    "jittered_layout",
    "simulate_attack",
    "run_monte_carlo_attack",
    "MonteCarloResult",
]
