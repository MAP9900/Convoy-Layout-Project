"""Core package for the Optimal Convoy Layout Project.

The package surfaces a geometry-based simulator for WWII-style convoys and
straight-running torpedoes on a 2D plane measured in meters.
"""

from convoy_sim.geometry import (
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
from convoy_sim.entities import Convoy, Ship, ShipClass, Torpedo, torpedo_hits_ship
from convoy_sim.layouts import (
    apply_jitter,
    make_hexagonal_convoy,
    make_rectangular_convoy,
    make_staggered_convoy,
)
from convoy_sim.noise import NoiseModel
from convoy_sim.risk import empirical_cvar, empirical_var
from convoy_sim.objectives import ObjectiveSpec, aggregate_objective, score_trial_result
from convoy_sim.attackers import fan_spread, parallel_spread
from convoy_sim.rl_wrapper import ActionSpaceMap, OBS_SCHEMA_VERSION, RLEpisode, build_observation
from convoy_sim.defender_policy import (
    DefenderPolicy,
    LayoutAction,
    ThreatPrior,
    ThreatType,
    evaluate_defender_policy,
    make_deterministic_policy,
    make_uniform_policy,
)
from convoy_sim.simulation import (
    run_monte_carlo_attack,
    run_monte_carlo_attack_dynamic,
    run_monte_carlo_attack_scored,
    sample_fan_spread,
    sample_parallel_spread,
    sample_parallel_torpedoes,
    sample_torpedo_spread_fixed_origin,
    simulate_attack,
    simulate_attack_dynamic_once,
    simulate_attack_once,
    simulate_attack_static,
    simulate_attack_once_scored,
    torpedo_hits_ship_dynamic,
)
from convoy_sim.ship_catalog import SHIP_CATALOG, make_ship
from convoy_sim.attack_profiles import (
    AttackProfile,
    AttackProfileLibrary,
    build_scaffolded_attack_profile_library,
    make_placeholder_profile,
    make_placeholder_profile_library,
    DEFAULT_ATTACK_PROFILE_LIBRARY,
)

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
    "ObjectiveSpec",
    "score_trial_result",
    "aggregate_objective",
    "simulate_attack_once",
    "simulate_attack_once_scored",
    "simulate_attack",
    "simulate_attack_static",
    "simulate_attack_dynamic_once",
    "run_monte_carlo_attack",
    "run_monte_carlo_attack_dynamic",
    "run_monte_carlo_attack_scored",
    "torpedo_hits_ship_dynamic",
    "fan_spread",
    "parallel_spread",
    "ActionSpaceMap",
    "OBS_SCHEMA_VERSION",
    "RLEpisode",
    "build_observation",
    "ThreatType",
    "ThreatPrior",
    "LayoutAction",
    "DefenderPolicy",
    "make_uniform_policy",
    "make_deterministic_policy",
    "evaluate_defender_policy",
    "sample_fan_spread",
    "sample_parallel_spread",
    "sample_torpedo_spread_fixed_origin",
    "sample_parallel_torpedoes",
    "SHIP_CATALOG",
    "make_ship",
    "AttackProfile",
    "AttackProfileLibrary",
    "build_scaffolded_attack_profile_library",
    "make_placeholder_profile",
    "make_placeholder_profile_library",
    "DEFAULT_ATTACK_PROFILE_LIBRARY",
]
