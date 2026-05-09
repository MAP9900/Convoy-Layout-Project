"""Target-zone geometry helpers for synthetic attack-profile generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from convoy_sim.entities import Ship


APPROACH_SIDES: tuple[str, ...] = (
    "port",
    "starboard",
    "ahead",
    "astern",
    "port_forward_quarter",
    "starboard_forward_quarter",
    "port_aft_quarter",
    "starboard_aft_quarter",
)


@dataclass(frozen=True)
class ConvoyFrame:
    """Convoy-local coordinate frame.

    Local x is aligned with convoy heading. Local y is the port-side lateral axis.
    """

    origin: tuple[float, float]
    heading_rad: float

    @classmethod
    def from_ships(cls, ships: Sequence[Ship]) -> "ConvoyFrame":
        if not ships:
            raise ValueError("ships must be non-empty")
        positions = np.asarray([np.asarray(ship.position, dtype=float) for ship in ships], dtype=float)
        origin = np.mean(positions, axis=0)
        return cls(origin=(float(origin[0]), float(origin[1])), heading_rad=float(ships[0].heading_rad))

    @property
    def _rotation(self) -> np.ndarray:
        cos_h = math.cos(self.heading_rad)
        sin_h = math.sin(self.heading_rad)
        return np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=float)

    def world_to_local(self, point: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(point, dtype=float)
        if arr.shape != (2,):
            raise ValueError("point must be a 2D vector")
        return self._rotation.T @ (arr - np.asarray(self.origin, dtype=float))

    def local_to_world(self, point: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(point, dtype=float)
        if arr.shape != (2,):
            raise ValueError("point must be a 2D vector")
        return np.asarray(self.origin, dtype=float) + self._rotation @ arr


@dataclass(frozen=True)
class ConvoyEnvelope:
    """Local-space convoy extents and derived spacing estimates."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    spacing_x_m: float
    spacing_y_m: float

    @classmethod
    def from_local_positions(cls, local_positions: np.ndarray) -> "ConvoyEnvelope":
        if local_positions.ndim != 2 or local_positions.shape[1] != 2 or local_positions.shape[0] == 0:
            raise ValueError("local_positions must have shape (n, 2)")
        unique_x = _unique_sorted_with_tolerance(local_positions[:, 0])
        unique_y = _unique_sorted_with_tolerance(local_positions[:, 1])
        return cls(
            min_x=float(np.min(local_positions[:, 0])),
            max_x=float(np.max(local_positions[:, 0])),
            min_y=float(np.min(local_positions[:, 1])),
            max_y=float(np.max(local_positions[:, 1])),
            spacing_x_m=_median_spacing(unique_x, fallback=500.0),
            spacing_y_m=_median_spacing(unique_y, fallback=900.0),
        )


@dataclass(frozen=True)
class AttackIntent:
    """Target-relative generation intent stored alongside synthetic profiles."""

    target_zone_id: str
    target_zone_kind: str
    target_ship_ids: tuple[str, ...]
    target_point: tuple[float, float]
    target_local: tuple[float, float]
    spawn_local: tuple[float, float]
    approach_side: str
    approach_lane: str
    range_to_target_m: float
    planned_bearing_error_deg: float
    intended_label: str
    convoy_heading_rad: float
    spawn_region: str = ""
    inside_convoy_envelope: bool = False
    target_aspect_deg: float = 0.0
    target_score: float = 0.0
    nearest_ship_clearance_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_zone_id": self.target_zone_id,
            "target_zone_kind": self.target_zone_kind,
            "target_ship_ids": list(self.target_ship_ids),
            "target_point": [float(v) for v in self.target_point],
            "target_local": [float(v) for v in self.target_local],
            "spawn_local": [float(v) for v in self.spawn_local],
            "approach_side": self.approach_side,
            "approach_lane": self.approach_lane,
            "range_to_target_m": float(self.range_to_target_m),
            "planned_bearing_error_deg": float(self.planned_bearing_error_deg),
            "intended_label": self.intended_label,
            "convoy_heading_rad": float(self.convoy_heading_rad),
            "spawn_region": self.spawn_region,
            "inside_convoy_envelope": bool(self.inside_convoy_envelope),
            "target_aspect_deg": float(self.target_aspect_deg),
            "target_score": float(self.target_score),
            "nearest_ship_clearance_m": float(self.nearest_ship_clearance_m),
        }


def _unique_sorted_with_tolerance(values: np.ndarray, *, tol: float = 1e-6) -> list[float]:
    sorted_values = sorted(float(v) for v in values)
    unique: list[float] = []
    for value in sorted_values:
        if not unique or abs(value - unique[-1]) > tol:
            unique.append(value)
    return unique


def _median_spacing(values: Sequence[float], *, fallback: float) -> float:
    if len(values) < 2:
        return float(fallback)
    diffs = np.diff(np.asarray(values, dtype=float))
    positive = diffs[diffs > 1e-6]
    if positive.size == 0:
        return float(fallback)
    return float(np.median(positive))


def _approach_unit_local(approach_side: str) -> np.ndarray:
    directions: Mapping[str, tuple[float, float]] = {
        "port": (0.0, 1.0),
        "starboard": (0.0, -1.0),
        "ahead": (1.0, 0.0),
        "astern": (-1.0, 0.0),
        "port_forward_quarter": (1.0, 1.0),
        "starboard_forward_quarter": (1.0, -1.0),
        "port_aft_quarter": (-1.0, 1.0),
        "starboard_aft_quarter": (-1.0, -1.0),
    }
    if approach_side not in directions:
        raise ValueError(f"Unknown approach_side: {approach_side}")
    vec = np.asarray(directions[approach_side], dtype=float)
    return vec / float(np.linalg.norm(vec))


def _ship_id(ship: Ship) -> str:
    return str(getattr(ship, "id", "unknown"))


def _zone_kind_for_ship(local: np.ndarray, envelope: ConvoyEnvelope) -> str:
    x_margin = max(0.35 * envelope.spacing_x_m, 1.0)
    y_margin = max(0.35 * envelope.spacing_y_m, 1.0)
    if local[1] >= envelope.max_y - y_margin:
        return "port_edge"
    if local[1] <= envelope.min_y + y_margin:
        return "starboard_edge"
    if local[0] >= envelope.max_x - x_margin:
        return "lead_column"
    if local[0] <= envelope.min_x + x_margin:
        return "trailing_column"
    return "interior"


def _lane_for_approach(approach_side: str, target_local: np.ndarray, envelope: ConvoyEnvelope) -> str:
    if approach_side in {"port", "starboard"}:
        if target_local[0] >= envelope.max_x - 0.35 * envelope.spacing_x_m:
            return f"{approach_side}_lead_edge"
        if target_local[0] <= envelope.min_x + 0.35 * envelope.spacing_x_m:
            return f"{approach_side}_trailing_edge"
        return f"{approach_side}_broadside"
    if approach_side in {"ahead", "astern"}:
        if target_local[1] >= envelope.max_y - 0.35 * envelope.spacing_y_m:
            return f"{approach_side}_port_edge"
        if target_local[1] <= envelope.min_y + 0.35 * envelope.spacing_y_m:
            return f"{approach_side}_starboard_edge"
        return f"{approach_side}_center_lane"
    return approach_side


def convoy_frame_and_envelope(ships: Sequence[Ship]) -> tuple[ConvoyFrame, ConvoyEnvelope, np.ndarray]:
    """Return local frame, envelope, and ship local positions."""

    frame = ConvoyFrame.from_ships(ships)
    local_positions = np.asarray([frame.world_to_local(ship.position) for ship in ships], dtype=float)
    envelope = ConvoyEnvelope.from_local_positions(local_positions)
    return frame, envelope, local_positions


def _inside_envelope(local: np.ndarray, envelope: ConvoyEnvelope) -> bool:
    return bool(
        envelope.min_x <= float(local[0]) <= envelope.max_x
        and envelope.min_y <= float(local[1]) <= envelope.max_y
    )


def _nearest_clearance_m(local: np.ndarray, local_positions: np.ndarray) -> float:
    distances = np.linalg.norm(local_positions - np.asarray(local, dtype=float), axis=1)
    return float(np.min(distances))


def _column_gap_centers(local_positions: np.ndarray, envelope: ConvoyEnvelope) -> np.ndarray:
    unique_y = _unique_sorted_with_tolerance(local_positions[:, 1])
    if len(unique_y) < 2:
        return np.asarray([0.5 * (envelope.min_y + envelope.max_y)], dtype=float)
    return 0.5 * (np.asarray(unique_y[:-1], dtype=float) + np.asarray(unique_y[1:], dtype=float))


def _sample_gap_y(
    *,
    rng: np.random.Generator,
    local_positions: np.ndarray,
    envelope: ConvoyEnvelope,
) -> float:
    centers = _column_gap_centers(local_positions, envelope)
    center = float(rng.choice(centers))
    jitter = float(rng.uniform(-0.22, 0.22) * envelope.spacing_y_m)
    return float(np.clip(center + jitter, envelope.min_y, envelope.max_y))


def _sample_tactical_spawn_local(
    *,
    rng: np.random.Generator,
    envelope: ConvoyEnvelope,
    local_positions: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Sample a historically motivated tactical U-boat spawn in convoy-local space."""

    region = str(
        rng.choice(
            np.asarray(
                [
                    "outside_perimeter",
                    "ahead_beam",
                    "beam_attack",
                    "astern_infiltration",
                    "inside_columns",
                ],
                dtype=object,
            ),
            p=np.asarray([0.35, 0.20, 0.15, 0.15, 0.15], dtype=float),
        )
    )
    center_x = 0.5 * (envelope.min_x + envelope.max_x)
    center_y = 0.5 * (envelope.min_y + envelope.max_y)
    half_x = 0.5 * (envelope.max_x - envelope.min_x)
    half_y = 0.5 * (envelope.max_y - envelope.min_y)

    if region == "outside_perimeter":
        angle = float(rng.uniform(-np.pi, np.pi))
        standoff = float(rng.uniform(700.0, 3800.0))
        spawn = np.asarray(
            [
                center_x + np.cos(angle) * (half_x + standoff),
                center_y + np.sin(angle) * (half_y + standoff),
            ],
            dtype=float,
        )
    elif region == "ahead_beam":
        side = -1.0 if float(rng.random()) < 0.5 else 1.0
        spawn = np.asarray(
            [
                envelope.max_x + float(rng.uniform(450.0, 3200.0)),
                float(rng.uniform(envelope.min_y - 0.25 * envelope.spacing_y_m, envelope.max_y + 0.25 * envelope.spacing_y_m))
                + side * float(rng.uniform(0.0, 0.35 * envelope.spacing_y_m)),
            ],
            dtype=float,
        )
    elif region == "beam_attack":
        side = -1.0 if float(rng.random()) < 0.5 else 1.0
        edge_y = envelope.max_y if side > 0.0 else envelope.min_y
        spawn = np.asarray(
            [
                float(rng.uniform(envelope.min_x - 0.35 * envelope.spacing_x_m, envelope.max_x + 0.35 * envelope.spacing_x_m)),
                edge_y + side * float(rng.uniform(500.0, 3200.0)),
            ],
            dtype=float,
        )
    elif region == "astern_infiltration":
        spawn = np.asarray(
            [
                float(rng.uniform(envelope.min_x - 1800.0, envelope.min_x + 0.35 * (envelope.max_x - envelope.min_x))),
                _sample_gap_y(rng=rng, local_positions=local_positions, envelope=envelope),
            ],
            dtype=float,
        )
    else:
        spawn = np.asarray(
            [
                float(rng.uniform(envelope.min_x + 0.12 * envelope.spacing_x_m, envelope.max_x - 0.12 * envelope.spacing_x_m)),
                _sample_gap_y(rng=rng, local_positions=local_positions, envelope=envelope),
            ],
            dtype=float,
        )
    return spawn, region


def _approach_side_from_spawn(spawn_local: np.ndarray, envelope: ConvoyEnvelope) -> str:
    if _inside_envelope(spawn_local, envelope):
        return "inside_columns"
    dx_ahead = max(0.0, float(spawn_local[0] - envelope.max_x))
    dx_astern = max(0.0, float(envelope.min_x - spawn_local[0]))
    dy_port = max(0.0, float(spawn_local[1] - envelope.max_y))
    dy_starboard = max(0.0, float(envelope.min_y - spawn_local[1]))
    if dx_ahead > 0.0 and dy_port > 0.0:
        return "port_forward_quarter"
    if dx_ahead > 0.0 and dy_starboard > 0.0:
        return "starboard_forward_quarter"
    if dx_astern > 0.0 and dy_port > 0.0:
        return "port_aft_quarter"
    if dx_astern > 0.0 and dy_starboard > 0.0:
        return "starboard_aft_quarter"
    if dy_port >= max(dx_ahead, dx_astern):
        return "port"
    if dy_starboard >= max(dx_ahead, dx_astern):
        return "starboard"
    if dx_ahead >= dx_astern:
        return "ahead"
    return "astern"


def _target_aspect_deg(spawn_local: np.ndarray, target_local: np.ndarray) -> float:
    line = np.asarray(spawn_local, dtype=float) - np.asarray(target_local, dtype=float)
    dist = float(np.linalg.norm(line))
    if dist <= 1e-9:
        return 0.0
    local_heading = np.asarray([1.0, 0.0], dtype=float)
    cos_val = abs(float(np.dot(line / dist, local_heading)))
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def _score_target_candidate(
    *,
    ship: Ship,
    spawn_local: np.ndarray,
    target_local: np.ndarray,
    spawn_region: str,
    envelope: ConvoyEnvelope,
) -> float:
    range_m = float(np.linalg.norm(target_local - spawn_local))
    if range_m < 300.0 or range_m > 4800.0:
        return -1e9
    aspect_deg = _target_aspect_deg(spawn_local, target_local)
    broadside_bonus = 1.0 - abs(aspect_deg - 90.0) / 90.0
    ideal_range = 850.0 if spawn_region in {"inside_columns", "astern_infiltration"} else 1600.0
    range_bonus = float(np.exp(-((range_m - ideal_range) / 1400.0) ** 2))
    central_y = 1.0 - min(1.0, abs(float(target_local[1])) / max(abs(envelope.max_y), abs(envelope.min_y), 1.0))
    value_bonus = float(getattr(ship, "value_weight", 1.0))
    return float((1.6 * value_bonus) + (1.7 * broadside_bonus) + (1.3 * range_bonus) + (0.35 * central_y))


def _select_tactical_target(
    *,
    ships: Sequence[Ship],
    spawn_local: np.ndarray,
    spawn_region: str,
    envelope: ConvoyEnvelope,
    local_positions: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, float, float]:
    scores = np.asarray(
        [
            _score_target_candidate(
                ship=ship,
                spawn_local=spawn_local,
                target_local=local_positions[idx],
                spawn_region=spawn_region,
                envelope=envelope,
            )
            for idx, ship in enumerate(ships)
        ],
        dtype=float,
    )
    viable = np.where(scores > -1e8)[0]
    if viable.size == 0:
        distances = np.linalg.norm(local_positions - spawn_local, axis=1)
        ship_index = int(np.argmin(distances))
        return ship_index, float(scores[ship_index]), _target_aspect_deg(spawn_local, local_positions[ship_index])
    viable_scores = scores[viable]
    top_n = min(8, int(viable.size))
    top_order = viable[np.argsort(viable_scores)[-top_n:]]
    top_scores = scores[top_order]
    scaled = top_scores - float(np.max(top_scores))
    probs = np.exp(scaled / 0.65)
    probs = probs / float(np.sum(probs))
    ship_index = int(rng.choice(top_order, p=probs))
    return ship_index, float(scores[ship_index]), _target_aspect_deg(spawn_local, local_positions[ship_index])


def sample_random_tactical_attack_intent(
    ships: Sequence[Ship],
    *,
    rng: np.random.Generator,
    sequence_id: int,
    intended_label: str,
    min_clearance_m: float = 250.0,
    max_attempts: int = 100,
) -> AttackIntent:
    """Sample v4 tactical intent by choosing spawn first, then target/firing intent."""

    if intended_label not in {"credible_hit_threat", "credible_near_miss"}:
        raise ValueError("intended_label must be credible_hit_threat or credible_near_miss")
    frame, envelope, local_positions = convoy_frame_and_envelope(ships)
    for _ in range(int(max_attempts)):
        spawn_local, spawn_region = _sample_tactical_spawn_local(
            rng=rng,
            envelope=envelope,
            local_positions=local_positions,
        )
        clearance_m = _nearest_clearance_m(spawn_local, local_positions)
        if clearance_m < float(min_clearance_m):
            continue
        ship_index, target_score, aspect_deg = _select_tactical_target(
            ships=ships,
            spawn_local=spawn_local,
            spawn_region=spawn_region,
            envelope=envelope,
            local_positions=local_positions,
            rng=rng,
        )
        ship = ships[ship_index]
        target_jitter_local = np.asarray(
            [
                float(rng.uniform(-0.28, 0.28) * float(ship.length)),
                float(rng.uniform(-0.28, 0.28) * float(ship.beam)),
            ],
            dtype=float,
        )
        target_local = local_positions[ship_index] + target_jitter_local
        target_point = frame.local_to_world(target_local)
        zone_kind = _zone_kind_for_ship(local_positions[ship_index], envelope)
        approach_side = _approach_side_from_spawn(spawn_local, envelope)
        if intended_label == "credible_hit_threat":
            planned_error = float(rng.uniform(-4.5, 4.5))
        else:
            miss_sign = -1.0 if float(rng.random()) < 0.5 else 1.0
            planned_error = float(miss_sign * rng.uniform(8.5, 11.5))
        return AttackIntent(
            target_zone_id=f"TV4{int(sequence_id):06d}_{spawn_region}_{zone_kind}",
            target_zone_kind=zone_kind,
            target_ship_ids=(_ship_id(ship),),
            target_point=(float(target_point[0]), float(target_point[1])),
            target_local=(float(target_local[0]), float(target_local[1])),
            spawn_local=(float(spawn_local[0]), float(spawn_local[1])),
            approach_side=approach_side,
            approach_lane=f"{spawn_region}:{approach_side}",
            range_to_target_m=float(np.linalg.norm(target_local - spawn_local)),
            planned_bearing_error_deg=planned_error,
            intended_label=intended_label,
            convoy_heading_rad=float(frame.heading_rad),
            spawn_region=spawn_region,
            inside_convoy_envelope=_inside_envelope(spawn_local, envelope),
            target_aspect_deg=float(aspect_deg),
            target_score=float(target_score),
            nearest_ship_clearance_m=float(clearance_m),
        )
    raise ValueError("Unable to sample tactical intent satisfying minimum clearance")


def build_curated_attack_intents(
    ships: Sequence[Ship],
    *,
    count: int,
    start_index: int = 1,
) -> list[AttackIntent]:
    """Build deterministic target-zone intents for smoke/regression datasets."""

    if count <= 0:
        raise ValueError("count must be positive")
    frame, envelope, local_positions = convoy_frame_and_envelope(ships)
    ordered_indices = np.argsort(local_positions[:, 0] + 0.25 * local_positions[:, 1])
    intents: list[AttackIntent] = []
    for offset in range(count):
        ship_index = int(ordered_indices[offset % len(ordered_indices)])
        approach_side = APPROACH_SIDES[offset % len(APPROACH_SIDES)]
        intended_label = "credible_hit_threat" if offset % 4 != 3 else "credible_near_miss"
        standoff_m = (1350.0, 1900.0, 2600.0, 3300.0)[offset % 4]
        planned_error = 0.0 if intended_label == "credible_hit_threat" else (9.2 if offset % 2 else -9.2)
        intents.append(
            make_attack_intent(
                ships=ships,
                ship_index=ship_index,
                approach_side=approach_side,
                standoff_m=standoff_m,
                planned_bearing_error_deg=planned_error,
                intended_label=intended_label,
                frame=frame,
                envelope=envelope,
                local_positions=local_positions,
                sequence_id=start_index + offset,
                target_jitter_local=(0.0, 0.0),
            )
        )
    return intents


def sample_random_attack_intent(
    ships: Sequence[Ship],
    *,
    rng: np.random.Generator,
    sequence_id: int,
    intended_label: str,
) -> AttackIntent:
    """Sample one structured-random target-zone intent."""

    if intended_label not in {"credible_hit_threat", "credible_near_miss"}:
        raise ValueError("intended_label must be credible_hit_threat or credible_near_miss")
    frame, envelope, local_positions = convoy_frame_and_envelope(ships)
    indices_by_zone: dict[str, list[int]] = {}
    for idx, local in enumerate(local_positions):
        indices_by_zone.setdefault(_zone_kind_for_ship(local, envelope), []).append(int(idx))
    zone_kind = str(rng.choice(np.asarray(sorted(indices_by_zone), dtype=object)))
    ship_index = int(rng.choice(np.asarray(indices_by_zone[zone_kind], dtype=int)))
    approach_side = str(rng.choice(np.asarray(APPROACH_SIDES, dtype=object)))
    standoff_m = float(rng.uniform(950.0, 3800.0))
    if intended_label == "credible_hit_threat":
        planned_error = float(rng.uniform(-5.0, 5.0))
    else:
        miss_sign = -1.0 if float(rng.random()) < 0.5 else 1.0
        planned_error = float(miss_sign * rng.uniform(8.4, 11.0))
    ship = ships[ship_index]
    target_jitter_local = (
        float(rng.uniform(-0.35, 0.35) * float(ship.length)),
        float(rng.uniform(-0.35, 0.35) * float(ship.beam)),
    )
    return make_attack_intent(
        ships=ships,
        ship_index=ship_index,
        approach_side=approach_side,
        standoff_m=standoff_m,
        planned_bearing_error_deg=planned_error,
        intended_label=intended_label,
        frame=frame,
        envelope=envelope,
        local_positions=local_positions,
        sequence_id=sequence_id,
        target_jitter_local=target_jitter_local,
    )


def make_attack_intent(
    *,
    ships: Sequence[Ship],
    ship_index: int,
    approach_side: str,
    standoff_m: float,
    planned_bearing_error_deg: float,
    intended_label: str,
    frame: ConvoyFrame,
    envelope: ConvoyEnvelope,
    local_positions: np.ndarray,
    sequence_id: int,
    target_jitter_local: tuple[float, float],
) -> AttackIntent:
    """Build an attack intent from selected target and approach parameters."""

    if not 0 <= ship_index < len(ships):
        raise ValueError("ship_index out of range")
    target_local = local_positions[ship_index] + np.asarray(target_jitter_local, dtype=float)
    approach_unit = _approach_unit_local(approach_side)
    spawn_local = target_local + approach_unit * float(standoff_m)
    access_margin_x = max(400.0, 0.45 * envelope.spacing_x_m)
    access_margin_y = max(400.0, 0.45 * envelope.spacing_y_m)
    if approach_unit[0] > 0.0:
        spawn_local[0] = max(float(spawn_local[0]), envelope.max_x + access_margin_x)
    elif approach_unit[0] < 0.0:
        spawn_local[0] = min(float(spawn_local[0]), envelope.min_x - access_margin_x)
    if approach_unit[1] > 0.0:
        spawn_local[1] = max(float(spawn_local[1]), envelope.max_y + access_margin_y)
    elif approach_unit[1] < 0.0:
        spawn_local[1] = min(float(spawn_local[1]), envelope.min_y - access_margin_y)
    target_point = frame.local_to_world(target_local)
    zone_kind = _zone_kind_for_ship(local_positions[ship_index], envelope)
    approach_lane = _lane_for_approach(approach_side, target_local, envelope)
    range_to_target_m = float(np.linalg.norm(spawn_local - target_local))
    return AttackIntent(
        target_zone_id=f"TZ{int(sequence_id):06d}_{zone_kind}_{approach_side}",
        target_zone_kind=zone_kind,
        target_ship_ids=(_ship_id(ships[ship_index]),),
        target_point=(float(target_point[0]), float(target_point[1])),
        target_local=(float(target_local[0]), float(target_local[1])),
        spawn_local=(float(spawn_local[0]), float(spawn_local[1])),
        approach_side=approach_side,
        approach_lane=approach_lane,
        range_to_target_m=range_to_target_m,
        planned_bearing_error_deg=float(planned_bearing_error_deg),
        intended_label=intended_label,
        convoy_heading_rad=float(frame.heading_rad),
    )


def spawn_world_from_intent(intent: AttackIntent | Mapping[str, Any], ships: Sequence[Ship]) -> tuple[float, float]:
    """Return world-space spawn position for an intent."""

    frame = ConvoyFrame.from_ships(ships)
    if isinstance(intent, AttackIntent):
        spawn_local = np.asarray(intent.spawn_local, dtype=float)
    else:
        spawn_local = np.asarray(intent["spawn_local"], dtype=float)
    spawn = frame.local_to_world(spawn_local)
    return (float(spawn[0]), float(spawn[1]))
