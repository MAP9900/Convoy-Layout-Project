from __future__ import annotations

import numpy as np

from convoy_sim.target_zones import (
    ConvoyFrame,
    build_curated_attack_intents,
    convoy_frame_and_envelope,
    sample_random_attack_intent,
    spawn_world_from_intent,
)
from scenarios.convoy_profiles import get_convoy_layout_profile


def _ships() -> list:
    return get_convoy_layout_profile("convoy_layout_1").build_ships()


def test_convoy_frame_roundtrips_world_points() -> None:
    ships = _ships()
    frame = ConvoyFrame.from_ships(ships)
    point = np.asarray(ships[0].position, dtype=float)

    local = frame.world_to_local(point)
    world = frame.local_to_world(local)

    assert np.allclose(world, point)


def test_curated_attack_intents_include_target_and_spawn() -> None:
    ships = _ships()
    intents = build_curated_attack_intents(ships, count=8)

    assert len(intents) == 8
    assert {intent.approach_side for intent in intents}
    assert all(intent.target_ship_ids for intent in intents)
    assert all(intent.range_to_target_m > 0.0 for intent in intents)

    spawn = spawn_world_from_intent(intents[0], ships)
    assert len(spawn) == 2


def test_random_attack_intent_carries_requested_label() -> None:
    ships = _ships()
    rng = np.random.default_rng(1945)

    intent = sample_random_attack_intent(
        ships,
        rng=rng,
        sequence_id=1,
        intended_label="credible_near_miss",
    )

    assert intent.intended_label == "credible_near_miss"
    assert abs(intent.planned_bearing_error_deg) >= 8.4
    assert intent.target_zone_id.startswith("TZ")


def test_random_attack_spawn_is_outside_convoy_access_envelope() -> None:
    ships = _ships()
    frame, envelope, _local_positions = convoy_frame_and_envelope(ships)
    rng = np.random.default_rng(1945)

    for sequence_id in range(1, 40):
        intent = sample_random_attack_intent(
            ships,
            rng=rng,
            sequence_id=sequence_id,
            intended_label="credible_hit_threat",
        )
        spawn_world = spawn_world_from_intent(intent, ships)
        spawn_local = frame.world_to_local(spawn_world)
        outside_x = spawn_local[0] < envelope.min_x or spawn_local[0] > envelope.max_x
        outside_y = spawn_local[1] < envelope.min_y or spawn_local[1] > envelope.max_y
        assert outside_x or outside_y
