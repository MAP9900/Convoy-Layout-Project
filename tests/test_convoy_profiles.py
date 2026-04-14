"""Tests for convoy profile registry entries."""

from __future__ import annotations

from collections import Counter

from convoy_sim.entities import ShipClass
from scenarios.convoy_profiles import get_convoy_layout_profile, list_convoy_layout_profiles


def test_mixed_convoy_profile_is_registered() -> None:
    assert "convoy_layout_mixed_1" in list_convoy_layout_profiles()


def test_mixed_convoy_profile_builds_heterogeneous_fleet() -> None:
    profile = get_convoy_layout_profile("convoy_layout_mixed_1")
    ships = profile.build_ships()
    counts = Counter(ship.ship_class for ship in ships)
    assert counts[ShipClass.FREIGHTER] > 0
    assert counts[ShipClass.TANKER] > 0
    assert counts[ShipClass.ESCORT] > 0
