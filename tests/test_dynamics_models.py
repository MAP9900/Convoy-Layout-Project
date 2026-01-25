"""Tests for convoy temporal dynamics scaffolding."""

import math

from convoy_sim.dynamics import RouteLeg, RoutePlan, ZigZagPlan


def test_route_plan_heading_at_leg_boundaries() -> None:
    plan = RoutePlan(
        legs=[
            RouteLeg(duration_s=10.0, heading_rad=0.1),
            RouteLeg(duration_s=5.0, heading_rad=0.2),
            RouteLeg(duration_s=3.0, heading_rad=0.3),
        ]
    )

    assert math.isclose(plan.heading_at(0.0), 0.1)
    assert math.isclose(plan.heading_at(9.999), 0.1)
    assert math.isclose(plan.heading_at(10.0), 0.2)
    assert math.isclose(plan.heading_at(14.999), 0.2)
    assert math.isclose(plan.heading_at(15.0), 0.3)
    assert math.isclose(plan.heading_at(100.0), 0.3)


def test_zigzag_plan_delta_heading_bounds() -> None:
    disabled = ZigZagPlan(enabled=False, amplitude_rad=0.5, period_s=20.0)
    assert disabled.delta_heading_at(0.0) == 0.0
    assert disabled.delta_heading_at(10.0) == 0.0

    enabled = ZigZagPlan(enabled=True, amplitude_rad=0.4, period_s=12.0)
    for t in [0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 20.0]:
        assert abs(enabled.delta_heading_at(t)) <= 0.4 + 1e-9
