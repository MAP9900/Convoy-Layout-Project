"""Diagnostics for decoded VAE attack-profile samples."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

from convoy_sim.attack_profiles import AttackProfile
from convoy_sim.entities import Ship
from convoy_sim.profile_audit import AuditThresholds, audit_attack_profiles
from convoy_sim.profile_outcome_audit import OutcomeAuditConfig, OutcomeAuditContext


def nearest_ship_clearance_m(u_pos: Sequence[float], ships: Sequence[Ship]) -> float:
    """Return the distance from a U-boat position to the nearest ship center."""

    if not ships:
        raise ValueError("ships must be non-empty")
    u = np.asarray(u_pos, dtype=float)
    if u.shape != (2,):
        raise ValueError("u_pos must be a 2D position")
    ship_positions = np.asarray([np.asarray(ship.position, dtype=float) for ship in ships], dtype=float)
    return float(np.min(np.linalg.norm(ship_positions - u, axis=1)))


def audit_decoded_vae_payloads(
    payloads: Sequence[Mapping[str, Any]],
    ships: Sequence[Ship],
    *,
    min_clearance_m: float = 250.0,
    rng_seed: int = 1945,
    outcome_cfg: OutcomeAuditConfig | None = None,
    static_thresholds: AuditThresholds | None = None,
) -> list[dict[str, Any]]:
    """Audit decoded VAE payloads without relying on centroid-only pass/fail.

    Decoded regular-VAE samples do not contain v4 intent metadata. This helper
    therefore treats the moving-convoy outcome audit as the primary behavioral
    diagnostic, while keeping the legacy static centroid audit under clearly
    named ``centroid_static_*`` fields for comparison.
    """

    profiles = [AttackProfile(**dict(payload)) for payload in payloads]
    static_rows = audit_attack_profiles(
        profiles,
        list(ships),
        thresholds=static_thresholds or AuditThresholds(),
    )
    static_by_id = {str(row["profile_id"]): dict(row) for row in static_rows}
    context = OutcomeAuditContext.from_ships(list(ships), cfg=outcome_cfg or OutcomeAuditConfig())

    rows: list[dict[str, Any]] = []
    for index, payload in enumerate(payloads):
        profile = profiles[index]
        profile_id = str(payload["profile_id"])
        static_row = static_by_id[profile_id]
        clearance_m = nearest_ship_clearance_m(payload["u_pos"], ships)
        outcome_row = context.audit_profile(
            profile,
            intent={"profile_first_outcome_label": True},
            rng_seed=int(rng_seed),
            profile_index=index,
        )
        rows.append(
            {
                "profile_id": profile_id,
                "name": str(payload.get("name", "")),
                "u_pos_x": float(payload["u_pos"][0]),
                "u_pos_y": float(payload["u_pos"][1]),
                "u_pos": [float(payload["u_pos"][0]), float(payload["u_pos"][1])],
                "clearance_m": float(clearance_m),
                "clearance_ok": bool(clearance_m >= float(min_clearance_m)),
                "actual_outcome_label": str(outcome_row["actual_outcome_label"]),
                "any_ship_hit": bool(outcome_row["any_ship_hit"]),
                "n_hits": int(outcome_row["n_hits"]),
                "unique_ships_hit": int(outcome_row["unique_ships_hit"]),
                "closest_any_ship_distance_m": float(outcome_row["closest_any_ship_distance_m"]),
                "closest_any_ship_id": str(outcome_row["closest_any_ship_id"]),
                "centroid_static_label": str(static_row["suggested_label"]),
                "centroid_static_flags": list(static_row["flags"]),
                "centroid_static_bearing_error_deg": float(static_row["bearing_error_deg"]),
                "passes_safety_gate": bool(clearance_m >= float(min_clearance_m)),
                "passes_gate": bool(clearance_m >= float(min_clearance_m)),
                "profile_payload": dict(payload),
                "outcome": dict(outcome_row),
                "centroid_static_audit": static_row,
            }
        )
    return rows


def summarize_decoded_vae_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return compact decoded-VAE diagnostics suitable for notebook display."""

    if not rows:
        return {
            "samples": 0,
            "actual_outcome_labels": {},
            "centroid_static_labels": {},
            "clearance_ok_rate": 0.0,
            "passes_safety_gate_rate": 0.0,
        }
    return {
        "samples": int(len(rows)),
        "actual_outcome_labels": dict(Counter(str(row["actual_outcome_label"]) for row in rows)),
        "centroid_static_labels": dict(Counter(str(row["centroid_static_label"]) for row in rows)),
        "clearance_ok_rate": float(mean(1.0 if bool(row["clearance_ok"]) else 0.0 for row in rows)),
        "passes_safety_gate_rate": float(mean(1.0 if bool(row["passes_safety_gate"]) else 0.0 for row in rows)),
        "min_clearance_m": float(min(float(row["clearance_m"]) for row in rows)),
        "any_ship_hit_rate": float(mean(1.0 if bool(row["any_ship_hit"]) else 0.0 for row in rows)),
        "mean_hits": float(mean(float(row["n_hits"]) for row in rows)),
        "mean_unique_ships_hit": float(mean(float(row["unique_ships_hit"]) for row in rows)),
        "mean_closest_any_ship_distance_m": float(
            mean(float(row["closest_any_ship_distance_m"]) for row in rows)
        ),
    }
