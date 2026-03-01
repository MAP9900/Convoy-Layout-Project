"""Double-oracle style loop for expanding strategy sets."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from convoy_sim.game import (
    AttackerStrategy,
    DefenderStrategy,
    estimate_payoff_matrix,
    expected_loss,
    exploitability,
)
from convoy_sim.nash import fictitious_play


def _best_response_value_defender(q: np.ndarray, m: np.ndarray) -> float:
    return float(np.min(m @ q))


def _best_response_value_attacker(p: np.ndarray, m: np.ndarray) -> float:
    return float(np.max(p @ m))


def double_oracle_loop(
    initial_defenders: list[DefenderStrategy],
    initial_attackers: list[AttackerStrategy],
    br_defender_generator: Callable[[AttackerStrategy], DefenderStrategy],
    br_attacker_generator: Callable[[DefenderStrategy], AttackerStrategy],
    eval_config: dict[str, Any],
    n_outer_iters: int,
) -> dict[str, Any]:
    """Run a coarse double-oracle loop with approximate Nash at each step."""

    defenders = list(initial_defenders)
    attackers = list(initial_attackers)
    history: list[dict[str, Any]] = []
    epsilon = float(eval_config.get("epsilon", 1e-6))

    for outer in range(1, n_outer_iters + 1):
        payoff = estimate_payoff_matrix(
            defenders=defenders,
            attackers=attackers,
            prior=eval_config.get("prior"),
            env=eval_config.get("env"),
            constraints=eval_config.get("constraints"),
            dynamics=eval_config.get("dynamics"),
            sim_params=eval_config.get("sim_params", {}),
            objective=eval_config.get("objective"),
            n_trials=int(eval_config.get("n_trials", 10)),
            rng=np.random.default_rng(int(eval_config.get("rng_seed", 0)) + outer),
            include_raw=False,
        )
        m = payoff["matrix_mean_loss"]
        nash = fictitious_play(m, n_iters=int(eval_config.get("nash_iters", 200)))
        p = nash["p"]
        q = nash["q"]
        exp = expected_loss(p, q, m)
        def_br_value = _best_response_value_defender(q, m)
        atk_br_value = _best_response_value_attacker(p, m)

        most_likely_def = defenders[int(np.argmax(p))]
        most_likely_atk = attackers[int(np.argmax(q))]
        new_def = br_defender_generator(most_likely_atk)
        new_atk = br_attacker_generator(most_likely_def)

        added_def = False
        if new_def.name not in {d.name for d in defenders} and def_br_value < exp - epsilon:
            defenders.append(new_def)
            added_def = True

        added_atk = False
        if new_atk.name not in {a.name for a in attackers} and atk_br_value > exp + epsilon:
            attackers.append(new_atk)
            added_atk = True

        history.append(
            {
                "outer_iter": outer,
                "defender_names": [d.name for d in defenders],
                "attacker_names": [a.name for a in attackers],
                "expected_loss": exp,
                "exploitability": exploitability(p, q, m),
                "added_defender": added_def,
                "added_attacker": added_atk,
            }
        )

    return {
        "defenders": [d.name for d in defenders],
        "attackers": [a.name for a in attackers],
        "history": history,
    }
