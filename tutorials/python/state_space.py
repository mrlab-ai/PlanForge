#!/usr/bin/env python3
"""Enumerate a complete graph in Rust and compare goal count with exact h*."""

from __future__ import annotations

import math
from pathlib import Path

import planforge


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    fixture = repo / "tests/assets/strips-pddl-files/blocks-minimal"
    task = planforge.Task.from_pddl(
        fixture / "domain.pddl", fixture / "probBLOCKS-2-reverse.pddl"
    )

    # All three bounds are mandatory. Reaching one raises EnumerationError and
    # returns no plausible-looking partial h* array.
    graph = task.enumerate_state_space(
        max_states=100,
        max_transitions=1_000,
        max_time=10.0,
    )
    print(
        f"graph: states={graph.state_count}, transitions={graph.transition_count}, "
        f"goals={graph.goal_state_count}, dead_ends={graph.dead_end_count}, "
        f"diameter={graph.diameter:g}"
    )

    operators = task.operators()
    start, stop = graph.transition_offsets[0:2]
    print("outgoing transitions from state 0:")
    for edge in range(int(start), int(stop)):
        operator = operators[int(graph.transition_operator_ids[edge])]
        successor = int(graph.transition_successor_ids[edge])
        cost = float(graph.transition_costs[edge])
        print(f"  {operator.name} --{cost:g}--> state {successor}")

    print("state  values  goal_count  h_star  underestimation")
    for state_id, values in enumerate(graph.propositional_values):
        goal_count = sum(values[var] != value for var, value in task.goals)
        exact = float(graph.h_star[state_id])
        error = math.inf if math.isinf(exact) else exact - goal_count
        print(
            f"{state_id:>5}  {values.tolist()}  {goal_count:>10}  "
            f"{exact:>6g}  {error:>15g}"
        )

    assert float(graph.h_star[0]) == 4.0
    assert all(float(graph.h_star[state]) == 0.0 for state in graph.goal_states.nonzero()[0])

    print("h_star histogram:", graph.h_star_histogram)
    print(
        "scaling reference (recorded Stage 2 clean run): Blocks-8 had "
        "695417 states, 2094752 transitions, one goal, zero dead ends, "
        "diameter 28, and h*(initial)=18; Rust enumeration took 0.762 s "
        "(1.41 s CLI wall, 228596 KiB peak RSS including CSV output)."
    )


if __name__ == "__main__":
    main()
