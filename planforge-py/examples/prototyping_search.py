#!/usr/bin/env python3
"""Python-driven A* and h_max over PlanForge's Rust task/state primitives."""

from __future__ import annotations

import heapq
import itertools
import math
from pathlib import Path

import planforge


Fact = tuple[int, int]


def h_max(task: planforge.Task, state: planforge.State, operators: list) -> float:
    """Delete-relaxation h_max for finite-domain operator effects."""
    fact_cost: dict[Fact, float] = {
        (variable, value): 0.0 for variable, value in enumerate(state.values)
    }

    changed = True
    while changed:
        changed = False
        for operator in operators:
            for effect in operator.effects:
                requirements = list(operator.preconditions) + list(effect.conditions)
                if effect.precondition_value is not None:
                    requirements.append((effect.variable, effect.precondition_value))
                if any(fact not in fact_cost for fact in requirements):
                    continue
                precondition_cost = max(
                    (fact_cost[fact] for fact in requirements), default=0.0
                )
                candidate = precondition_cost + operator.cost
                achieved = (effect.variable, effect.value)
                if candidate < fact_cost.get(achieved, math.inf):
                    fact_cost[achieved] = candidate
                    changed = True

    return max((fact_cost.get(goal, math.inf) for goal in task.goals), default=0.0)


def astar(task: planforge.Task) -> tuple[list[str], float, int]:
    """A small eager A* whose expansion loop is entirely Python."""
    operators = task.operators()
    initial = task.initial_state()
    initial_h = h_max(task, initial, operators)
    if math.isinf(initial_h):
        raise RuntimeError("h_max proves the initial state is a dead end")

    counter = itertools.count()
    open_list = [(initial_h, 0.0, next(counter), initial)]
    best_g = {initial: 0.0}
    parent: dict[planforge.State, tuple[planforge.State, str]] = {}
    expanded = 0

    while open_list:
        _f, g_value, _tie, state = heapq.heappop(open_list)
        if g_value != best_g.get(state):
            continue
        if task.is_goal(state):
            plan = []
            cursor = state
            while cursor in parent:
                cursor, operator_name = parent[cursor]
                plan.append(operator_name)
            plan.reverse()
            return plan, g_value, expanded

        expanded += 1
        for operator in task.applicable_operators(state):
            successor, transition_cost = task.apply_with_cost(state, operator)
            successor_g = g_value + transition_cost
            if successor_g >= best_g.get(successor, math.inf):
                continue
            successor_h = h_max(task, successor, operators)
            if math.isinf(successor_h):
                continue
            best_g[successor] = successor_g
            parent[successor] = (state, operator.name)
            heapq.heappush(
                open_list,
                (successor_g + successor_h, successor_g, next(counter), successor),
            )

    raise RuntimeError("task is unsolvable")


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    fixture = repo / "tests/assets/strips-pddl-files/blocks-minimal"
    task = planforge.Task.from_pddl(
        fixture / "domain.pddl", fixture / "probBLOCKS-2-reverse.pddl"
    )
    plan, cost, expanded = astar(task)
    assert cost == 4.0
    assert task.is_goal(
        # Replay through the same public primitives as a final API check.
        _replay(task, plan)
    )
    print(f"solved with cost={cost:g}, expanded={expanded}, plan_length={len(plan)}")
    for step, operator in enumerate(plan, start=1):
        print(f"{step}: {operator}")


def _replay(task: planforge.Task, plan: list[str]) -> planforge.State:
    state = task.initial_state()
    for name in plan:
        matching = [op for op in task.applicable_operators(state) if op.name == name]
        if len(matching) != 1:
            raise RuntimeError(f"expected one applicable operator named {name!r}")
        state = task.apply(state, matching[0])
    return state


if __name__ == "__main__":
    main()
