"""Serialise an enumerated state space to JSON, one record per state.

Doubles as a check that the Python bindings expose what a learning or analysis
pipeline needs from a state space: for every state its label, whether it is a
goal, initial or dead-end state, its distance to the goal, and its incoming and
outgoing edges. Serialisation happens here rather than in Rust.

The distances are costs. PlanForge computes h* over the task's operator costs,
so a distance equals a number of steps only when every operator costs one.
"""

import json
import sys

import planforge


def edge_to_dict(edge):
    return {
        "source": edge.source,
        "target": edge.target,
        "operator_id": edge.operator_id,
        "operator_name": edge.operator_name,
        "cost": edge.cost,
        "source_label": edge.source_label,
        "target_label": edge.target_label,
    }


def state_to_dict(space, state):
    return {
        "index": state,
        "label": space.get_state_label(state),
        "atoms": space.get_atoms(state),
        "assignment": [list(fact) for fact in space.get_assignment(state)],
        "numeric_variables": space.get_numeric_variables(state),
        "is_goal": space.is_goal_state(state),
        "is_initial": space.is_initial_state(state),
        "is_dead_end": space.is_dead_end_state(state),
        "cost_to_goal": space.get_cost_to_goal(state),
        "outgoing": [edge_to_dict(e) for e in space.get_forward_transitions(state)],
        "incoming": [edge_to_dict(e) for e in space.get_backward_transitions(state)],
    }


def space_to_dict(space):
    return {
        "num_states": space.get_num_states(),
        "num_alive_states": space.get_num_alive_states(),
        "num_dead_end_states": space.get_num_dead_end_states(),
        "max_cost_to_goal": space.get_max_cost_to_goal(),
        "transition_count": space.transition_count,
        "states": [state_to_dict(space, s) for s in space.get_states()],
    }


def check(space, dump):
    """Everything the API reports must agree with what it hands over."""
    assert len(dump["states"]) == space.get_num_states() == len(space)

    # Each edge must be reported by both of its endpoints.
    outgoing = sum(len(s["outgoing"]) for s in dump["states"])
    incoming = sum(len(s["incoming"]) for s in dump["states"])
    assert outgoing == incoming == space.transition_count, "edge counts disagree"
    forward = {(e["source"], e["target"], e["operator_id"])
               for s in dump["states"] for e in s["outgoing"]}
    backward = {(e["source"], e["target"], e["operator_id"])
                for s in dump["states"] for e in s["incoming"]}
    assert forward == backward, "incoming and outgoing edges disagree"

    # Distance classes must match the histogram the space reports separately.
    histogram = dict(space.h_star_histogram)
    grouped = {cost: len(space.get_states_at_cost_to_goal(cost)) for cost in histogram}
    assert histogram == grouped, "distance grouping disagrees with the histogram"
    alive = sum(histogram.values())
    assert alive + space.get_num_dead_end_states() == space.get_num_states()

    # A goal state is exactly a state at distance zero, and dead ends have none.
    goals = [s["index"] for s in dump["states"] if s["is_goal"]]
    assert goals == space.get_states_at_cost_to_goal(0.0), "goals are not the states at cost 0"
    assert all(s["cost_to_goal"] is None for s in dump["states"] if s["is_dead_end"])

    # Sampling is reproducible from a seed, and respects the requested distance.
    space.set_seed(7)
    first = [space.sample_state() for _ in range(5)]
    space.set_seed(7)
    assert first == [space.sample_state() for _ in range(5)], "sampling is not seeded"
    for cost in histogram:
        drawn = space.sample_state_at_cost_to_goal(cost)
        assert space.get_cost_to_goal(drawn) == cost, "sampled the wrong distance"

    # Literals are read back from the state's own assignment.
    for state in dump["states"][:20]:
        literals = [tuple(fact) for fact in state["assignment"]]
        assert space.literal_holds(state["index"], literals)
        variable, value = literals[0]
        assert not space.literal_holds(state["index"], [(variable, value, False)])


def main():
    domain, problem = sys.argv[1], sys.argv[2]
    task = planforge.Task.from_pddl(domain, problem)
    space = task.enumerate_state_space(
        max_states=1_000_000, max_transitions=10_000_000, max_time=300.0
    )
    dump = space_to_dict(space)
    check(space, dump)

    print(f"states:      {dump['num_states']} ({dump['num_alive_states']} alive, "
          f"{dump['num_dead_end_states']} dead ends)")
    print(f"transitions: {dump['transition_count']}")
    print(f"max cost to goal: {dump['max_cost_to_goal']}")
    print(json.dumps(dump["states"][0], indent=2)[:400] + "\n...")

    with open("state_space.json", "w") as handle:
        json.dump(dump, handle, indent=2)
    print("wrote state_space.json")


if __name__ == "__main__":
    main()
