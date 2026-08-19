"""Serialise a grounded task to JSON, and count ground atoms and actions.

Doubles as a check that the Python bindings expose everything needed to
reconstruct the grounded task: every field below is read off the API, and the
serialisation is done here rather than in Rust.
"""

import json
import sys

import planforge


def atom_to_dict(atom):
    return {
        "variable": atom.variable,
        "value": atom.value,
        "name": atom.name,
        "predicate": atom.predicate,
        "arguments": atom.arguments,
        "negated": atom.negated,
    }


def effect_to_dict(effect):
    return {
        "variable": effect.variable,
        "value": effect.value,
        "precondition_value": effect.precondition_value,
        "conditions": [list(fact) for fact in effect.conditions],
    }


def numeric_effect_to_dict(effect):
    return {
        "affected_variable": effect.affected_variable,
        "operation": effect.operation,
        "source_variable": effect.source_variable,
        "conditional": effect.conditional,
        "conditions": [list(fact) for fact in effect.conditions],
    }


def operator_to_dict(operator):
    return {
        "id": operator.id,
        "name": operator.name,
        "action": operator.action,
        "arguments": operator.arguments,
        "cost": operator.cost,
        "preconditions": [list(fact) for fact in operator.preconditions],
        "effects": [effect_to_dict(e) for e in operator.effects],
        "numeric_effects": [numeric_effect_to_dict(e) for e in operator.numeric_effects],
    }


def task_to_dict(task):
    return {
        "variables": [
            {"index": index, "name": name, "domain_size": size}
            for index, (name, size) in enumerate(
                zip(task.variable_names, task.variable_domain_sizes)
            )
        ],
        "numeric_variables": [
            {"index": index, "name": name, "type": kind}
            for index, (name, kind) in enumerate(
                zip(task.numeric_variable_names, task.numeric_variable_types)
            )
        ],
        "goals": [list(fact) for fact in task.goals],
        "uses_metric": task.metric,
        "atoms": [atom_to_dict(a) for a in task.atoms()],
        "operators": [operator_to_dict(o) for o in task.operators()],
    }


def main():
    domain, problem = sys.argv[1], sys.argv[2]
    task = planforge.Task.from_pddl(domain, problem)
    dump = task_to_dict(task)

    # The counts the API reports must match what it actually hands over.
    assert len(dump["atoms"]) == task.num_atoms, "atom count disagrees with num_atoms"
    assert len(dump["operators"]) == task.num_operators, "operator count disagrees"
    assert task.num_atoms == sum(task.variable_domain_sizes), "atoms are variable values"

    print(f"ground atoms:   {len(dump['atoms'])}")
    print(f"ground actions: {len(dump['operators'])}")
    print(json.dumps(dump, indent=2)[:400] + "\n...")

    with open("task.json", "w") as handle:
        json.dump(dump, handle, indent=2)
    print("wrote task.json")


if __name__ == "__main__":
    main()
