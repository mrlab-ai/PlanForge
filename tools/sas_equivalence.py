#!/usr/bin/env python3
"""Whether two numeric SAS files written by planforge-translator describe the
same task, up to the order of variables, operators and axioms and the names
of derived numeric variables.

Usage: sas_equivalence.py A.sas B.sas  (exit 0: equivalent, 1: they differ)

Numeric variables are identified by structure: a regular variable by its
name, a constant by its value, a derived one by its operator and operands. A
comparison variable is identified by its comparison, any other propositional
variable by its value names. Axiom layers follow from that structure and are
not compared.
"""
import sys
from collections import Counter


def parse(path):
    lines = iter(line.rstrip("\n") for line in open(path))

    def take():
        return next(lines)

    def expect(token):
        line = take()
        assert line == token, f"{path}: expected {token!r}, got {line!r}"

    def facts():
        return [tuple(map(int, take().split())) for _ in range(int(take()))]

    def inline_facts(fields):
        count = int(fields[0])
        pairs = [(int(fields[1 + 2 * i]), int(fields[2 + 2 * i])) for i in range(count)]
        return pairs, fields[1 + 2 * count:]

    task = {}
    expect("begin_version"); take(); expect("end_version")
    expect("begin_metric"); task["metric"] = take().split(); expect("end_metric")
    task["variables"] = []
    for _ in range(int(take())):
        expect("begin_variable")
        name, layer, size = take(), take(), int(take())
        task["variables"].append((name, [take() for _ in range(size)]))
        expect("end_variable")
    take()  # numeric variable count
    expect("begin_numeric_variables")
    task["numeric"] = []
    while (line := take()) != "end_numeric_variables":
        kind, _layer, name = line.split(" ", 2)
        task["numeric"].append((kind, name))
    task["mutexes"] = []
    for _ in range(int(take())):
        expect("begin_mutex_group"); task["mutexes"].append(facts()); expect("end_mutex_group")
    expect("begin_state")
    task["state"] = [int(take()) for _ in task["variables"]]
    expect("end_state")
    expect("begin_numeric_state")
    task["numeric_state"] = [take() for _ in task["numeric"]]
    expect("end_numeric_state")
    expect("begin_goal"); task["goal"] = facts(); expect("end_goal")
    task["operators"] = []
    for _ in range(int(take())):
        expect("begin_operator")
        name = take()
        prevail = facts()
        effects = []
        for _ in range(int(take())):
            conditions, (var, pre, post) = inline_facts(take().split())
            effects.append((conditions, int(var), int(pre), int(post)))
        numeric_effects = []
        for _ in range(int(take())):
            conditions, (affected, op, var) = inline_facts(take().split())
            numeric_effects.append((conditions, int(affected), op, int(var)))
        cost = take()
        expect("end_operator")
        task["operators"].append((name, prevail, effects, numeric_effects, cost))
    task["rules"] = []
    for _ in range(int(take())):
        expect("begin_rule")
        conditions = facts()
        var, pre, post = map(int, take().split())
        expect("end_rule")
        task["rules"].append((conditions, var, pre, post))
    take()
    expect("begin_comparison_axioms")
    task["comparisons"] = {}
    while (line := take()) != "end_comparison_axioms":
        var, op, left, right = line.split()
        task["comparisons"][int(var)] = (op, int(left), int(right))
    take()
    expect("begin_numeric_axioms")
    task["definitions"] = {}
    while (line := take()) != "end_numeric_axioms":
        fields = line.split()
        if fields[1] == "linear":
            terms = [(int(fields[3 + 2 * i]), fields[4 + 2 * i]) for i in range(int(fields[2]))]
            task["definitions"][int(fields[0])] = ("linear", terms)
        else:
            task["definitions"][int(fields[0])] = (fields[1], int(fields[2]), int(fields[3]))
    expect("begin_global_constraint")
    task["global"] = tuple(map(int, take().split()))
    expect("end_global_constraint")
    expect("begin_SG")
    return task


def canonical(task):
    numeric_keys = {}

    def numeric_key(index):
        if index in numeric_keys:
            return numeric_keys[index]
        kind, name = task["numeric"][index]
        if kind == "C":
            key = ("C", task["numeric_state"][index])
        elif kind == "D":
            definition = task["definitions"][index]
            if definition[0] == "linear":
                key = ("L", tuple(sorted((numeric_key(v), c) for v, c in definition[1])))
            else:
                op, left, right = definition
                key = ("D", op, numeric_key(left), numeric_key(right))
        else:
            key = (kind, name)
        numeric_keys[index] = key
        return key

    def variable_key(var):
        if var in task["comparisons"]:
            op, left, right = task["comparisons"][var]
            return ("CMP", op, numeric_key(left), numeric_key(right))
        return ("VAR", tuple(task["variables"][var][1]))

    keys = [variable_key(var) for var in range(len(task["variables"]))]
    assert len(set(keys)) == len(keys), "two propositional variables share a key"

    def fact(var, value):
        if var in task["comparisons"]:
            return (keys[var], value)
        return (keys[var], task["variables"][var][1][value])

    def fact_set(pairs):
        return frozenset(fact(var, value) for var, value in pairs)

    def value(var, index):
        return -1 if index == -1 else fact(var, index)[1]

    operators = Counter()
    for name, prevail, effects, numeric_effects, cost in task["operators"]:
        operators[(
            name,
            fact_set(prevail),
            frozenset(
                (fact_set(conditions), keys[var], value(var, pre), value(var, post))
                for conditions, var, pre, post in effects
            ),
            frozenset(
                (fact_set(conditions), numeric_key(affected), op, numeric_key(var))
                for conditions, affected, op, var in numeric_effects
            ),
            cost,
        )] += 1
    rules = Counter(
        (fact_set(conditions), keys[var], value(var, pre), value(var, post))
        for conditions, var, pre, post in task["rules"]
    )
    regular_state = frozenset(
        (name, task["numeric_state"][index])
        for index, (kind, name) in enumerate(task["numeric"])
        if kind in ("R", "I")
    )
    direction, metric_var = task["metric"]
    return {
        "metric": (direction, None if metric_var == "-1" else numeric_key(int(metric_var))),
        "variables": frozenset(keys),
        "numeric": frozenset(numeric_key(i) for i in range(len(task["numeric"]))),
        "mutexes": Counter(fact_set(group) for group in task["mutexes"]),
        "state": fact_set(enumerate(task["state"])),
        "numeric_state": regular_state,
        "goal": fact_set(task["goal"]),
        "operators": operators,
        "rules": rules,
        "global": fact(*task["global"]),
    }


def main():
    left, right = (canonical(parse(path)) for path in sys.argv[1:3])
    differing = [part for part in left if left[part] != right[part]]
    for part in differing:
        only_left, only_right = left[part], right[part]
        if isinstance(only_left, (Counter, frozenset)):
            only_left, only_right = only_left - only_right, only_right - only_left
        print(f"{part} differs:\n  only in A: {str(only_left)[:400]}\n  only in B: {str(only_right)[:400]}")
    sys.exit(1 if differing else 0)


if __name__ == "__main__":
    main()
