#! /usr/bin/env python
"""Combine the 0057 label-SCP collections with the matching LM-cut runs."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import project


HERE = Path(__file__).resolve().parent
SOURCE_0057 = HERE / "data/0057-project-eval"
SOURCE_LMCUT = HERE / "data/0055-project-eval"
CONFIGS_0057 = {
    "da-whole-min-interleaved-label-scp",
    "da-large-complementary-label-scp",
    "ca-whole-min-complementary-label-scp",
    "ca-desired-maxsteps-label-scp-10k-100k",
    "ca-desired-maxsteps-label-scp-100k-1m",
    "ca-whole-maxsteps-label-scp-10k-100k",
    "mixed-da-ca-label-scp",
}
ALGORITHMS = CONFIGS_0057 | {"lmcut"}
EXPECTED_TASKS = 642


def keep_0057(run):
    return run["algorithm"] in CONFIGS_0057


def keep_lmcut(run):
    return run["algorithm"] == "lmcut"


def validate_combination(exp):
    path = Path(exp.eval_dir) / "properties"
    runs = json.loads(path.read_text())
    expected = EXPECTED_TASKS * len(ALGORITHMS)
    if len(runs) != expected:
        raise RuntimeError(f"expected {expected} combined rows, got {len(runs)}")

    by_algorithm = Counter(run["algorithm"] for run in runs.values())
    expected_counts = {algorithm: EXPECTED_TASKS for algorithm in ALGORITHMS}
    if by_algorithm != expected_counts:
        raise RuntimeError(
            f"unexpected algorithm counts: {dict(sorted(by_algorithm.items()))}"
        )

    identities = {
        (run["algorithm"], run["domain"], run["problem"])
        for run in runs.values()
    }
    if len(identities) != expected:
        raise RuntimeError("combined report contains duplicate run identities")

    task_sets = {
        algorithm: {
            (run["domain"], run["problem"])
            for run in runs.values()
            if run["algorithm"] == algorithm
        }
        for algorithm in ALGORITHMS
    }
    reference_tasks = task_sets["lmcut"]
    for algorithm, tasks in task_sets.items():
        if tasks != reference_tasks:
            raise RuntimeError(
                f"{algorithm} task set differs from LM-cut: "
                f"missing={len(reference_tasks - tasks)}, "
                f"extra={len(tasks - reference_tasks)}"
            )

    costs = defaultdict(set)
    for run in runs.values():
        if run.get("coverage") == 1 and run.get("cost") is not None:
            costs[(run["domain"], run["problem"])].add(
                round(float(run["cost"]), 6)
            )
    disagreements = {
        task: values for task, values in costs.items() if len(values) > 1
    }
    if disagreements:
        raise RuntimeError(f"cross-algorithm cost disagreements: {disagreements}")

    for algorithm in sorted(ALGORITHMS):
        coverage = sum(
            run.get("coverage") == 1
            for run in runs.values()
            if run["algorithm"] == algorithm
        )
        print(f"[coverage] {algorithm}: {coverage}/{EXPECTED_TASKS}")


exp = project.PlanForgeExperiment()
exp.add_fetcher(
    SOURCE_0057,
    filter=keep_0057,
    merge=False,
    name="fetch-0057-label-scp",
)
exp.add_fetcher(
    SOURCE_LMCUT,
    filter=keep_lmcut,
    merge=True,
    name="fetch-0055-lmcut",
)
exp.add_step("validate", validate_combination, exp)

ATTRIBUTES = [
    "coverage",
    "error",
    "cost",
    "plan_valid",
    "val_cost",
    "initial_h_value_float",
    "abstraction_count",
    "abstraction_total_states",
    "abstraction_states",
    "domain_abstract_operators",
    "cartesian_transitions",
    "abstraction_construction_time",
    "scp_partitions_retained",
    "scp_partitions_evaluated",
    "scp_diversification_samples",
    "scp_table_size_kib",
    "planner_exit_code",
    "planner_wall_clock_time",
    "search_time",
    "expansions",
    "evaluations",
    "peak_memory_kb",
    "timeout",
    "memory_out",
    "scheduler_termination",
    "crash",
]

project.add_absolute_report(
    exp,
    name="0057-label-scp-with-lmcut-abs",
    attributes=ATTRIBUTES,
    report_class=project.PlanForgeReport,
)
exp.run_steps()
