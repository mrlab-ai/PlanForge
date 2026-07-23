#! /usr/bin/env python
"""0056: native single domain and Cartesian abstractions."""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import project
from lab.environments import TetralithEnvironment

LAB_FILES = Path(project.get_repo_base()) / "lab-files"
if str(LAB_FILES) not in sys.path:
    sys.path.insert(0, str(LAB_FILES))
from abstraction_parser import AbstractionParser


REVISION = "8f569e5"
REPO = project.get_repo_base()
PROFILE = "release"
BUILD_OPTIONS = ["-j", "6"]
ACCOUNT = "naiss2025-5-382"
PLANNER_TIME = "1800s"
PLANNER_MEMORY = "8G"
WALL_TIME = 2700
VAL = "/home/x_mfrit/bin/validate"
BENCHMARKS_IPC = os.environ["NUMERIC_BENCHMARKS_IPC2023"]
BENCHMARKS_OTHERS = os.environ["NUMERIC_BENCHMARKS_OTHERS"]
MAX_STATES = 18_446_744_073_709_551_615

ENV = TetralithEnvironment(
    email="markus.fritzsche@liu.se",
    memory_per_cpu="10G",
    extra_options=f"#SBATCH -A {ACCOUNT}",
    time_limit_per_task="24:00:00",
)

BASE_OPTIONS = [
    "--restrict-task",
    "--compact-numeric-states",
    "--max-time", PLANNER_TIME,
    "--max-memory", PLANNER_MEMORY,
]
CONFIGS = {
    "single-domain-native": BASE_OPTIONS + [
        "--search",
        "astar(domain_abstraction("
        f"max_abstraction_size={MAX_STATES},max_time=900,"
        "combine_labels=false,random_seed=1,"
        "flaw_treatment=min_growth_single_atom,"
        "flaw_kind=execute_entire_plan))",
    ],
    "single-cartesian-native": BASE_OPTIONS + [
        "--search",
        "astar(canonical(cartesian("
        f"max_states={MAX_STATES},max_time=900,"
        "combine_labels=false,random_seed=1,"
        "flaw_kind=execute_entire_plan)))",
    ],
}
EXPECTED_TASKS = 642
EXPECTED_DOMAINS = 30
EXPECTED_ERRORS = {
    "solved",
    "search-out-of-time",
    "search-out-of-memory",
}
REFERENCE_PROPERTIES = (
    Path(__file__).resolve().parent
    / "data/0055-project-eval/properties"
)


def validate_results(exp):
    path = Path(exp.eval_dir) / "properties"
    runs = json.loads(path.read_text())
    expected = EXPECTED_TASKS * len(CONFIGS)
    if len(runs) != expected:
        raise RuntimeError(f"expected {expected} rows, got {len(runs)}")
    identities = {
        (run["algorithm"], run["domain"], run["problem"])
        for run in runs.values()
    }
    if len(identities) != expected:
        raise RuntimeError("duplicate experiment identities")
    domains = {run["domain"] for run in runs.values()}
    if len(domains) != EXPECTED_DOMAINS:
        raise RuntimeError(
            f"expected {EXPECTED_DOMAINS} domains, got {len(domains)}"
        )

    costs = defaultdict(set)
    if REFERENCE_PROPERTIES.is_file():
        for run in json.loads(REFERENCE_PROPERTIES.read_text()).values():
            if run.get("coverage") == 1 and run.get("cost") is not None:
                costs[(run["domain"], run["problem"])].add(
                    round(float(run["cost"]), 6)
                )

    counts = Counter()
    for run in runs.values():
        identity = (run["algorithm"], run["domain"], run["problem"])
        if run.get("planner_exit_code") is None:
            raise RuntimeError(f"unfinished run: {identity}")
        if run.get("unsupported"):
            raise RuntimeError(f"unsupported task: {identity}")
        if run.get("scheduler_termination") or run.get("crash"):
            raise RuntimeError(f"crashed run: {identity}")
        if run.get("error") not in EXPECTED_ERRORS:
            raise RuntimeError(
                f"unexpected error {run.get('error')!r}: {identity}"
            )
        if run.get("coverage") == 1:
            if run.get("plan_valid") != 1:
                raise RuntimeError(f"unvalidated solution: {identity}")
            if run.get("cost") is None or run.get("val_cost") is None:
                raise RuntimeError(f"solution without cost: {identity}")
            if abs(float(run["cost"]) - float(run["val_cost"])) > 1e-3:
                raise RuntimeError(f"planner/VAL cost mismatch: {identity}")
            costs[(run["domain"], run["problem"])].add(
                round(float(run["cost"]), 6)
            )
            counts[run["algorithm"]] += 1

    disagreements = {
        task: values for task, values in costs.items() if len(values) > 1
    }
    if disagreements:
        raise RuntimeError(f"cost disagreements: {disagreements}")
    for algorithm in sorted(CONFIGS):
        print(f"[coverage] {algorithm}: {counts[algorithm]}/{EXPECTED_TASKS}")


exp = project.PlanForgeExperiment(
    environment=ENV,
    wall_time_limit=WALL_TIME,
    val_binary=VAL,
)
for name, options in CONFIGS.items():
    exp.add_algorithm(
        name=name,
        repo=REPO,
        rev=REVISION,
        target=PROFILE,
        build_options=BUILD_OPTIONS,
        options=options,
    )

exp.add_suite(
    BENCHMARKS_IPC,
    [s + "_sas" for s in project.SUITE_NUMERIC_IPC23_ALL_NO_0_COVERAGE],
)
exp.add_suite(
    BENCHMARKS_OTHERS,
    [s + "_sas" for s in project.SUITE_NUMERIC_OTHERS_NO_0_COVERAGE],
)
exp.add_suite(
    BENCHMARKS_OTHERS,
    [s + "_sas" for s in project.SUITE_NUMERIC_OTHERS_NEW],
)
exp.add_suite(BENCHMARKS_IPC, project.SUITE_NUMERIC_IPC26)

exp.add_parser(project.PlanForgeParser())
exp.add_parser(AbstractionParser())

ATTRIBUTES = [
    "coverage",
    "error",
    "cost",
    "plan_valid",
    "val_cost",
    "initial_h_value_float",
    "abstraction_construction_time",
    "abstraction_states",
    "cartesian_transitions",
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


def print_info():
    tasks = exp._get_tasks()
    domains = {task.domain for task in tasks}
    if len(tasks) != EXPECTED_TASKS or len(domains) != EXPECTED_DOMAINS:
        raise RuntimeError(
            f"expected {EXPECTED_TASKS} tasks from {EXPECTED_DOMAINS} domains, "
            f"got {len(tasks)} tasks from {len(domains)} domains"
        )
    print(
        f"revision={REVISION} tasks={len(tasks)} configs={len(CONFIGS)} "
        f"runs={len(tasks) * len(CONFIGS)} "
        f"planner={PLANNER_TIME}/{PLANNER_MEMORY}"
    )


exp.add_step("info", print_info)
exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)
exp.add_step("parse", exp.parse)
exp.add_fetcher(name="fetch")
exp.add_step("validate", validate_results, exp)
project.add_absolute_report(
    exp,
    name="0056-project-native-single-abs",
    attributes=ATTRIBUTES,
    report_class=project.PlanForgeReport,
)
exp.run_steps()
