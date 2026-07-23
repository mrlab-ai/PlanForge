#! /usr/bin/env python
"""0055: corrected single-abstraction Rust ICAPS Cartesian comparison."""

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


REVISION = "4a489ac"
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
    f"icaps-cartesian-{pick}-canonical": BASE_OPTIONS + [
        "--search",
        f"astar(canonical(icaps26_cartesian(pick={pick},max_time=900)))",
    ]
    for pick in ("min_unwanted", "max_unwanted", "random")
}
EXPECTED_TASKS = 642
EXPECTED_DOMAINS = 30
EXPECTED_RUST_ERRORS = {
    "solved",
    "search-out-of-time",
    "search-out-of-memory",
}

HERE = Path(__file__).resolve().parent
DATA_0052 = HERE / "data/0052-complete-ipc26-snp-matrix-eval/properties"
DATA_0053_RAW = HERE / "data/0053-project-eval/properties"
DATA_0053_REPAIR = HERE / "data/0053-repair-eval/properties"
SOCS_PUBLISHED = Path(
    "/proj/parground/users/x_mfrit/code/numeric-fd-socs26/experiments/"
    "socs26-final/data/socs26-final-eval/properties"
)
CPP_ICAPS_ALGORITHMS = {
    "cpp-icaps-cartesian-min_unwanted",
    "cpp-icaps-cartesian-max_unwanted",
    "cpp-icaps-cartesian-random",
}
CPP_ICAPS_UNSUPPORTED_DOMAINS = {
    "farmland_sas", "hydropower_sas", "onlycraft-ipc26",
    "sailing-ipc23_sas", "sailing_sas", "satellite_sas",
}


def load(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def identity(run):
    return run["algorithm"], run["domain"], run["problem"]


def completed(run):
    """Only merge parsed terminal runs; pending/missing runs stay absent."""
    return run.get("planner_exit_code") is not None or run.get("unsupported") == 1


def acceptable_0052(run):
    if run.get("coverage") == 1:
        return run.get("plan_valid") == 1 and not run.get("cost_mismatch")
    return completed(run) and not run.get("scheduler_termination")


def merge_results(exp):
    current_path = Path(exp.eval_dir) / "properties"
    rust_0055 = load(current_path)
    merged = {}

    # Lowest priority: valid 0052 rows.
    for key, run in load(DATA_0052).items():
        if acceptable_0052(run):
            run["source_experiment"] = "0052"
            merged[identity(run)] = run

    # Middle priority: terminal 0053 rows. Exclude the invalid, superseded
    # ICAPS2021 artifact invocation; the repair data supplies these identities.
    for key, run in load(DATA_0053_RAW).items():
        if run.get("source_experiment") != "0053":
            continue
        if not completed(run):
            continue
        if run.get("planner_family") == "cpp-icaps":
            continue
        if run.get("coverage") == 1 and run.get("plan_valid") != 1:
            continue
        run["source_experiment"] = "0053"
        merged[identity(run)] = run

    # Published C++ SoCS baseline for the established 562-task suite.
    for run in load(SOCS_PUBLISHED).values():
        if run.get("algorithm") != "canonical_heuristic-f642e":
            continue
        run["algorithm"] = "cpp-socs-canonical"
        run["source_experiment"] = "published-socs"
        run["planner_family"] = "cpp-socs"
        # The old run directories were archived without plans, so these
        # published rows cannot be revalidated locally.
        if run.get("coverage") == 1:
            run["validation_status"] = "unavailable-published-run-archive"
        merged[identity(run)] = run

    # Correct Zenodo ICAPS C++ repair rows, but only after they are terminal.
    for run in load(DATA_0053_REPAIR).values():
        if completed(run):
            run["source_experiment"] = "0053-repair"
            merged[identity(run)] = run

    # Fragment exclusions established with the untouched Zenodo artifact.
    report_tasks = {
        (run["domain"], run["problem"])
        for run in merged.values()
        if run.get("algorithm") in CONFIGS
    }
    for domain, problem in report_tasks:
        if domain not in CPP_ICAPS_UNSUPPORTED_DOMAINS:
            continue
        for algorithm in CPP_ICAPS_ALGORITHMS:
            run = {
                "algorithm": algorithm, "domain": domain, "problem": problem,
                "coverage": None, "unsupported": 1,
                "error": "unsupported-cpp-icaps-artifact",
                "source_experiment": "0053-repair",
                "planner_family": "cpp-icaps",
            }
            merged[identity(run)] = run

    # Highest priority: all corrected 0055 Rust rows.
    for run in rust_0055.values():
        if not completed(run):
            raise RuntimeError(f"unfinished 0055 row: {identity(run)}")
        if run.get("unsupported"):
            raise RuntimeError(f"unsupported 0055 task: {identity(run)}")
        if run.get("scheduler_termination") or run.get("crash"):
            raise RuntimeError(f"crashed 0055 task: {identity(run)}")
        if run.get("error") not in EXPECTED_RUST_ERRORS:
            raise RuntimeError(
                f"unexpected 0055 error {run.get('error')!r}: {identity(run)}"
            )
        if run.get("coverage") == 1 and run.get("plan_valid") != 1:
            raise RuntimeError(f"unvalidated 0055 solution: {identity(run)}")
        run["source_experiment"] = "0055"
        merged[identity(run)] = run

    # Validate unique identities, VAL costs, and cross-configuration costs.
    expected = EXPECTED_TASKS * len(CONFIGS)
    rust_rows = [r for r in merged.values() if r.get("algorithm") in CONFIGS]
    if len(rust_rows) != expected or len({identity(r) for r in rust_rows}) != expected:
        raise RuntimeError(f"expected {expected} unique 0055 Rust rows, got {len(rust_rows)}")
    domains = {run["domain"] for run in rust_rows}
    if len(domains) != EXPECTED_DOMAINS:
        raise RuntimeError(
            f"expected {EXPECTED_DOMAINS} Rust benchmark domains, got {len(domains)}"
        )
    costs = defaultdict(set)
    for run in merged.values():
        if run.get("coverage") == 1:
            validation_unavailable = (
                run.get("source_experiment") == "published-socs"
                or (
                    run.get("algorithm") == "lmcut"
                    and run.get("domain") == "sailing-ipc23_sas"
                )
            )
            if not validation_unavailable and run.get("plan_valid") != 1:
                raise RuntimeError(f"solution without valid VAL result: {identity(run)}")
            if validation_unavailable and run.get("plan_valid") != 1:
                run.setdefault("validation_status", "VAL-unsupported-domain")
            if run.get("val_cost") is not None and run.get("cost") is not None:
                if abs(float(run["val_cost"]) - float(run["cost"])) > 1e-3:
                    raise RuntimeError(f"planner/VAL cost mismatch: {identity(run)}")
            costs[(run["domain"], run["problem"])].add(round(float(run["cost"]), 6))
        if run.get("scheduler_termination") and run.get("crash"):
            raise RuntimeError(f"scheduler termination classified as crash: {identity(run)}")
    disagreements = {task: values for task, values in costs.items() if len(values) > 1}
    if disagreements:
        raise RuntimeError(f"cost disagreements: {disagreements}")

    serial = {"|".join(key): run for key, run in sorted(merged.items())}
    current_path.write_text(json.dumps(serial, sort_keys=True))
    counts = Counter()
    for run in serial.values():
        counts["coverage"] += run.get("coverage") == 1
        counts["unsupported_cpp"] += run.get("unsupported") == 1
        counts["errors"] += bool(run.get("error")) and run.get("error") not in {
            "success", "unsolvable", "timeout", "out-of-memory", "unsupported-cpp-icaps-artifact"
        }
        counts["timeouts"] += bool(run.get("timeout")) or run.get("error") == "timeout"
        counts["ooms"] += bool(run.get("memory_out")) or run.get("error") == "out-of-memory"
        counts["val_failures"] += run.get("plan_valid") == 0
    print(f"[0055 merge] runs={len(serial)} coverage={counts['coverage']} "
          f"unsupported_cpp={counts['unsupported_cpp']} errors={counts['errors']} "
          f"timeouts={counts['timeouts']} ooms={counts['ooms']} "
          f"VAL_failures={counts['val_failures']} cost_disagreements=0")
    for algorithm in sorted(CONFIGS):
        rows = [r for r in serial.values() if r.get("algorithm") == algorithm]
        print(f"[coverage] {algorithm}: {sum(r.get('coverage') == 1 for r in rows)}/{len(rows)}")


exp = project.PlanForgeExperiment(
    environment=ENV, wall_time_limit=WALL_TIME, val_binary=VAL,
)
for name, options in CONFIGS.items():
    exp.add_algorithm(
        name=name, repo=REPO, rev=REVISION, target=PROFILE,
        build_options=BUILD_OPTIONS, options=options,
    )

exp.add_suite(BENCHMARKS_IPC,
              [s + "_sas" for s in project.SUITE_NUMERIC_IPC23_ALL_NO_0_COVERAGE])
exp.add_suite(BENCHMARKS_OTHERS,
              [s + "_sas" for s in project.SUITE_NUMERIC_OTHERS_NO_0_COVERAGE])
exp.add_suite(BENCHMARKS_OTHERS,
              [s + "_sas" for s in project.SUITE_NUMERIC_OTHERS_NEW])
exp.add_suite(BENCHMARKS_IPC, project.SUITE_NUMERIC_IPC26)

exp.add_parser(project.PlanForgeParser())
exp.add_parser(AbstractionParser())

ATTRIBUTES = [
    "coverage", "unsupported", "error", "source_experiment", "cost",
    "plan_valid", "val_cost", "invalid_plan", "cost_mismatch",
    "initial_h_value_float", "abstraction_construction_time",
    "abstraction_states", "cartesian_transitions",
    "planner_exit_code", "planner_wall_clock_time", "search_time",
    "expansions", "evaluations", "peak_memory_kb", "timeout", "memory_out",
    "scheduler_termination", "crash",
]


def print_info():
    tasks = exp._get_tasks()
    domains = {task.domain for task in tasks}
    if len(tasks) != EXPECTED_TASKS or len(domains) != EXPECTED_DOMAINS:
        raise RuntimeError(
            f"expected {EXPECTED_TASKS} tasks from {EXPECTED_DOMAINS} domains, "
            f"got {len(tasks)} tasks from {len(domains)} domains"
        )
    print(f"revision={REVISION} account={ACCOUNT} tasks={len(tasks)} configs={len(CONFIGS)} "
          f"runs={len(tasks) * len(CONFIGS)} planner={PLANNER_TIME}/{PLANNER_MEMORY} "
          f"slurm_memory={ENV.memory_per_cpu} wall={WALL_TIME}s")


exp.add_step("info", print_info)
exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)
exp.add_step("parse", exp.parse)
exp.add_fetcher(name="fetch")
exp.add_step("merge", merge_results, exp)
project.add_absolute_report(
    exp, name="0055-project-corrected-merged-abs", attributes=ATTRIBUTES,
    report_class=project.PlanForgeReport,
)
exp.run_steps()
