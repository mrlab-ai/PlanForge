#! /usr/bin/env python
"""0057: strongest domain and Cartesian label-SCP collections."""

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


REVISION = "9337c8d"
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
SCP_OPTIONS = (
    "construction_max_time=900,"
    "table_construction_max_time=540,"
    "diversify=true,samples=1000,max_orders=128,orders=diverse_orders,"
    "saturator=all,residual_sweeps=0,order_optimization_max_time=0,"
    "online=false,combine_labels=false,random_seed=1,partitioning=label"
)

DA_WHOLE_MIN = (
    "domain("
    "max_abstraction_size=10000,abstraction_generation_max_time=540,"
    "collection_strategy=standard,max_collection_size=100000,total_max_time=540,"
    "combine_labels=false,random_seed=1,"
    "flaw_kind=execute_entire_plan,flaw_treatment=min_growth_single_atom,"
    "interleave_split_directions=true)"
)
DA_LARGE_COMPLEMENTARY = (
    "domain("
    "max_abstraction_size=100000,abstraction_generation_max_time=540,"
    "collection_strategy=complementary,max_collection_size=10000000,"
    "total_max_time=540,combine_labels=false,random_seed=1)"
)
CA_WHOLE_MIN = (
    "cartesian_collection("
    "max_states=10000,variants_per_goal=10,progressive_goal_roots=true,"
    "collection_strategy=complementary,max_collection_size=100000,"
    "total_max_time=540,combine_labels=false,random_seed=1,"
    "flaw_kind=execute_entire_plan,split_selection=min_transition_growth)"
)
CA_DESIRED_MAXSTEPS_SMALL = (
    "cartesian_collection("
    "max_states=10000,variants_per_goal=10,progressive_goal_roots=true,"
    "collection_strategy=complementary,max_collection_size=100000,"
    "total_max_time=540,combine_labels=false,random_seed=1,"
    "flaw_kind=progression,refinement_direction=progression,"
    "abstract_plan=backward_shortest_path,flaw_candidates=desired_region,"
    "split_selection=max_additive_steps)"
)
CA_DESIRED_MAXSTEPS_LARGE = (
    "cartesian_collection("
    "max_states=100000,variants_per_goal=10,progressive_goal_roots=true,"
    "collection_strategy=complementary,max_collection_size=1000000,"
    "total_max_time=540,combine_labels=false,random_seed=1,"
    "flaw_kind=progression,refinement_direction=progression,"
    "abstract_plan=backward_shortest_path,flaw_candidates=desired_region,"
    "split_selection=max_additive_steps)"
)
CA_WHOLE_MAXSTEPS = (
    "cartesian_collection("
    "max_states=10000,variants_per_goal=10,progressive_goal_roots=true,"
    "collection_strategy=complementary,max_collection_size=100000,"
    "total_max_time=540,combine_labels=false,random_seed=1,"
    "flaw_kind=execute_entire_plan,abstract_plan=backward_shortest_path,"
    "flaw_candidates=general,split_selection=max_additive_steps)"
)
DA_MIXED = (
    "domain("
    "max_abstraction_size=10000,abstraction_generation_max_time=300,"
    "collection_strategy=standard,max_collection_size=100000,total_max_time=300,"
    "combine_labels=false,random_seed=1,"
    "flaw_kind=execute_entire_plan,flaw_treatment=min_growth_single_atom)"
)
CA_MIXED = (
    "cartesian_collection("
    "max_states=10000,variants_per_goal=10,progressive_goal_roots=true,"
    "collection_strategy=complementary,max_collection_size=100000,"
    "total_max_time=300,combine_labels=false,random_seed=1,"
    "flaw_kind=progression,abstract_plan=backward_shortest_path,"
    "flaw_candidates=desired_region,split_selection=max_additive_steps)"
)


def label_scp(*sources, options=SCP_OPTIONS):
    return f"astar(scp({','.join(sources)},{options}))"


SEARCHES = {
    "da-whole-min-interleaved-label-scp": label_scp(DA_WHOLE_MIN),
    "da-large-complementary-label-scp": label_scp(DA_LARGE_COMPLEMENTARY),
    "ca-whole-min-complementary-label-scp": label_scp(CA_WHOLE_MIN),
    "ca-desired-maxsteps-label-scp-10k-100k": label_scp(
        CA_DESIRED_MAXSTEPS_SMALL
    ),
    "ca-desired-maxsteps-label-scp-100k-1m": label_scp(
        CA_DESIRED_MAXSTEPS_LARGE
    ),
    "ca-whole-maxsteps-label-scp-10k-100k": label_scp(CA_WHOLE_MAXSTEPS),
    "mixed-da-ca-label-scp": label_scp(
        DA_MIXED,
        CA_MIXED,
        options=SCP_OPTIONS.replace(
            "table_construction_max_time=540",
            "table_construction_max_time=300",
        ),
    ),
}
CONFIGS = {
    name: BASE_OPTIONS + ["--search", search]
    for name, search in SEARCHES.items()
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
        unexplained = run.get("unexplained_errors", [])
        expected_memory_warnings = [
            error
            for error in unexplained
            if run.get("error") == "search-out-of-memory"
            and "memory limit threshold reached" in error
            and "releasing memory padding" in error
        ]
        remaining_unexplained = [
            error for error in unexplained
            if error not in expected_memory_warnings
        ]
        if remaining_unexplained:
            raise RuntimeError(
                f"unexplained run errors {remaining_unexplained}: {identity}"
            )
        if unexplained:
            run.pop("unexplained_errors", None)
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
        if run.get("coverage") != 1:
            continue
        val_unsupported = (
            run["domain"] == "sailing-ipc23_sas"
            and run.get("plan_valid") is None
            and run.get("val_cost") is None
        )
        if run.get("plan_valid") != 1 and not val_unsupported:
            raise RuntimeError(f"unvalidated solution: {identity}")
        if run.get("cost") is None:
            raise RuntimeError(f"solution without cost: {identity}")
        if val_unsupported:
            run["validation_status"] = "VAL-unsupported-domain"
        elif run.get("val_cost") is None:
            raise RuntimeError(f"validated solution without VAL cost: {identity}")
        elif abs(float(run["cost"]) - float(run["val_cost"])) > 1e-3:
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
    path.write_text(json.dumps(runs, sort_keys=True))
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


def print_info():
    tasks = exp._get_tasks()
    domains = {task.domain for task in tasks}
    if len(tasks) != EXPECTED_TASKS or len(domains) != EXPECTED_DOMAINS:
        raise RuntimeError(
            f"expected {EXPECTED_TASKS} tasks from {EXPECTED_DOMAINS} domains, "
            f"got {len(tasks)} tasks from {len(domains)} domains"
        )
    print(
        f"revision={REVISION} account={ACCOUNT} tasks={len(tasks)} "
        f"configs={len(CONFIGS)} runs={len(tasks) * len(CONFIGS)} "
        f"planner={PLANNER_TIME}/{PLANNER_MEMORY} "
        f"slurm_memory={ENV.memory_per_cpu} wall={WALL_TIME}s"
    )
    for name, search in SEARCHES.items():
        if "partitioning=label" not in search:
            raise RuntimeError(f"{name} is not label SCP")
        if "canonical(" in search or "partitioning=region" in search:
            raise RuntimeError(f"{name} contains a forbidden combinator")
        print(f"[config] {name}: {search}")


exp.add_step("info", print_info)
exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)
exp.add_step("parse", exp.parse)
exp.add_fetcher(name="fetch")
exp.add_step("validate", validate_results, exp)
project.add_absolute_report(
    exp,
    name="0057-project-label-scp-collections-abs",
    attributes=ATTRIBUTES,
    report_class=project.PlanForgeReport,
)
exp.run_steps()
