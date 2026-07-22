#! /usr/bin/env python
"""0053: corrected Rust/C++ artifact comparison, merged with valid 0052 data."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
import re
import shlex
import sys

import project
from downward import suites
from downward.cached_revision import CachedFastDownwardRevision
from downward.experiment import FastDownwardAlgorithm
from lab import tools
from lab.environments import TetralithEnvironment
from lab.experiment import Experiment, Run, get_default_data_dir
from lab.parser import Parser

LAB_FILES = Path(project.get_repo_base()) / "lab-files"
if str(LAB_FILES) not in sys.path:
    sys.path.insert(0, str(LAB_FILES))
from abstraction_parser import AbstractionParser


PLANFORGE_REVISION = "5687d0b"
ICAPS_ARTIFACT_REPO = "/proj/parground/users/x_mfrit/code/numeric-fd"
ICAPS_ARTIFACT_REVISION = "ICAPS2021"
SOCS_ARTIFACT_REPO = "/proj/parground/users/x_mfrit/code/numeric-fd-socs26"
SOCS_ARTIFACT_REVISION = "f642e715316939fafd88f8705429980138f32ade"
REPO = str(project.get_repo_base())
BENCHMARKS_IPC = os.environ["NUMERIC_BENCHMARKS_IPC2023"]
BENCHMARKS_OTHERS = os.environ["NUMERIC_BENCHMARKS_OTHERS"]
VAL = "/home/x_mfrit/bin/validate"

PLANNER_TIME = "1800s"
PLANNER_MEMORY = "8G"
EXTERNAL_WALL_TIME = 2700
SLURM_ACCOUNT = "naiss2025-5-382"
ENV = TetralithEnvironment(
    email="markus.fritzsche@liu.se",
    memory_per_cpu="10G",
    extra_options=f"#SBATCH -A {SLURM_ACCOUNT}",
    time_limit_per_task="24:00:00",
)

ALL_SUITES = [
    (BENCHMARKS_IPC, [s + "_sas" for s in project.SUITE_NUMERIC_IPC23_ALL_NO_0_COVERAGE]),
    (BENCHMARKS_OTHERS, [s + "_sas" for s in project.SUITE_NUMERIC_OTHERS_NO_0_COVERAGE]),
    (BENCHMARKS_OTHERS, [s + "_sas" for s in project.SUITE_NUMERIC_OTHERS_NEW]),
    (BENCHMARKS_IPC, project.SUITE_NUMERIC_IPC26),
]
NEW_IPC26_SUITES = [(BENCHMARKS_IPC, project.SUITE_NUMERIC_IPC26)]

BASE_RUST_OPTIONS = [
    "--restrict-task", "--compact-numeric-states",
    "--max-time", PLANNER_TIME, "--max-memory", PLANNER_MEMORY,
]
RUST_ICAPS = OrderedDict(
    (
        f"icaps-cartesian-{pick}-canonical",
        BASE_RUST_OPTIONS + [
            "--search",
            f"astar(canonical(icaps26_cartesian(pick={pick}),construction_max_time=900))",
        ],
    )
    for pick in ("min_unwanted", "max_unwanted", "random")
)

CPP_ICAPS = OrderedDict(
    (
        f"cpp-icaps-cartesian-{pick.lower()}",
        ["--search", f"astar(cegar(pick={pick}))"],
    )
    for pick in ("MIN_UNWANTED", "MAX_UNWANTED", "RANDOM")
)
CPP_SOCS_NAME = "cpp-socs-canonical"
CPP_SOCS_OPTIONS = [
    "--search",
    "astar(canonical_heuristic([domain_abstractions(multiple_domain_abstractions_cegar("
    "blacklist_trigger_percentage=0.6, total_max_time=700, "
    "flaw_treatment=max_refined_single_atom, numeric_split_strategy=STANDARD, "
    "use_progress_weighted_flaw_selection=false, "
    "use_threshold_aware_numeric_splits=false, "
    "exec_entire_plan=execute_entire_plan, max_abstraction_size=1000000, "
    "max_collection_size=10000000), combine_labels=true)]))",
]

HERE = Path(__file__).resolve().parent
DATA_0052 = HERE / "data" / "0052-complete-ipc26-snp-matrix-eval" / "properties"
RUNS_0052 = HERE / "data" / "0052-complete-ipc26-snp-matrix"
BUG_TEXT = (
    "shared abstraction construction deadline exceeded",
    "VmRSS missing for live process",
)


def load_0052():
    if not DATA_0052.is_file():
        raise RuntimeError(f"0052 properties missing: {DATA_0052}")
    with DATA_0052.open() as stream:
        return json.load(stream)


def exact_bug_reruns():
    """Return algorithm -> task keys/options for only the two fixed exit-1 bugs."""
    selected = defaultdict(set)
    options = {}
    for run in load_0052().values():
        if run.get("planner_exit_code") != 1:
            continue
        algorithm = run["algorithm"]
        if algorithm in RUST_ICAPS:
            continue
        run_err = RUNS_0052 / run["run_dir"] / "run.err"
        text = run_err.read_text(errors="replace") if run_err.is_file() else ""
        if not any(marker in text for marker in BUG_TEXT):
            continue
        selected[algorithm].add((run["domain"], run["problem"]))
        current = tuple(run["options"])
        if algorithm in options and tuple(options[algorithm]) != current:
            raise RuntimeError(f"inconsistent 0052 options for {algorithm}")
        options[algorithm] = list(current)
    count = sum(map(len, selected.values()))
    if count != 77:
        raise RuntimeError(f"expected exactly 77 fixed-bug reruns, found {count}")
    return selected, options


def task_key(task):
    return task.domain, task.problem


def val_inputs(run, task):
    is_sas = task.problem_file.suffix == ".sas"
    if not is_sas and task.domain_file:
        return "domain.pddl", "problem.pddl"
    if is_sas and task.problem_file.parent.name.endswith("_sas"):
        pddl_dir = task.problem_file.parent.parent / task.problem_file.parent.name[:-4]
        domain = pddl_dir / "domain.pddl"
        problem = pddl_dir / f"{task.problem_file.stem}.pddl"
        if domain.is_file() and problem.is_file():
            run.add_resource("val_domain", domain, "val_domain.pddl", symlink=True)
            run.add_resource("val_problem", problem, "val_problem.pddl", symlink=True)
            return "val_domain.pddl", "val_problem.pddl"
    return None


def add_val(run, task):
    inputs = val_inputs(run, task)
    if not inputs:
        run.set_property("val_enabled", False)
        return
    domain, problem = inputs
    command = (
        "if [ -f sas_plan ]; then "
        f"{shlex.quote(VAL)} {domain} {problem} sas_plan > validate.log 2>&1; "
        'echo "VAL_EXIT=$?" >> validate.log; '
        'else echo "VAL_SKIPPED=no-plan" > validate.log; fi; exit 0'
    )
    run.add_command("validate", ["bash", "-c", command], wall_time_limit=600)
    run.set_property("val_enabled", True)


class HybridRustRun(project.PlanForgeRun):
    def __init__(self, exp, algo, task):
        super().__init__(
            exp, algo, task, wall_time_limit=EXTERNAL_WALL_TIME,
            run_properties={"planner_family": "rust"}, val_binary=VAL,
        )


class CppRun(Run):
    def __init__(self, exp, algo, task, family):
        super().__init__(exp)
        is_sas = task.problem_file.suffix == ".sas"
        if is_sas:
            self.add_resource("task", task.problem_file, "task.sas", symlink=True)
            inputs = ["{task}"]
        else:
            self.add_resource("domain", task.domain_file, "domain.pddl", symlink=True)
            self.add_resource("problem", task.problem_file, "problem.pddl", symlink=True)
            inputs = ["{domain}", "{problem}"]
        driver = os.path.join(
            exp.path, algo.cached_revision.get_relative_exp_path("fast-downward.py")
        )
        if family == "cpp-icaps" and not is_sas:
            # The immutable ICAPS 2021 artifact is distributed for its original
            # translated SAS fragment. Its Python-2-era translator cannot ingest
            # the newer PDDL benchmarks, so record these without executing it.
            command = [
                "bash", "-c",
                "echo 'UNSUPPORTED_CPP_ICAPS_PDDL: artifact SAS fragment only'; exit 3",
            ]
            self.set_property("preclassified_unsupported", True)
        else:
            command = [sys.executable, driver] + algo.driver_options + inputs + algo.component_options
        self.add_command("planner", command, wall_time_limit=EXTERNAL_WALL_TIME)
        add_val(self, task)
        self.set_property("algorithm", algo.name)
        self.set_property("repo", algo.cached_revision.repo)
        self.set_property("local_revision", algo.cached_revision.local_rev)
        self.set_property("global_revision", algo.cached_revision.global_rev)
        self.set_property("build_options", algo.cached_revision.build_options)
        self.set_property("driver_options", algo.driver_options)
        self.set_property("component_options", algo.component_options)
        self.set_property("planner_family", family)
        self.set_property("max_time", PLANNER_TIME)
        self.set_property("max_memory", PLANNER_MEMORY)
        self.set_property("wall_time_limit", EXTERNAL_WALL_TIME)
        for key, value in task.properties.items():
            self.set_property(key, value)
        self.set_property("experiment_name", exp.name)
        self.set_property("id", [algo.name, task.domain, task.problem])


class HybridExperiment(Experiment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.revision_cache = os.path.join(get_default_data_dir(), "revision-cache")
        self.suite_specs = []
        self.rust_algorithms = OrderedDict()
        self.cpp_algorithms = OrderedDict()

    def add_suite(self, root, names, group):
        self.suite_specs.append((Path(root).resolve(), list(names), group))

    def tasks(self, group=None):
        result = []
        seen = set()
        for root, names, spec_group in self.suite_specs:
            if group is not None and spec_group != group:
                continue
            for task in suites.build_suite(root, names):
                key = task_key(task)
                if (spec_group, key) not in seen:
                    result.append(task)
                    seen.add((spec_group, key))
        return result

    def add_rust(self, name, options, allowed=None):
        revision = project.CachedRustPlannerRevision(
            self.revision_cache, REPO, PLANFORGE_REVISION, "release", ["-j", "6"]
        )
        algorithm = project.RustPlannerAlgorithm(name, revision, options)
        algorithm.allowed = None if allowed is None else set(allowed)
        self.rust_algorithms[name] = algorithm

    def add_cpp(self, name, repo, revision, options, family, group):
        cached = CachedFastDownwardRevision(
            self.revision_cache, repo, revision, ["release64", "-j6"]
        )
        driver = [
            "--overall-time-limit", PLANNER_TIME,
            "--overall-memory-limit", PLANNER_MEMORY,
            "--build", "release64",
        ]
        algorithm = FastDownwardAlgorithm(name, cached, driver, options)
        algorithm.family = family
        algorithm.group = group
        self.cpp_algorithms[name] = algorithm

    def revisions(self):
        algorithms = list(self.rust_algorithms.values()) + list(self.cpp_algorithms.values())
        return {algorithm.cached_revision for algorithm in algorithms}

    def build(self, **kwargs):
        for revision in self.revisions():
            revision.cache()
            self.add_resource("", revision.path, revision.get_relative_exp_path())
        all_tasks = self.tasks("all")
        for algorithm in self.rust_algorithms.values():
            for task in all_tasks:
                if algorithm.allowed is None or task_key(task) in algorithm.allowed:
                    self.add_run(HybridRustRun(self, algorithm, task))
        for algorithm in self.cpp_algorithms.values():
            for task in self.tasks(algorithm.group):
                self.add_run(CppRun(self, algorithm, task, algorithm.family))
        self.set_property("algorithms", list(self.rust_algorithms) + list(self.cpp_algorithms))
        self.set_property("slurm_account", SLURM_ACCOUNT)
        super().build(**kwargs)


UNSUPPORTED_MARKERS = (
    "UNSUPPORTED_CPP_ICAPS_PDDL",
    "Tried to use unsupported feature",
    "does not support axioms",
    "does not support conditional effects",
    "This configuration does not support",
)


def read_file(name):
    try:
        return Path(name).read_text(errors="replace")
    except OSError:
        return ""


def parse_hybrid(_content, props):
    if props.get("planner_family") == "cpp-icaps":
        text = read_file("run.log") + "\n" + read_file("run.err")
        if props.get("planner_exit_code") == 3 or any(x.lower() in text.lower() for x in UNSUPPORTED_MARKERS):
            props["unsupported"] = 1
            props["error"] = "unsupported-cpp-icaps-artifact"
            props["coverage"] = None
        else:
            props["unsupported"] = 0
    else:
        props.setdefault("unsupported", 0)
    val = read_file("validate.log")
    if "Plan valid" in val:
        props["plan_valid"] = 1
        match = re.search(r"(?:Final value|Value): ([-\d.]+)", val)
        if match:
            props["val_cost"] = float(match.group(1))
    elif re.search(r"Plan invalid|Goal not satisfied|Bad plan|Failed plans|Plan failed", val):
        props["plan_valid"] = 0
    if props.get("coverage") == 1 and props.get("plan_valid") == 0:
        props["invalid_plan"] = 1
    if (props.get("coverage") == 1 and props.get("plan_valid") == 1
            and props.get("cost") is not None and props.get("val_cost") is not None
            and abs(props["cost"] - props["val_cost"]) > 1e-3):
        props["cost_mismatch"] = 1


class HybridParser(Parser):
    def __init__(self):
        super().__init__()
        self.add_function(parse_hybrid, file="driver.log")


def parse_rust_only(content, props):
    if props.get("planner_family") == "rust":
        parser_globals = project.PlanForgeParser.__init__.__globals__
        parser_globals["_parse_planforge"](content, props)


class RustParser(Parser):
    def __init__(self):
        super().__init__()
        self.add_function(parse_rust_only, file="driver.log")


def merge_0052(exp):
    current_path = Path(exp.eval_dir) / "properties"
    current = json.loads(current_path.read_text())
    old = load_0052()
    merged = {}
    replacement_keys = {
        (run["algorithm"], run["domain"], run["problem"])
        for run in current.values() if run.get("planner_family") == "rust"
    }
    for key, run in old.items():
        identity = (run["algorithm"], run["domain"], run["problem"])
        if identity in replacement_keys:
            continue
        # Only accept solved 0052 plans after VAL validation. Keep all non-solves
        # so genuine limits and remaining errors stay visible in the report.
        if run.get("coverage") == 1 and run.get("plan_valid") != 1:
            continue
        run["source_experiment"] = "0052"
        merged[f"0052:{key}"] = run
    for key, run in current.items():
        run["source_experiment"] = "0053"
        merged[f"0053:{key}"] = run

    costs = defaultdict(set)
    members = defaultdict(list)
    for key, run in merged.items():
        if run.get("coverage") == 1 and run.get("plan_valid") == 1 and run.get("cost") is not None:
            task = (run["domain"], run["problem"])
            costs[task].add(round(float(run["cost"]), 6))
            members[task].append(key)
    disagreements = {task for task, values in costs.items() if len(values) > 1}
    for task in disagreements:
        for key in members[task]:
            merged[key]["cost_disagreement"] = 1
    current_path.write_text(json.dumps(merged, sort_keys=True))

    counts = Counter()
    for run in merged.values():
        counts["unsupported_cpp_tasks"] += int(run.get("unsupported") == 1)
        counts["val_failures"] += int(run.get("invalid_plan") == 1)
        counts["remaining_errors"] += int(bool(run.get("error")) and run.get("error") not in ("solved", "unsupported-cpp-icaps-artifact"))
    print(f"[merge] runs={len(merged)}, unsupported={counts['unsupported_cpp_tasks']}, "
          f"remaining_errors={counts['remaining_errors']}, VAL_failures={counts['val_failures']}, "
          f"cost_disagreement_tasks={len(disagreements)}")
    for domain, problem in sorted(disagreements):
        print(f"[cost-disagreement] {domain}/{problem}: {sorted(costs[(domain, problem)])}")


def print_info(exp, reruns):
    print(f"PlanForge revision: {PLANFORGE_REVISION}")
    print(f"Tetralith account: {SLURM_ACCOUNT}")
    print(f"all tasks: {len(exp.tasks('all'))}")
    print(f"new IPC26 tasks: {len(exp.tasks('new'))}")
    print(f"selective non-ICAPS bug reruns: {sum(map(len, reruns.values()))}")
    print(f"limits: planner={PLANNER_TIME}, memory={PLANNER_MEMORY}, wall={EXTERNAL_WALL_TIME}s")


reruns, rerun_options = exact_bug_reruns()
exp = HybridExperiment(environment=ENV)
for root, names in ALL_SUITES:
    exp.add_suite(root, names, "all")
for root, names in NEW_IPC26_SUITES:
    exp.add_suite(root, names, "new")
for name, options in RUST_ICAPS.items():
    exp.add_rust(name, options)
for name, allowed in sorted(reruns.items()):
    exp.add_rust(name, rerun_options[name], allowed)
for name, options in CPP_ICAPS.items():
    exp.add_cpp(name, ICAPS_ARTIFACT_REPO, ICAPS_ARTIFACT_REVISION, options, "cpp-icaps", "all")
exp.add_cpp(
    CPP_SOCS_NAME, SOCS_ARTIFACT_REPO, SOCS_ARTIFACT_REVISION,
    CPP_SOCS_OPTIONS, "cpp-socs", "new",
)

exp.add_parser(exp.EXITCODE_PARSER if hasattr(exp, "EXITCODE_PARSER") else project.ExitcodeParser())
exp.add_parser(project.TranslatorParser())
exp.add_parser(project.SingleSearchParser())
exp.add_parser(project.PlannerParser())
exp.add_parser(AbstractionParser())
exp.add_parser(RustParser())
exp.add_parser(HybridParser())

exp.add_step("info", print_info, exp, reruns)
exp.add_step("build", exp.build)
exp.add_step("start", exp.start_runs)
exp.add_step("parse", exp.parse)
exp.add_fetcher(name="fetch")
exp.add_step("merge-0052", merge_0052, exp)
exp.add_step("validate-costs", project.validate_costs, exp)
project.add_absolute_report(
    exp,
    name="0053-project-merged-abs",
    attributes=[
        "coverage", "unsupported", "error", "planner_family", "source_experiment",
        "cost", "plan_valid", "val_cost", "invalid_plan", "cost_mismatch",
        "cost_disagreement", "planner_wall_clock_time", "planner_time", "search_time",
        "expansions", "evaluations", "peak_memory_kb", "memory", "timeout",
        "memory_out", "scheduler_termination", "crash",
    ],
    report_class=project.PlanForgeReport,
)
exp.run_steps()
