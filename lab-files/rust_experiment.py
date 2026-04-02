import logging
import os
from collections import OrderedDict, defaultdict
from pathlib import Path

from downward import suites
from downward.parsers.anytime_search_parser import AnytimeSearchParser
from downward.parsers.exitcode_parser import ExitcodeParser
from downward.parsers.planner_parser import PlannerParser
from downward.parsers.single_search_parser import SingleSearchParser
from downward.parsers.translator_parser import TranslatorParser

from lab.experiment import Experiment, Run, get_default_data_dir

from cached_rust_revision import CachedRustPlannerRevision


class RustPlannerAlgorithm:
    """
    A Rust planner configuration:
    - revision
    - CLI options
    """

    def __init__(self, name, cached_revision, options, binary_name="planners"):
        self.name = name
        self.cached_revision = cached_revision
        self.options = options
        self.binary_name = binary_name

    def __eq__(self, other):
        return (
            self.cached_revision == other.cached_revision
            and self.options == other.options
            and self.binary_name == other.binary_name
        )


class RustPlannerRun(Run):
    """
    A single run of the Rust planner on one task.
    """

    def __init__(self, exp, algo, task):
        super().__init__(exp)

        # Add input files
        if task.domain_file is None:
            logging.critical("Rust planner requires PDDL domain file.")
        else:
            self.add_resource("domain", task.domain_file, "domain.pddl", symlink=True)
            self.add_resource("problem", task.problem_file, "problem.pddl", symlink=True)

        binary = os.path.join(
            exp.path,
            algo.cached_revision.get_binary_path(algo.binary_name),
        )

        cmd = (
            [binary]
            + algo.options
            + ["{domain}", "{problem}"]
        )

        self.add_command("planner", cmd)

        self._set_properties(algo, task)

    def _set_properties(self, algo, task):
        self.set_property("algorithm", algo.name)
        self.set_property("repo", algo.cached_revision.repo)
        self.set_property("local_revision", algo.cached_revision.local_rev)
        self.set_property("global_revision", algo.cached_revision.global_rev)
        self.set_property("build_options", algo.cached_revision.build_options)
        self.set_property("options", algo.options)

        for key, value in task.properties.items():
            self.set_property(key, value)

        self.set_property("experiment_name", self.experiment.name)
        self.set_property("id", [algo.name, task.domain, task.problem])


class RustPlannerExperiment(Experiment):
    """
    Experiment framework for Rust planners (Fast Downward style).
    """

    # Reuse Fast Downward parsers
    EXITCODE_PARSER = ExitcodeParser()
    TRANSLATOR_PARSER = TranslatorParser()
    SINGLE_SEARCH_PARSER = SingleSearchParser()
    ANYTIME_SEARCH_PARSER = AnytimeSearchParser()
    PLANNER_PARSER = PlannerParser()

    def __init__(self, path=None, environment=None, revision_cache=None):
        super().__init__(path=path, environment=environment)

        self.revision_cache = revision_cache or os.path.join(
            get_default_data_dir(), "revision-cache"
        )

        self._suites = defaultdict(list)
        self._algorithms = OrderedDict()

    # ---------- Suites ----------

    def add_suite(self, benchmarks_dir, suite):
        if isinstance(suite, str):
            suite = [suite]

        benchmarks_dir = Path(benchmarks_dir).resolve()
        if not benchmarks_dir.is_dir():
            logging.critical(f"Benchmarks directory {benchmarks_dir} not found.")

        self._suites[benchmarks_dir].extend(suite)

    def _get_tasks(self):
        tasks = []
        for benchmarks_dir, suite in self._suites.items():
            tasks.extend(suites.build_suite(benchmarks_dir, suite))
        return tasks

    def add_algorithm(
        self,
        name,
        repo,
        rev,
        options,
        build_options=None,
        target="release",
        binary_name="planners",
    ):
        if not isinstance(name, str):
            logging.critical(f"Algorithm name must be string: {name}")

        if name in self._algorithms:
            logging.critical(f"Duplicate algorithm name: {name}")

        cached_rev = CachedRustPlannerRevision(
            self.revision_cache, repo, rev, target, build_options
        )

        algo = RustPlannerAlgorithm(
            name, cached_rev, options, binary_name
        )

        for existing in self._algorithms.values():
            if algo == existing:
                logging.critical(
                    f"Algorithms {existing.name} and {name} are identical."
                )

        self._algorithms[name] = algo

    def build(self, **kwargs):
        if not self._algorithms:
            logging.critical("You must add at least one algorithm.")

        serialized_suites = {
            str(benchmarks_dir): [str(problem) for problem in benchmarks]
            for benchmarks_dir, benchmarks in self._suites.items()
        }

        self.set_property("suite", serialized_suites)
        self.set_property("algorithms", list(self._algorithms.keys()))

        self._cache_revisions()
        self._add_code()
        self._add_runs()

        super().build(**kwargs)

    def _get_unique_cached_revisions(self):
        return set(algo.cached_revision for algo in self._algorithms.values())

    def _cache_revisions(self):
        for cached_rev in self._get_unique_cached_revisions():
            cached_rev.cache()

    def _add_code(self):
        for cached_rev in self._get_unique_cached_revisions():
            dest_path = cached_rev.get_relative_exp_path()
            self.add_resource("", cached_rev.path, dest_path)

    def _add_runs(self):
        tasks = self._get_tasks()
        for algo in self._algorithms.values():
            for task in tasks:
                self.add_run(RustPlannerRun(self, algo, task))