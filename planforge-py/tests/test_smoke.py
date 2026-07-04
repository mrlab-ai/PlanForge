import planforge


VALID = {"solved", "unsolvable", "timeout", "memory_limit"}


def test_solve_sas():
    # gbfs(ff()) solves example2 quickly; blind A* on it is very large, so bound
    # every search with max_time to keep the suite from hanging.
    r = planforge.solve(
        sas="tests/assets/numeric_sas/example2.sas",
        search="gbfs(ff())",
        max_time=60.0,
    )
    assert r.status in VALID
    assert isinstance(r.nodes_expanded, int) and r.nodes_expanded >= 0
    if r.status == "solved":
        assert r.plan is not None and len(r.plan) > 0
        assert r.cost is not None
        assert r.plan[0].name  # operators expose a name and cost
        assert r.plan_to_sas().startswith("(")


def test_solve_pddl_end_to_end():
    r = planforge.solve(
        domain="tests/assets/numeric-pddl-files/fn-counters-small_instances/domain.pddl",
        problem="tests/assets/numeric-pddl-files/fn-counters-small_instances/problem_2.pddl",
        search="gbfs(ff())",
        max_time=60.0,
    )
    assert r.status in VALID


def test_unparseable_spec_raises_specerror():
    import pytest

    # A syntactically invalid spec fails in the parser -> SpecError (a ValueError).
    with pytest.raises(planforge.SpecError):
        planforge.solve(sas_text="x", search="astar(")


def test_unknown_heuristic_raises_planforge_error():
    import pytest

    # A well-formed spec naming an unknown heuristic parses fine; the failure
    # surfaces later from heuristic construction as a PlanforgeError.
    with pytest.raises(planforge.PlanforgeError):
        planforge.solve(sas="tests/assets/numeric_sas/example2.sas", search="astar(nope())")


def test_bad_sas_raises_parseerror():
    import pytest

    with pytest.raises(planforge.ParseError):
        planforge.solve(sas_text="not a valid sas file")


def test_no_input_raises_valueerror():
    import pytest

    with pytest.raises(ValueError):
        planforge.solve()


def test_missing_file_raises_filenotfound():
    import pytest

    with pytest.raises(FileNotFoundError):
        planforge.solve(sas="tests/assets/does_not_exist.sas")


def test_task_reuse_and_exploration():
    task = planforge.Task.from_sas("tests/assets/numeric_sas/example2.sas")
    assert task.num_variables > 0
    assert len(task.variable_names) == task.num_variables
    assert len(task.operators()) == task.num_operators
    s0 = task.initial_state()
    assert isinstance(s0.values, list)
    # exploration
    succ = task.successors(s0)
    assert isinstance(succ, list)
    for op, s, cost in succ:
        assert isinstance(op.name, str)
        assert isinstance(cost, float)
        assert isinstance(task.is_goal(s), bool)
    # state value identity is stable
    assert task.initial_state() == s0
    assert hash(task.initial_state()) == hash(s0)
    # reuse the task for a full solve (no re-parse)
    r = task.solve("gbfs(ff())", max_time=60.0)
    assert r.status in {"solved", "unsolvable", "timeout", "memory_limit"}


def test_task_from_pddl():
    task = planforge.Task.from_pddl(
        "tests/assets/numeric-pddl-files/fn-counters-small_instances/domain.pddl",
        "tests/assets/numeric-pddl-files/fn-counters-small_instances/problem_2.pddl")
    assert task.num_operators > 0
    assert task.solve("gbfs(ff())", max_time=60.0).status in {"solved","unsolvable","timeout","memory_limit"}


def test_state_from_other_task_rejected():
    import pytest
    a = planforge.Task.from_sas("tests/assets/numeric_sas/example2.sas")
    b = planforge.Task.from_sas("tests/assets/numeric_sas/example5.sas")
    s = a.initial_state()
    with pytest.raises(ValueError):
        b.successors(s)


def test_search_with_python_heuristic():
    task = planforge.Task.from_pddl(
        "tests/assets/numeric-pddl-files/fn-counters-small_instances/domain.pddl",
        "tests/assets/numeric-pddl-files/fn-counters-small_instances/problem_2.pddl")
    goals = task.goals

    def goal_count(state):
        return float(sum(1 for (v, val) in goals if state.values[v] != val))

    r = task.search_with_heuristic(goal_count, max_time=60.0)
    assert r.status in {"solved", "unsolvable", "timeout", "memory_limit"}


def test_python_heuristic_error_propagates():
    import pytest

    task = planforge.Task.from_sas("tests/assets/numeric_sas/example2.sas")

    def bad(state):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        task.search_with_heuristic(bad, max_time=10.0)
