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
