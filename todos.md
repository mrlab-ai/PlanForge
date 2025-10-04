# Next Intermediate Steps — Numeric PDDL Translator (checkpoint 2025-10-04)

This checklist captures only the next actionable steps (with tests) so you can resume quickly in a new session. Current status: the internal rule engine (`translate/build_model.rs`) is green, minimal rule generation in `pddl_to_prolog.rs` derives action-head atoms for single-precondition actions, and an end-to-end unit test passes. Grounding still uses a naive path; invariants/fact-groups/numeric-axioms remain partial or stubs.

## immediate next steps

1) Normalize + rule-generation parity (minimal)
- Add `src/translate/normalize.rs` with three transforms used in Python before exploration:
  - remove_free_effect_variables
  - split_duplicate_arguments
  - convert_trivial_rules (turns 0-condition rules into facts)
- Wire a `normalize()` call inside `translate_from_ast()` before pushing `model_rules`.
- Tests:
  - Unit tests in `normalize.rs` for each transform (happy + 1 edge case).
  - Extend `pddl_to_prolog.rs` tests to cover a rule with duplicated args and a trivial rule becoming a fact.

2) Broaden condition support for exploration
- Extend `cond_to_symatoms()` to accept simple `And(Atom...)`, `Atom`, and ignore `True`; keep `Not`/`Comparison` as TODO but add explicit tests to ensure they’re skipped deterministically (documented behavior).
- Tests: add cases asserting skipped constructs don’t break model building (no panics, rules unaffected).

3) Multi-atom preconditions → correct rule kinds
- For 2 atoms → emit join rule; for >2 → emit product rule (temporary heuristic). Consider reusing `translate/greedy_join.rs` later.
- Tests: new e2e test where an action with two preconditions is derived only when both facts exist.

4) Effects and action-instance tracking (exploration facts)
- Emit an explicit action-instance atom (e.g., `act_move(?x)`) on precondition satisfaction to tag reachable operator instances. Keep normal predicate effects for later stages.
- Tests: assert both the predicate fact(s) and `act_move(a1)` appear given suitable init.

5) Integrate reachability into grounding
- In `instantiate.rs`, add an optional reachability filter: compute `model = compute_model_from_ast(...)` and only ground actions whose `act_<name>(...)` are present (or fall back to naive if disabled).
- Tests: integration test using `pddl/domain.pddl` + `pddl/pfile1.pddl` validating: reachable-guarded grounding yields ≤ naive grounding and > 0 actions.

6) Regression harness scaffold (for Python parity later)
- Add a test util that runs the Rust translator on small PDDL files and snapshots key artifacts (counts of operators/facts, a few symbol names). Store golden JSON beside tests.
- Tests: create 1-2 golden snapshots from `pddl/pfile1.pddl`, ensure they stay stable across refactors.

## supporting tasks (small, quick wins)

- Fix warnings: remove unused imports/vars in `build_model.rs` and `pddl_to_prolog.rs` (clippy-clean).
- Document `build_model.rs`: brief comments for Arg, RuleSpec, and rule kinds.
- Add a cargo alias for faster test runs of translate-only modules.

## how to test (quick reminders)

- Run unit tests
  - fish shell
  - cargo test -q

- Focused tests (examples)
  - cargo test -q translate::build_model
  - cargo test -q translate::pddl_to_prolog

- Optional: run translator binary on sample PDDL
  - cargo run -q --bin translator -- pddl/domain.pddl pddl/pfile1.pddl
  - Check that `output.sas` is produced and non-empty.

## definition of done (for this checkpoint)

- [ ] normalize.rs exists; unit tests for three transforms are green.
- [ ] pddl_to_prolog emits join/product appropriately; tests for 2-precond action pass.
- [ ] action-instance atoms `act_<name>(...)` are produced and used by grounding when enabled.
- [ ] instantiate.rs has a reachability-guarded path; integration test shows reduced (or equal) operator count vs naive and > 0.
- [ ] warnings reduced or eliminated in touched files.

## notes / parking lot

- Full Python parity (normalize, greedy split, axioms, invariants, fact groups, numeric layers) remains ahead. The above steps unblock a functional exploration-guided grounding path to iterate on.
