## numeric_planneRS — TODOS (focused plan to finish the Python -> Rust translator port)

Purpose
- Provide a complete, faithful Rust port of the Fast Downward "translate" pipeline (numeric planning support) so the repository contains no runtime Python and generated SAS files are semantically and textually compatible with the original translator for regression tests.

This document records a step-by-step translation checklist, current differences (concise table), and the next actionable work items prioritized for parity.

Tasks ordered by difficulty (easy → hard)

Easy (quick wins)
- [x] Remove unused imports and variables (warnings) in `pddl_parser.rs`, `to_sas.rs`, `numeric_axiom_rules.rs`.
- [x] Remove Python runtime call from CLI; keep Rust-only path.
- [x] Add fish-shell run snippet to docs.
- [ ] Remove `scripts/compute_groups.py` after parity is achieved (defer until final).

Medium
- [x] Port GroupCoverQueue and choose_groups semantics (partial-encoding) in `fact_groups.rs`.
- [x] Implement lexicographic sort of groups and group contents to approximate Python repr ordering.
- [ ] Align translation_key formatting fully with Python (negations, sentinels, exact string forms) — partial, continue.
- [ ] Tighten variable and value ordering in `to_sas` to match Python’s sequence.

Hard (priority for parity)
- [ ] Port `invariant_finder.py` semantics:
   - [ ] BalanceChecker.add_inequality_preconds (with reachable_action_params), heavy-action duplication.
   - [ ] check_balance/operator_unbalanced and candidate refinement with enqueue logic.
   - [ ] useful_groups: ensure crowding detection and parameter extraction identical to Python.
- [ ] Numeric axiom instantiation parity:
   - [ ] Full instantiation for arithmetic expressions and PNEs; stable names/order equal to Python.
   - [ ] Comparison axioms packing and variable creation ordering identical to Python.
- [ ] Derived-function canonicalization and integration.

Today (final comparison notes)
- I ran a faithful comparison between `output.sas.reference` (Python) and the current `output.sas` (Rust). Remaining differences discovered:
   - Rust produces 2 more finite-domain variables than Python (27 vs 19 numeric variables counted differently because of ordering and extra instrumentation-like entries).
   - The `begin_numeric_variables`/`numeric_axioms` section currently prints comparison ops (<=, >=, etc.) in places where Python prints arithmetic-derived names (+, -, ...); Rust normalizes comparisons into difference PNEs but some numeric_axioms still use binary comparison-format rather than explicit arithmetic parts.
   - The order of top-level sections differs (metric header and variable blocks ordering are not aligned with Python's output order).
   - The metric header in `output.sas` is hard-coded as `< 0` rather than the Python metric `< 15` and does not reference the correct metric variable id or type.
   - No instrumentation variable (`I` type) is being added in the numeric variables block while Python emits `I -1 PNE total-cost()`.
   - Derived PNE naming differs: Python uses placeholder-style names like `weight(?i)_PNE` and `current_load(?b)(?i, ?b)_PNE` while Rust currently emits concrete instantiations such as `weight(item1)` and `current_load(bot1)`. This causes semantic matches to look different textually.
   - Section ordering and variable id assignment differ causing numeric_axioms and compare-axiom variable indices to be different even when semantics align.

Next short-term todos (to add/track):
- Ensure the metric header is populated from the parsed task metric and uses the correct metric variable id and `< N` form to match Python formatting.
- Add instrumentation variable support so `I -1 PNE total-cost()` (or equivalent) appears in `begin_numeric_variables` when required.
- Canonicalize derived function names in `src/translate/derived_function_admin.rs` to produce placeholder tokens (e.g., `?i`, `?b`) so derived numeric variable names match Python's form.
- Align top-level section ordering in `src/translate/sas_writer.rs` to match Python's writer (metric, variables, numeric variables, mutex groups, numeric_axioms, operators) exactly.
- Refine numeric_axiom assembly so arithmetic ops (+,-,*,/) are printed as derived sum/product/difference tokens when appropriate (not only comparisons), and ensure constant axioms are handled identically.
- Add small debug scripts under `scripts/` for targeted comparisons:
   - `scripts/compare_section_order.py` — compare section ordering between two SAS files.
   - `scripts/compare_semantic_sas.py` — semantic comparator that normalizes numeric variable names (by canonical token) and compares semantics rather than raw ids.
   - `scripts/dump_numeric_vars.py` — print `begin_numeric_variables` parsed into a stable representation for quick diffing.

Priority ranking for next session
1. Canonicalize derived PNE naming (high) — fixes large number of textual mismatches.
2. Metric header and instrumentation var (high) — ensures the header and I-variable match reference.
3. Numeric axiom assembly printing (medium) — emit arithmetic-derived forms as Python does.
4. Section ordering and id assignment adjustments (medium) — cosmetic but important for byte-for-byte parity.
5. Add the small scripts (low) — aids fast iteration and debugging.

I added these items into the TODO list here; when you want to continue I will implement (1) and then re-run the comparison scripts to iteratively close the delta.

Acceptance checkpoints
- After each hard task, run the translator and compare against `output.sas.reference`; record deltas and iterate.
- Final: byte-for-byte parity for provided references (ordering accepted as equal only if required by tests; minor ordering differences are acceptable during porting).

Mapping: Python files → Rust work items
- `python/translate/invariant_finder.py` -> `src/translate/invariant_finder.rs` (implement fully). HIGH
- `python/translate/fact_groups.py` -> `src/translate/fact_groups.rs` (match grouping + translation_key semantics). HIGH
- `python/translate/invariants.py` -> `src/translate/invariants.rs` (verify port). MEDIUM
- `python/translate/constraints.py` -> `src/translate/constraints.rs` (verify port). MEDIUM
- `python/translate/numeric_axiom_rules.py` -> `src/translate/numeric_axiom_rules.rs` (finish instantiate flow). HIGH
- `python/translate/derived_function_admin.py` -> `src/translate/derived_function_admin.rs` (implement canonicalization). HIGH

Current differences — concise table
| Area | Python source (behavior) | Rust status | Difference / Effect | Priority |
|---|---:|---|---|---:|
| Invariant finding & grouping | `invariant_finder.py` (BalanceChecker, find_invariants, useful_groups) | placeholder deterministic grouping in `src/translate/invariant_finder.rs` | Groups differ; packing into FDR vars does not match Python translation_key; causes many SAS differences | High |
| Fact-group instantiation | `fact_groups.py` (instantiate + choose_groups + translation_key) | simplified `src/translate/fact_groups.rs` & `to_sas` usage | `translation_key` formatting and group selection (choose_groups/GroupCoverQueue semantics) differ => different mutex vars and value lists | High |
| Invariants helpers | `invariants.py` | `src/translate/invariants.rs` (ported) | Mostly ported; verify edge cases (possible_matches, instantiate mappings) and deterministic ordering | Medium |
| Constraint system | `constraints.py` | `src/translate/constraints.rs` (ported) | Port completed; verify identical enumeration order/behavior and any corner-case differences in combinatorial enumeration | Medium |
| Numeric axiom instantiation | `numeric_axiom_rules.py` & `numeric_axiom` classes | `src/translate/numeric_axiom_rules.rs` partially ported; instantiate semantics incomplete | Missing nested instantiation/parameter mapping; names/order of generated axioms may differ | High |
| Derived-function canonicalization | `derived_function_admin.py` | `src/translate/derived_function_admin.rs` (stub) | Missing full canonicalization and derived-name generation -> different derived vars and numeric init handling | High |
| Comparison-axiom variables & packing | comparison axiom creation code | partly wired in `to_sas` | Value ordering and sentinel value (`<none of those>`) semantics must match Python exactly | High |
| CLI Python helper usage | `scripts/compute_groups.py` (temporary) | translator optionally calls Python helper | Present; must be removed after faithful Rust port | Medium (remove) |

Acceptance criteria (concrete)
- `get_groups(domain, problem)` implemented in Rust returns the same translation_key as Python for representative test inputs.
- After replacing Python groups, `output.sas` for provided test domains must be byte-for-byte identical to `output.sas.reference` (or a small, explained delta that we iterate on until parity).
- No runtime Python required by the repository.

Immediate next steps (ordered by difficulty)
1) Medium: Align translation_key formatting and sorting with Python precisely (fact_groups.rs + to_sas.rs), then run/diff.
2) Hard: Implement BalanceChecker.add_inequality_preconds + heavy-action handling; port minimal operator_unbalanced to refine two-part invariants, then run/diff.
3) Hard: Extend invariant refinement to multi-part cases and finalize enqueue rules; run/diff.
4) Hard: Numeric axioms instantiation parity; run/diff.

Note about running commands in fish shell
- When running the translator in fish, use the following pattern to preserve the exit code handling used in earlier notes:

```fish
cd /home/markus/code/sas_parser; cargo run --bin translator -- translate pddl/domain.pddl pddl/pfile1.pddl > /dev/null; echo exit:$status; diff -u output.sas output.sas.reference | sed -n '1,120p' || true
```

Ordering note
- Group ordering and the relative order of variables/values may differ between the Rust port and the original Python implementation during porting. This is acceptable while we iteratively converge on semantics; the final goal is to match translation_key string contents and ordering where tests require it.

Checkpoint (short):
- Implemented GroupCoverQueue semantics in `src/translate/fact_groups.rs` and removed the runtime call to the Python helper in `src/bin/translator.rs`. Built and ran the translator; differences remain concentrated in packing/translation_key and will be the focus of the next iteration.

Progress cadence & verification
- After I implement each of the high-priority ports (invariant_finder, fact_groups, numeric instantiation, derived functions), I'll run the build and one regression check. If a change edits more than 3 files in a burst, I'll pause and post a compact checkpoint.

Notes
- I inferred some ordering and small implementation choices from repository conventions (e.g., deterministic ordering using sorted keys and stable stringify of atoms). If you prefer a different deterministic tie-breaker, tell me and I will adopt it.
