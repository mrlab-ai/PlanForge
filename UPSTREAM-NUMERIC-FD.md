# Upstream bug report draft: numeric Fast Downward

**Status: DRAFT. Not submitted. A human must review this before it goes anywhere.**

This is a working document, not a filed report. Every claim below was checked
against a local checkout of the fork this project was ported from
(`/home/markus/code/numeric-fd2`, HEAD `faff88f`, 2024-12-17, Daniel Gnad) and
against current mainline Fast Downward (`/home/markus/code/downward`, HEAD
`9c09645`, 2024-08-16) where mainline is the relevant comparison. Line numbers
are the ones in that numeric-fd2 checkout and will drift.

Several bugs we fixed here do **not** apply upstream: they were introduced by the
Rust port and have no counterpart in the Python or C++ original. They are listed
anyway, under "Not applicable", because a report that includes them would be
wrong and a report that silently drops them invites someone to re-check them.

Verdict summary:

| # | Issue | Present in numeric-fd2? |
|---|---|---|
| 1 | Conditional numeric effects dropped | **CONFIRMED**, in two independent places |
| 2 | Invariance analysis depends on set iteration order | **CONFIRMED** (mainline fixed it; the fork has the pre-fix code) |
| 3 | Invariant-generation budget measured on the wall clock | **CONFIRMED** |
| 4 | `negate_literal` falling back to the un-negated literal | Not applicable — Rust port artefact |
| 5 | `fluent_facts` holding predicate names instead of reachable atoms | Not applicable — Rust port artefact |
| 6 | `(:derived ...)` parsed as PDDL 2.1 `(:axiom :vars :context)` | Not applicable — Rust port artefact |
| 7 | Derived goal restated by the rule proving the opposite fact | Not applicable — the code carrying it is ours |

---

## 1. Conditional numeric effects are read and then discarded (CONFIRMED)

### Where

* `src/preprocess/operator.cc:43-57` — the reader.
* `src/preprocess/operator.cc:174-184` — the writer, which faithfully writes an
  always-empty condition list.
* `src/search/state_registry.cc:133` and `:176` — `get_numeric_successor`, which applies
  every assignment effect unconditionally.

### What is wrong

The preprocessor's operator reader parses the effect conditions of a numeric
assignment effect into `ecs` and then never uses it:

```cpp
    in >> count; // number of assignment effects
    for (int i = 0; i < count; i++) {
        int eff_conds;
        vector<EffCond> ecs;
        in >> eff_conds;
        for (int j = 0; j < eff_conds; j++) {
            int var, value;
            in >> var >> value;
            ecs.push_back(EffCond(variables[var], value));   // built...
        }
        int af_var, ex_var;
        foperator operato;
        in >> af_var >> operato >> ex_var >> ws;
        assign_effects.push_back(
            NumericEffect(numeric_variables[af_var], operato,
                          numeric_variables[ex_var]));       // ...and never passed
    }
```

`Operator::NumericEffect` has a conditional constructor
(`NumericEffect(NumericVariable*, vector<EffCond>, foperator, NumericVariable*)`,
`operator.h:47-52`) which sets `is_conditional_effect = true`. It is never
called. The propositional `pre_post` loop 15 lines above does the same thing
correctly (`operator.cc:37`), which is what makes this look like an oversight
rather than a decision.

`Operator::generate_cpp_input` then writes `neff.effect_conds.size()`, which is
always `0`, so an assignment effect that arrived as

```
1 <var> <val> <affected> <op> <operand>
```

leaves the preprocessor as

```
0 <affected> <op> <operand>
```

The guard is gone and the effect is now unconditional. Nothing downstream
notices:

* the search-side reader does support the conditions —
  `AssignEffect::AssignEffect(istream&)` (`global_operator.cc:42-55`) reads them
  and sets `is_conditional_effect` — so the file format is not the limitation;
* but `is_conditional_effect` and `AssignEffect::conditions` are read *nowhere*
  else in `src/search` (checked by grep). `StateRegistry::get_numeric_successor`
  and its packed-state sibling loop over `op.get_assign_effects()` and apply each
  one with no reference to its conditions;
* `verify_no_conditional_effects` (`src/search/task_tools.cc:65-74`) only
  inspects `op.get_effects()`, the propositional effects. A task whose *only*
  conditional effects are numeric passes that guard.

So a conditional numeric effect is silently treated as unconditional from end to
end, in both directions of wrongness: a resource is consumed or produced in
states where the PDDL says it is not, which can make an invalid plan look valid
or an optimal plan look more expensive than it is.

### Minimal reproduction

```lisp
(define (domain condnum)
  (:requirements :strips :fluents)
  (:predicates (open))
  (:functions (level))
  (:action pour
    :precondition ()
    :effect (and (when (open) (increase (level) 1)))))
```

```lisp
(define (problem condnum-1)
  (:domain condnum)
  (:init (= (level) 0))
  (:goal (>= (level) 1)))
```

`(open)` never holds, so `pour` can never raise `(level)` and the task is
unsolvable. Translate, then diff the assignment-effect line of `pour` in
`output.sas` against the same line after `preprocess`: the leading condition
count goes from `1` to `0`. The search then solves the task with one `pour`.

The same shape with the effect condition reachable but not initially true gives
the cheaper-than-optimal variant: the plan skips whatever action would have
established the condition.

### How we fixed it here

`planforge-translate/src/preprocess/operator.rs`, commit **`70ce786`**
(*fix(translate): fail loudly instead of silently rewriting the task*). Our port
had reproduced the bug literally as `let _ = ecs;` next to an unused
`NumericEffect::new_conditional`. The fix passes the conditions through, and the
writer — which indexed them into the numeric variable table instead of the
propositional one — was corrected at the same time. Note for whoever files this:
upstream's writer already indexes them correctly, so only the reader (and the
successor generator) need changing there.

A pre-existing test in our tree asserted that the conditions *are* discarded,
under a name claiming they are preserved. Worth checking whether
`src/preprocess` upstream has anything similar pinning the behaviour.

---

## 2. The invariance analysis depends on hash-set iteration order (CONFIRMED)

### Where

* `src/translate/invariant_finder.py:17` — `self.predicates_to_add_actions = defaultdict(set)`
* `src/translate/invariant_finder.py:41-42` — `get_threats` returns that `set`
* `src/translate/invariants.py:240-251` — `check_balance`
* `src/translate/invariants.py:199` — `self.parts = frozenset(parts)`

### What is wrong

```python
    def check_balance(self, balance_checker, enqueue_func):
        actions_to_check = set()
        for part in self.parts:
            actions_to_check |= balance_checker.get_threats(part.predicate)
        for action in actions_to_check:
            heavy_action = balance_checker.get_heavy_action(action)
            if self.operator_too_heavy(heavy_action):
                return False
            if self.operator_unbalanced(action, enqueue_func):
                return False
        return True
```

The loop **stops at the first action that fails** and `operator_unbalanced`
**enqueues refined candidates as a side effect on the way out**. Which action is
examined first therefore decides which refinements are generated, hence which
invariants are found, hence which SAS variables the task gets — and every number
measured from the resulting encoding.

The order is not fixed. `actions_to_check` is a `set`, and it is filled by
iterating `self.parts`, a `frozenset` of `InvariantPart` whose
`__hash__` is `hash((self.predicate, tuple(self.order)))` over predicate *names*
(`invariants.py:126-127`). Python 3 randomizes string hashing per process unless
`PYTHONHASHSEED` is set, so both the accumulation order and the iteration order
of `actions_to_check` vary between runs of the same command on the same input.

Mainline Fast Downward fixed exactly this and says so in the code
(`downward/src/translate/invariants.py:319-334`):

```python
    def check_balance(self, balance_checker, enqueue_func):
        actions_to_check = dict()
        # We will only use the keys of the dictionary. We do not use a set
        # because it's not stable and introduces non-determinism in the
        # invariance analysis.
        for part in sorted(self.parts):
            for a in balance_checker.get_threats(part.predicate):
                actions_to_check[a] = True
        ...
```

with `predicates_to_add_actions = defaultdict(list)` and a seeded
`random.Random(314159)` to draw the next action from
(`downward/src/translate/invariant_finder.py:17-18`). numeric-fd2 has none of
this; it is the pre-fix code.

### Reproduction, and its limits — please read before filing

We could **not** produce an input on which the encoding actually changes. Sweep:
all 21 numeric benchmark domains in `tests/assets/numeric-pddl-files`, translated
with numeric-fd2's `src/translate/translate.py` under `PYTHONHASHSEED` 1..4, then
comparing the propositional fact groups of `output.sas` as an
order-insensitive canonical form. All 21 are identical across all four seeds.
(`output.sas` itself differs byte-wise between seeds, but only in the ordering of
lines and in numeric-variable numbering, not in which invariants were found.)

That matches our own experience: our fix was byte-identical on 22 fixtures and
unchanged on all 21 benchmarks. Two things damp the effect in practice: `set` of
`Action` uses identity hashing, since `pddl.Action` defines no `__hash__`, so
that layer is not string-hash randomized; and on these domains the balance check
apparently never reaches an ordering where the first failure differs.

So this is best filed as **a latent defect with a known consequence**, on the
strength of the code and of mainline's own comment, not as a reproduced
behavioural difference. If a reproduction is wanted, the place to look is a
domain with several predicates in one candidate invariant and several
add-effect actions per predicate.

### How we fixed it here

Commit **`92a597f`** (*fix(invariants): make the balance check independent of the
run*). The threatening actions are kept in an insertion-ordered list per
predicate and collected in the order of the invariant's sorted parts. We did not
copy mainline's `random.Random(314159)` draw: reproducing CPython's Mersenne
Twister to agree on an order would be a strange thing for a Rust translator to
do, and the collection order is already reproducible without agreeing on
anything. For upstream, mainline's version is the obvious patch to take.

---

## 3. The invariant-generation budget is measured on the wall clock (CONFIRMED)

### Where

`src/translate/invariant_finder.py:102` and `:105`.

### What is wrong

```python
    start_time = time.time()
    while candidates:
        candidate = candidates.popleft()
        if time.time() - start_time > options.invariant_generation_max_time:
            print("Time limit reached, aborting invariant generation")
            return
```

Mainline uses `time.process_time()` in both places
(`downward/src/translate/invariant_finder.py:107,110`). Against a wall clock, how
many invariants a task gets depends on how busy the machine was, so the same
command on the same input produces a different — and worse — encoding under load
than on an idle machine. That is a correctness-adjacent reproducibility problem
in its own right, and it also silently degrades results in parallel experiment
runs, which is where this code spends most of its life.

### Reproduction

The budget demonstrably changes the encoding. On
`tests/assets/numeric-pddl-files/satellite`, with numeric-fd2's translator:

| `--invariant-generation-max-time` | propositional fact groups | group sizes |
|---|---|---|
| `300` (default is 300) | 8 | 2,2,2,2,2,2,2,**7** |
| `0` | 15 | fifteen binary groups |

The size-7 group is a real multi-valued variable; losing it costs the search a
strictly weaker encoding. Since the deadline is compared against the wall clock,
a loaded machine walks the table from the first row towards the second.

### How we fixed it here

Same commit, **`92a597f`**: the budget is spent in process CPU time.

---

## 4. Not applicable: `negate_literal` falling back to the un-negated literal

Our `planforge-translate/src/axiom_rules.rs` had four sites of the form

```rust
literal.negate_literal().unwrap_or_else(|| literal.clone())
```

which substitutes `L` for `¬L` — an inversion of meaning dressed up as a
default. Fixed in commit **`70ce786`** by a named helper that panics, since after
normalization every axiom literal is negatable.

**This has no upstream counterpart.** `src/translate/axiom_rules.py:168-192`
calls `condition[0].negate()` and `axioms[0].effect.negate()` directly;
`Literal.negate()` (`src/translate/pddl/conditions.py:295-296` and the
`NegatedAtom` sibling) returns the negated literal unconditionally and cannot
fail, so there is no failure branch and nothing to fall back to. Our bug came
from giving the operation an `Option` return in the port and then handling
`None` badly.

---

## 5. Not applicable: `fluent_facts` holding predicate names instead of atoms

This was the most serious bug we found: `GroundingTables::fluent_facts` was a set
of predicate *names*, so instantiation classified every ground atom of a fluent
predicate as fluent, including atoms the reachability model never proves. Such an
atom has no SAS variable, and the condition translator dropped it as "static and
true" — but an unreachable atom is statically *false*, so a condition that can
never hold became one that always holds. On axioms, which are instantiated over
all parameter tuples rather than only reachable ones, this turned a derived
predicate unconditionally true and produced plans **cheaper than the task
admits**. Fixed in commit **`40db948`**.

**This has no upstream counterpart.** numeric-fd2 is already atom-level:

```python
def get_fluent_facts(task, model):
    fluent_predicates = normalize.get_fluent_predicates(task)
    return set([fact for fact in model if fact.predicate in fluent_predicates])
```
(`src/translate/instantiate.py:16-18`)

and `Atom.instantiate` tests the atom, not its predicate, and raises `Impossible`
for an atom that is neither fluent nor in the initial state:

```python
        atom = Atom(self.predicate, args)
        if atom in fluent_facts:
            result.append(atom)
        elif atom not in init_facts:
            raise Impossible()
```
(`src/translate/pddl/conditions.py:287-294`)

That is the correct behaviour and the one we ported *to*. Our bug was introduced
by the port.

---

## 6. Not applicable: the `(:derived ...)` parse path

Our `parse_axiom` read a `(:derived (head ...) CONDITION)` block as if it were
PDDL 2.1's `(:axiom :vars ... :context ...)`, taking the head predicate list for
an atom and panicking. Every `:derived` block in every domain hit it, which is
why our fixture corpus had none until commit **`fd69eca`**.

**This has no upstream counterpart.** `src/translate/pddl_parser/parsing_functions.py:433-441`
is the correct modern form:

```python
def parse_axiom(alist, type_dict, predicate_dict):
    assert len(alist) == 3
    assert alist[0] == ":derived"
    predicate = parse_predicate(alist[1])
    condition = parse_condition(alist[2], type_dict, predicate_dict)
    return pddl.Axiom(predicate.name, predicate.arguments,
                      len(predicate.arguments), condition)
```

Again a port artefact on our side.

---

## 7. Not applicable: a derived goal restated by the rule proving the opposite fact

For completeness, because it is the bug that prompted this document and it is
tempting to assume a shared ancestor.

Eight places in our search restated a goal at a derived variable by looking the
*variable* up in the axioms and taking a rule's body. Every rule proves its head,
so at the variable's default value — a negated derived goal, `(not (alarm))` —
that hands the goal the body of a rule establishing the opposite fact, and the
abstraction ends up measuring the distance to the negation of the goal. On the
new `tests/assets/derived-predicates/negated-goal` fixture this made
`canonical(domain())` return a four-action plan for a three-action task. Fixed in
commits **`7cca841`** and **`a05ff3e`**.

**This has no upstream counterpart, and could not have.** numeric-fd2's
heuristics do not support propositional axioms at all:

* `src/search/numeric_pdbs/pdb_heuristic.cc:55` documents
  `"axioms", "not supported"`, and `NumericTaskProxy::verify_is_restricted_numeric_task`
  calls `verify_no_non_numeric_axioms`, which exits with `UNSUPPORTED` for
  anything beyond the two numeric goal axioms
  (`src/search/task_tools.cc:31-49`);
* the only `compute_abstract_goals` in the tree maps the goal fact straight
  through the domain mapping with no axiom substitution at all
  (`src/search/domain_abstractions/domain_abstraction_factory copy.cc:125-136`),
  and that file is not referenced from `DownwardFiles.cmake`, i.e. it is not
  compiled.

The substitution is an extension we added to support derived predicates, so the
bug is entirely ours.

---

## Loose end found while checking, not part of the report

On the `negated-goal` fixture our own `scp(domain())` (saturated cost
partitioning over domain abstractions) is inadmissible — `h = inf` for a state
two steps from the goal — and it stays inadmissible when the derived predicate is
removed and the goal written as `(not (breach v1))` directly. So it is a separate
defect in the cost partitioning, unrelated to axioms, and unrelated to anything
upstream. Recorded here only so it is not lost.
