//! Hand-written fixtures for PDDL `:derived` predicates.
//!
//! No benchmark in `assets/numeric-pddl-files` declares one, and the only
//! propositional axiom those tasks have is the empty-bodied global constraint.
//! So nothing there reaches axiom layering, negation by failure, disjunctive
//! support or recursive axioms, and a green corpus run says nothing about them.
//!
//! Every fixture here is built so that *losing* the behaviour it covers changes
//! the optimal cost: each problem file derives the optimum by hand together with
//! the cost each plausible failure mode would produce instead. A fixture that
//! still solved at the same cost with axioms mishandled would be worthless, so
//! where a failure mode is only detectable as unsolvability that is called out
//! rather than relied on.
//!
//! [`FIXTURES`] additionally pins the *shape* each fixture translates to - how
//! many axiom layers it occupies, how many of its derived variables default to
//! true, whether its support graph is cyclic - because that is what says the
//! corpus reaches those code paths at all, rather than merely passing.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

use planforge_sas::numeric_task::{AbstractNumericTask, NumericRootTask};
use planforge_sas::state_registry::StateRegistry;

use crate::corpus::{
    self, Scratch, Solution, assert_fixture_set_is_pinned, blind_astar, subdirectory_names,
    translate_in_memory, translate_to_disk,
};
use crate::numeric_corpus_tests::assert_axiom_layering_contract;

/// The structure a fixture translates to, which is what says it reaches the code
/// path it was written for.
#[derive(Debug, PartialEq)]
struct Shape {
    /// Number of distinct axiom layers the derived propositional variables
    /// occupy, counting the global-constraint atom the translator injects. More
    /// than one means the task needs the layered evaluation order rather than a
    /// single fixpoint.
    layers: usize,
    /// Derived variables whose axiom default is *true*, i.e. the ones the
    /// translator only ever has to refute. These are the variables mainline's
    /// issue453 removes by making every derived variable default to false.
    true_defaults: usize,
    /// Derived facts - a variable *and* a value - that more than one axiom
    /// proves: disjunctive support, which no `axioms_by_atom` entry may collapse
    /// to one rule.
    proved: usize,
    /// Derived variables that positively support themselves through a cycle -
    /// the same strongly connected component. Recursive axioms are the reason
    /// derived predicates exist, and a cyclic component is what issue453's
    /// cluster overapproximation is about.
    cyclic: usize,
    /// Compiled numeric conditions feeding the propositional axioms.
    comparisons: usize,
}

/// Optimum of every fixture, measured with `planforge --search 'astar(blind())'`
/// and derived by hand in each problem file's header. Blind A* is optimal, so
/// the cost is the true optimum and the length pins the plan realising it.
const FIXTURE_OPTIMA: &[(&str, f64, u64)] = &[
    ("conjunctive-chain", 4.0, 4),
    ("disjunctive-support", 5.0, 5),
    ("goal-condition", 5.0, 5),
    ("layered-chain", 4.0, 4),
    ("negated-dependency", 3.0, 3),
    ("numeric-body", 3.0, 3),
    ("recursive-closure", 3.0, 3),
];

/// What each fixture translates to. Both tables are compared set-wise against
/// the fixtures on disk, so neither can go stale and a new fixture cannot be
/// added without being pinned in both.
#[rustfmt::skip]
const FIXTURE_SHAPES: &[(&str, Shape)] = &[
    ("conjunctive-chain",
     Shape { layers: 1, true_defaults: 0, proved: 0, cyclic: 0, comparisons: 0 }),
    ("disjunctive-support",
     Shape { layers: 1, true_defaults: 0, proved: 1, cyclic: 0, comparisons: 0 }),
    ("goal-condition",
     Shape { layers: 1, true_defaults: 0, proved: 0, cyclic: 0, comparisons: 0 }),
    ("layered-chain",
     Shape { layers: 3, true_defaults: 0, proved: 1, cyclic: 0, comparisons: 0 }),
    ("negated-dependency",
     Shape { layers: 2, true_defaults: 1, proved: 0, cyclic: 0, comparisons: 0 }),
    ("numeric-body",
     Shape { layers: 1, true_defaults: 0, proved: 0, cyclic: 0, comparisons: 1 }),
    ("recursive-closure",
     Shape { layers: 1, true_defaults: 0, proved: 1, cyclic: 4, comparisons: 0 }),
];

/// The pinned shape of `name`, which [`assert_fixture_set_is_pinned`] has
/// already established exists.
fn pinned_shape(name: &str) -> &'static Shape {
    &FIXTURE_SHAPES
        .iter()
        .find(|(pinned, _)| *pinned == name)
        .unwrap_or_else(|| panic!("no pinned shape for {name}"))
        .1
}

fn fixtures_root() -> PathBuf {
    corpus::assets().join("derived-predicates")
}

/// Translates the way the `planforge` binary does for a two-argument PDDL
/// invocation, which keeps genuine multi-valued variables.
fn fixture_task(name: &str) -> NumericRootTask {
    let dir = fixtures_root().join(name);
    translate_in_memory(&dir.join("domain.pddl"), &dir.join("problem.pddl"))
}

/// The heads of the compiled numeric conditions. They carry an axiom layer like
/// a derived propositional variable but are three-valued and are evaluated by
/// the comparison pass, so everything below excludes them.
fn comparison_heads(task: &NumericRootTask) -> BTreeSet<usize> {
    task.comparison_axioms()
        .iter()
        .map(|axiom| axiom.get_affected_var_id())
        .collect()
}

fn derived_propositional_variables(task: &NumericRootTask) -> Vec<usize> {
    let comparisons = comparison_heads(task);
    (0..task.get_num_variables())
        .filter(|&var| {
            task.get_variable_axiom_layer(var)
                .expect("variable is in range")
                .is_some()
                && !comparisons.contains(&var)
        })
        .collect()
}

/// Measures the shape [`FIXTURES`] pins.
fn shape_of(name: &str, task: &NumericRootTask) -> Shape {
    let derived = derived_propositional_variables(task);
    let defaults: BTreeMap<usize, usize> = derived
        .iter()
        .map(|&var| {
            assert_eq!(
                task.get_variable_domain_size(var),
                Ok(2),
                "{name}: derived variable {var} is not binary, so it has no single \
                 negation-by-failure value"
            );
            let default = task
                .get_variable_default_axiom_value(var)
                .expect("variable is in range");
            (var, default)
        })
        .collect();

    // Value 0 of a derived variable is the atom, value 1 its negation, so a
    // default of 0 is a variable the translator only ever refutes.
    let true_defaults = defaults.values().filter(|&&value| value == 0).count();

    // Keyed by the derived *fact*, not its variable: a variable that is both
    // proved and refuted has one axiom per value and is not disjunctively
    // supported.
    let mut proofs_of: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    // Positive support only: an axiom reading a derived variable at the value
    // its own axioms prove is the monotone case, which one layer's fixpoint
    // closes, and it is the edge a strongly connected component is made of. A
    // reading at the default value is negation by failure and is checked by
    // `assert_axiom_layering_contract` instead.
    let mut supports: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for axiom in task.axioms() {
        let head = axiom.var_id();
        assert!(
            defaults.contains_key(&head),
            "{name}: axiom writes non-derived variable {head}"
        );
        *proofs_of.entry((head, axiom.effect_value())).or_default() += 1;
        for condition in axiom.conditions() {
            let var = condition.var();
            if defaults.get(&var).is_some_and(|&d| d != condition.value()) {
                supports.entry(head).or_default().insert(var);
            }
        }
    }

    Shape {
        layers: derived
            .iter()
            .map(|&var| {
                task.get_variable_axiom_layer(var)
                    .expect("variable is in range")
                    .expect("derived variable has a layer")
            })
            .collect::<BTreeSet<usize>>()
            .len(),
        true_defaults,
        proved: proofs_of.values().filter(|&&count| count > 1).count(),
        cyclic: derived
            .iter()
            .filter(|&&var| supports_itself(var, &supports))
            .count(),
        comparisons: task.comparison_axioms().len(),
    }
}

/// Whether `start` positively supports itself, i.e. lies on a cycle of the
/// support graph.
fn supports_itself(start: usize, supports: &BTreeMap<usize, BTreeSet<usize>>) -> bool {
    let mut seen = BTreeSet::new();
    let mut stack: Vec<usize> = supports
        .get(&start)
        .into_iter()
        .flatten()
        .copied()
        .collect();
    while let Some(var) = stack.pop() {
        if var == start {
            return true;
        }
        if seen.insert(var) {
            stack.extend(supports.get(&var).into_iter().flatten().copied());
        }
    }
    false
}

#[test]
fn derived_predicate_fixtures_keep_their_optima_and_shape() {
    let discovered = subdirectory_names(&fixtures_root());
    assert_fixture_set_is_pinned(
        "derived-predicates optima",
        &discovered,
        &FIXTURE_OPTIMA
            .iter()
            .map(|(name, _, _)| *name)
            .collect::<Vec<&str>>(),
    );
    assert_fixture_set_is_pinned(
        "derived-predicates shapes",
        &discovered,
        &FIXTURE_SHAPES
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<&str>>(),
    );

    for &(name, cost, length) in FIXTURE_OPTIMA {
        let task = fixture_task(name);
        assert_axiom_layering_contract(name, &task);

        let expected = Solution { cost, length };
        let found = blind_astar(&task).unwrap_or_else(|| {
            panic!("{name}: blind A* found no plan, expected the optimum {expected:?}")
        });
        assert!(
            found.matches(&expected),
            "{name}: blind A* returned {found:?}, expected {expected:?}"
        );
        assert_eq!(
            &shape_of(name, &task),
            pinned_shape(name),
            "{name}: translated shape"
        );
    }
}

/// The same optima through the `..._fast` translation path, which asks for
/// singleton fact groups.
///
/// Derived atoms are the ones most likely to be grouped with an ordinary fact
/// by the invariant analysis, and a derived variable sharing a variable with a
/// non-derived one would have an ambiguous axiom layer. Running both paths pins
/// that the optima do not depend on which grouping was chosen.
#[test]
fn derived_predicate_fixtures_keep_their_optima_with_singleton_fact_groups() {
    for &(name, cost, length) in FIXTURE_OPTIMA {
        let dir = fixtures_root().join(name);
        let scratch = Scratch::new(&format!("derived_{name}"));
        let task = translate_to_disk(
            &dir.join("domain.pddl"),
            &dir.join("problem.pddl"),
            &scratch,
        );
        assert_axiom_layering_contract(name, &task);

        let expected = Solution { cost, length };
        let found = blind_astar(&task).unwrap_or_else(|| {
            panic!("{name}: blind A* found no plan on the singleton-group path")
        });
        assert!(
            found.matches(&expected),
            "{name}: blind A* returned {found:?} on the singleton-group path, expected {expected:?}"
        );
    }
}

/// A derived predicate is recomputed for every state, not carried over.
///
/// `recursive-closure` is the fixture that can tell: `build-link n4 n5` attaches
/// the one node the graph cannot reach, so the transitive closure of the
/// successor state is strictly larger than its predecessor's. The optimum
/// already depends on it; this pins the mechanism, so a failure says which of
/// the two broke.
#[test]
fn a_derived_closure_grows_when_an_operator_extends_the_graph() {
    let task = fixture_task("recursive-closure");
    let derived = derived_propositional_variables(&task);
    assert_eq!(
        derived.len(),
        18,
        "the closure over five nodes has a `path` variable per reachable pair, plus the \
         global-constraint atom"
    );

    let arc = Arc::new(&task);
    let mut registry = StateRegistry::for_task(arc.clone());
    let build_link = task
        .get_operators()
        .iter()
        .find(|operator| operator.name() == "build-link n4 n5")
        .expect("the fixture has a `build-link n4 n5` operator");

    let initial = registry.get_initial_state();
    let extended = registry
        .get_successor_state(&initial, build_link)
        .expect("`build-link n4 n5` is applicable in the initial state");

    let proven = |state: &_| -> BTreeSet<usize> {
        let mut propositional = Vec::new();
        let mut numeric = Vec::new();
        registry
            .fill_state_and_numeric_vars(state, &mut propositional, &mut numeric)
            .expect("unpacking a state of the fixture");
        derived
            .iter()
            .copied()
            .filter(|&var| {
                propositional[var]
                    != task
                        .get_variable_default_axiom_value(var)
                        .expect("variable is in range")
            })
            .collect()
    };

    let before = proven(&initial);
    let after = proven(&extended);
    assert!(
        before.is_subset(&after) && before.len() < after.len(),
        "extending the graph must prove strictly more derived atoms, got {} before and {} after",
        before.len(),
        after.len()
    );
}
