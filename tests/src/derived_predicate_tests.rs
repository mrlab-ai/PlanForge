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
//! [`FIXTURE_SHAPES`] additionally pins the *shape* each fixture translates to:
//! how many axiom layers it occupies, how many derived facts have several
//! proofs, whether its support graph is cyclic. That is what says the corpus
//! reaches those code paths at all, rather than merely passing.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

use planforge_sas::numeric_task::{AbstractNumericTask, NumericRootTask, TaskRef};
use planforge_sas::state_registry::StateRegistry;
use planforge_search::evaluation::abstraction_collections::canonical_heuristic::CanonicalAbstractionHeuristic;
use planforge_search::evaluation::abstraction_collections::component::AbstractionComponent;
use planforge_search::evaluation::check_admissible::CheckAdmissibleHeuristic;
use planforge_search::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::{
    DomainAbstractionCollectionGeneratorMultipleCegar,
    DomainAbstractionCollectionGeneratorMultipleCegarConfig,
};
use planforge_search::evaluation::heuristic::Heuristic;
use planforge_search::evaluation::pattern_databases::pattern_generator_greedy::GreedyPatternGeneratorConfig;
use planforge_search::evaluation::pattern_databases::pdb_heuristic::GreedyNumericPdbHeuristic;
use planforge_search::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::{
    LandmarkCutNumericHeuristic, LmCutNumericConfig,
};
use planforge_search::search::{AStarSearch, SearchEngine, SearchStatus};

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
    /// Derived variables whose axiom default is *true*. Zero for every fixture,
    /// and pinned so that it stays zero: every derived variable defaults to
    /// false, and the layering is only sound because of it.
    true_defaults: usize,
    /// Derived facts - a variable *and* a value - that more than one axiom
    /// proves: disjunctive support, which no `axioms_by_atom` entry may collapse
    /// to one rule. Every axiom proves its head since issue454, so this counts
    /// genuine disjunctive support rather than also counting the several rules a
    /// cross-product negation used to produce for one refuted variable.
    proved: usize,
    /// Derived variables that positively support themselves through a cycle -
    /// the same strongly connected component. Recursive axioms are the reason
    /// derived predicates exist, and a cyclic component is what issue453's
    /// cluster overapproximation is about.
    cyclic: usize,
    /// Compiled numeric conditions feeding the propositional axioms.
    comparisons: usize,
    /// Goals at a derived variable's axiom *default* value - a negated derived
    /// goal. No rule proves such a goal, so an abstraction that keys its goal
    /// handling by the derived variable rather than by the derived fact hands it
    /// the body of a rule establishing the opposite fact and measures the
    /// distance to the negation of the goal. One fixture has one; pinned at zero
    /// everywhere else, because that is what let the bug stay latent.
    default_value_goals: usize,
}

/// Optimum of every fixture, measured with `planforge --search 'astar(blind())'`
/// and derived by hand in each problem file's header. Blind A* is optimal, so
/// the cost is the true optimum and the length pins the plan realising it.
const FIXTURE_OPTIMA: &[(&str, f64, u64)] = &[
    ("conjunctive-chain", 4.0, 4),
    ("cyclic-negation", 3.0, 3),
    ("disjunctive-support", 5.0, 5),
    ("goal-condition", 5.0, 5),
    ("layered-chain", 4.0, 4),
    ("negated-dependency", 3.0, 3),
    ("negated-goal", 3.0, 3),
    ("numeric-body", 3.0, 3),
    ("recursive-closure", 3.0, 3),
];

/// What each fixture translates to. Both tables are compared set-wise against
/// the fixtures on disk, so neither can go stale and a new fixture cannot be
/// added without being pinned in both.
#[rustfmt::skip]
const FIXTURE_SHAPES: &[(&str, Shape)] = &[
    ("conjunctive-chain",
     Shape { layers: 1, true_defaults: 0, proved: 0, cyclic: 0, comparisons: 0, default_value_goals: 0 }),
    ("cyclic-negation",
     Shape { layers: 1, true_defaults: 0, proved: 2, cyclic: 4, comparisons: 0, default_value_goals: 0 }),
    ("disjunctive-support",
     Shape { layers: 1, true_defaults: 0, proved: 1, cyclic: 0, comparisons: 0, default_value_goals: 0 }),
    ("goal-condition",
     Shape { layers: 1, true_defaults: 0, proved: 0, cyclic: 0, comparisons: 0, default_value_goals: 0 }),
    // `proved: 0` since issue454: the pair that used to have two rules was
    // `flawed` at its *default* value. `flawed <- cracked and not dirty` has a
    // two-literal body, so the cross-product negation produced both
    // `not flawed <- not cracked` and `not flawed <- dirty`. Nothing here is
    // disjunctively *proved* - every derived predicate has one `:derived` clause.
    ("layered-chain",
     Shape { layers: 3, true_defaults: 0, proved: 0, cyclic: 0, comparisons: 0, default_value_goals: 0 }),
    ("negated-dependency",
     Shape { layers: 2, true_defaults: 0, proved: 0, cyclic: 0, comparisons: 0, default_value_goals: 0 }),
    // The only fixture with a negated derived goal, and the only one solvable
    // under the `domain_abstraction` family: it has no derived precondition, and
    // a domain abstraction has no propositional axioms to establish one with.
    ("negated-goal",
     Shape { layers: 1, true_defaults: 0, proved: 0, cyclic: 0, comparisons: 0, default_value_goals: 1 }),
    ("numeric-body",
     Shape { layers: 1, true_defaults: 0, proved: 0, cyclic: 0, comparisons: 1, default_value_goals: 0 }),
    ("recursive-closure",
     Shape { layers: 1, true_defaults: 0, proved: 1, cyclic: 4, comparisons: 0, default_value_goals: 0 }),
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

/// Measures the shape [`FIXTURE_SHAPES`] pins.
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

    // Keyed by the derived *fact*, not its variable. Both halves of the key are
    // still worth having even though every axiom now proves its head: it is what
    // makes the assertion below able to say *which* fact a stray rule writes.
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
        default_value_goals: (0..task.get_num_goals())
            .map(|goal_id| task.get_goal_fact(goal_id))
            .filter(|goal| defaults.get(&goal.var()) == Some(&goal.value()))
            .count(),
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

/// The same optima under an admissible heuristic that *reads* the axioms.
///
/// This is the only test that can see whether the refuting rules are right.
/// Blind A* cannot: the axiom evaluator refutes a derived variable by finding it
/// unproven at the end of its layer, so it never fires such a rule and a wrong
/// set of them changes no plan. `lmcutnumeric` builds its relaxation out of them,
/// and since issue454 out of the ones it derives for itself with
/// `planforge_sas::default_value_axioms`. A derived variable it cannot refute
/// makes the state look like a dead end, which is how negating a cyclic component
/// literal by literal used to turn `cyclic-negation` into an unsolvable task with
/// h = infinity in the initial state. That is why this fixture is the canary for
/// anything touching the rules that refute a derived variable.
#[test]
fn an_axiom_reading_heuristic_finds_every_fixture_optimum() {
    for &(name, cost, length) in FIXTURE_OPTIMA {
        let task = fixture_task(name);
        let registry = StateRegistry::for_task(Arc::new(&task));
        let heuristic = LandmarkCutNumericHeuristic::from_config(
            &task as &dyn AbstractNumericTask,
            LmCutNumericConfig::default(),
        )
        .expect("the default lmcutnumeric config is supported");
        let mut search = AStarSearch::new(
            Arc::new(&task),
            registry,
            Some(Box::new(heuristic)),
            None,
            None,
        );
        let result = search.search().expect("lmcutnumeric A* search failed");

        let plan = match (&result.status, &result.plan) {
            (SearchStatus::Solved(_), Some(plan)) => plan,
            (status, _) => panic!(
                "{name}: lmcutnumeric A* must solve the fixture, got {status:?} after \
                 {} dead ends",
                result.dead_ends
            ),
        };
        let found = Solution {
            cost: result
                .solution_cost
                .unwrap_or_else(|| plan.iter().map(|op| op.cost() as f64).sum()),
            length: plan.len() as u64,
        };
        let expected = Solution { cost, length };
        assert!(
            found.matches(&expected),
            "{name}: lmcutnumeric A* returned {found:?}, expected the optimum {expected:?}; an \
             inadmissible axiom relaxation would overestimate here"
        );
    }
}

/// Domain abstractions of the negated-goal fixture are admissible and keep the
/// optimum.
///
/// `negated-goal` is the one fixture the `domain_abstraction` family can build at
/// all: an abstract operator never writes a derived variable, so a derived
/// *precondition* leaves the abstraction unable to establish it and the generator
/// refuses with an abstract dead end, which is what the other eight fixtures get.
/// What is left is the goal, and a goal at a derived variable's default value is
/// the case the abstraction used to restate as the body of the rule proving the
/// *opposite* fact: `(not (alarm v1))` became `(breach v1)`. That reversal is not
/// a weaker bound but a wrong one — `arm` cannot reopen a breach once the vault is
/// sealed, so the cheapest goal state came out as h = infinity, was pruned as a
/// dead end, and A* returned the four-action plan through `deliver-open` instead
/// of the three-action optimum.
///
/// A single abstraction does not see it: with `init_split_method = InitValue` it
/// never refines `sealed` far enough for the substituted goal to become
/// unreachable, and answers the same 2 either way. It takes the collection, whose
/// members seed their init split on different variables, so this test builds one.
/// The two clock-free stopping rules keep it a function of its inputs rather than
/// of the machine — see `sailing_simple_tests::standard_round7_collection_config`
/// for why that matters.
#[test]
fn domain_abstractions_of_a_negated_derived_goal_are_admissible_and_keep_the_optimum() {
    let name = "negated-goal";
    let &(_, cost, length) = FIXTURE_OPTIMA
        .iter()
        .find(|(pinned, _, _)| *pinned == name)
        .expect("the negated-goal fixture is pinned");
    let expected = Solution { cost, length };

    let task = fixture_task(name);
    assert_eq!(
        pinned_shape(name).default_value_goals,
        1,
        "{name} is the fixture that has to carry the negated derived goal"
    );

    let abstractions = DomainAbstractionCollectionGeneratorMultipleCegar::new(
        DomainAbstractionCollectionGeneratorMultipleCegarConfig {
            max_abstraction_size: 10_000,
            max_collection_size: 10_000,
            stagnation_limit: 0.0,
            enable_blacklist_on_stagnation: false,
            compute_operator_footprints: false,
            ..Default::default()
        },
    )
    .generate_collection(&task)
    .unwrap_or_else(|error| {
        panic!("{name}: domain abstractions must build over a negated derived goal: {error:#}")
    });
    assert!(
        abstractions.len() > 1,
        "{name}: the collection has to hold more than one abstraction for the members to \
         disagree on which variables they refine, got {}",
        abstractions.len()
    );

    let task_ref: TaskRef<'_> = Arc::new(&task);
    let components = abstractions
        .into_iter()
        .enumerate()
        .map(|(index, abstraction)| {
            AbstractionComponent::domain(Some(format!("negated_goal_{index}")), abstraction)
        })
        .collect();
    let canonical = CanonicalAbstractionHeuristic::new(None, &task, components)
        .expect("the canonical heuristic constructs over the collection");
    let guarded = CheckAdmissibleHeuristic::new(
        Some(Box::new(canonical) as Box<dyn Heuristic + '_>),
        Arc::clone(&task_ref),
    )
    .expect("the admissibility guard constructs over the fixture");

    let registry = StateRegistry::for_task(Arc::clone(&task_ref));
    let mut search = AStarSearch::new(task_ref, registry, Some(Box::new(guarded)), None, None);
    let result = search.search().unwrap_or_else(|error| {
        panic!("{name}: guarded canonical domain-abstraction A* failed: {error}")
    });

    let plan = match (&result.status, &result.plan) {
        (SearchStatus::Solved(_), Some(plan)) => plan,
        (status, _) => panic!(
            "{name}: the guarded canonical domain abstraction must solve the fixture, got \
             {status:?}"
        ),
    };
    let found = Solution {
        cost: result
            .solution_cost
            .unwrap_or_else(|| plan.iter().map(|op| op.cost() as f64).sum()),
        length: plan.len() as u64,
    };
    assert!(
        found.matches(&expected),
        "{name}: guarded canonical domain-abstraction A* returned {found:?}, expected the \
         optimum {expected:?}"
    );
}

/// A pattern database of the negated-goal fixture is admissible.
///
/// The projected task walks each goal and, for a goal at a derived variable,
/// replaces it by the body of the deriving rule. That expansion is what pulls the
/// body variables into the projection, and for a positive derived goal it is a
/// necessary condition, so the projected goal stays implied by the real one. At
/// the derived variable's default value it is the *opposite* condition: the
/// projected goal came out as `(breach v1)` where the real goal wants no breach,
/// and a pattern holding both `breach` and `sealed` then measured the distance to
/// a state the goal excludes.
///
/// The greedy generator is the one that gets there: `haul` reads `(breach v1)` and
/// leads to `(delivered v1)`, which makes `breach` a causal predecessor of a goal
/// variable and therefore a pattern member. The failure is pure inadmissibility -
/// A* still returned the optimum here - so `check_admissible` is what names it.
#[test]
fn a_pattern_database_of_a_negated_derived_goal_is_admissible() {
    let name = "negated-goal";
    let &(_, cost, length) = FIXTURE_OPTIMA
        .iter()
        .find(|(pinned, _, _)| *pinned == name)
        .expect("the negated-goal fixture is pinned");
    let expected = Solution { cost, length };

    let task = fixture_task(name);
    let pdb = GreedyNumericPdbHeuristic::new(
        &task as &dyn AbstractNumericTask,
        GreedyPatternGeneratorConfig::default(),
    )
    .unwrap_or_else(|error| panic!("{name}: the greedy numeric PDB must build: {error}"));

    let task_ref: TaskRef<'_> = Arc::new(&task);
    let guarded = CheckAdmissibleHeuristic::new(
        Some(Box::new(pdb) as Box<dyn Heuristic + '_>),
        Arc::clone(&task_ref),
    )
    .expect("the admissibility guard constructs over the fixture");

    let registry = StateRegistry::for_task(Arc::clone(&task_ref));
    let mut search = AStarSearch::new(task_ref, registry, Some(Box::new(guarded)), None, None);
    let result = search
        .search()
        .unwrap_or_else(|error| panic!("{name}: guarded greedy-PDB A* failed: {error}"));

    let plan = match (&result.status, &result.plan) {
        (SearchStatus::Solved(_), Some(plan)) => plan,
        (status, _) => {
            panic!("{name}: the guarded greedy PDB must solve the fixture, got {status:?}")
        }
    };
    let found = Solution {
        cost: result
            .solution_cost
            .unwrap_or_else(|| plan.iter().map(|op| op.cost() as f64).sum()),
        length: plan.len() as u64,
    };
    assert!(
        found.matches(&expected),
        "{name}: guarded greedy-PDB A* returned {found:?}, expected the optimum {expected:?}"
    );
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
