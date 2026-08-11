//! The numeric benchmark corpus under `assets/numeric-pddl-files`.
//!
//! Every test here is parameterised over the *discovered* fixture folders and
//! compared set-wise against a pinned table, so a fixture cannot be added,
//! renamed or removed without a test failing. That is what turns a green run
//! into evidence: the suite covers the corpus that is on disk, not the corpus
//! someone remembered to list.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{AbstractNumericTask, NumericRootTask, NumericType};
use planforge_sas::state_registry::StateRegistry;
use planforge_search::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::{
    LandmarkCutNumericHeuristic, LmCutNumericConfig,
};
use planforge_search::evaluation::numeric_landmarks::numeric_lm_cut_landmarks::LandmarkCutLandmarks;
use planforge_search::search::{AStarSearch, SearchEngine, SearchStatus};

use crate::corpus::{
    self, Scratch, Solution, assert_fixture_set_is_pinned, blind_astar, problem_file, single_file,
    subdirectory_names, translate_to_disk,
};

/// Known optima of every benchmark folder under `assets/numeric-pddl-files`,
/// measured with `planforge --search 'astar(blind())'`. Blind A* is optimal, so
/// `cost` is the true optimum of the task and `length` pins the plan realising
/// it.
///
/// The hand-written `sailing-simple` corpus has its own table in
/// `sailing_simple_tests` and is deliberately absent here.
const BENCHMARK_OPTIMA: &[(&str, f64, u64)] = &[
    ("counters-sym", 5.0, 5),
    ("delivery", 22.0, 10),
    ("depots", 22.0, 10),
    ("depots-sym", 22.0, 10),
    ("drone", 10.0, 10),
    ("expedition", 26.0, 26),
    ("farmland", 78.0, 78),
    ("farmland2", 78.0, 78),
    ("fn-counters-small_instances", 1.0, 1),
    ("forestfire", 24.0, 24),
    ("hydropower", 35.0, 35),
    ("minecraft-pogo-advanced", 6.0, 6),
    ("minecraft-sword-advanced", 2.0, 2),
    ("mprime", 4.0, 4),
    ("onlycraft-opt", 5.0, 5),
    ("pathwaysmetric", 12.0, 12),
    ("plant-watering", 13.0, 13),
    ("rover-unit", 10.0, 10),
    ("sailing", 23.0, 23),
    ("satellite", 108.586, 11),
    ("zenotravel", 9.0, 9),
];

/// Benchmarks that are excluded from everything that has to translate them:
/// both minecraft instances ground to roughly 275 variables and need about
/// seven seconds each in an unoptimized test build. Their recorded optima are
/// still checked against [`BENCHMARK_OPTIMA`] by
/// [`recorded_blind_stats_match_known_optima`].
const TOO_SLOW_TO_TRANSLATE_IN_TEST_BUILDS: &[&str] =
    &["minecraft-pogo-advanced", "minecraft-sword-advanced"];

/// Benchmarks with an operator that applies two numeric effects to the same
/// variable. Repeated additive effects on one variable used to be dropped
/// rather than accumulated, so which benchmarks exercise the path is worth
/// pinning: if this set shrinks, the corpus stopped covering it.
const WITH_REPEATED_NUMERIC_EFFECT_TARGET: &[&str] = &["mprime"];

/// Benchmarks with a numeric effect guarded by an effect condition. None of the
/// IPC-derived fixtures has one, which is why `numeric_condition_tests` carries
/// a hand-written fixture for that path.
const WITH_CONDITIONAL_NUMERIC_EFFECT: &[&str] = &[];

fn benchmarks_root() -> PathBuf {
    corpus::assets().join("numeric-pddl-files")
}

/// Every pinned benchmark, after asserting that the pinned set is exactly what
/// is on disk.
fn pinned_benchmarks() -> Vec<(&'static str, Solution, PathBuf)> {
    let root = benchmarks_root();
    let discovered: Vec<String> = subdirectory_names(&root)
        .into_iter()
        .filter(|name| name != "sailing-simple")
        .collect();
    let pinned: Vec<&str> = BENCHMARK_OPTIMA.iter().map(|(name, _, _)| *name).collect();
    assert_fixture_set_is_pinned("numeric-pddl-files", &discovered, &pinned);

    for slow in TOO_SLOW_TO_TRANSLATE_IN_TEST_BUILDS {
        assert!(
            pinned.contains(slow),
            "TOO_SLOW_TO_TRANSLATE_IN_TEST_BUILDS lists unknown benchmark {slow:?}"
        );
    }
    for name in WITH_REPEATED_NUMERIC_EFFECT_TARGET
        .iter()
        .chain(WITH_CONDITIONAL_NUMERIC_EFFECT)
    {
        assert!(
            pinned.contains(name) && !TOO_SLOW_TO_TRANSLATE_IN_TEST_BUILDS.contains(name),
            "numeric-effect table lists {name:?}, which is not a translated benchmark"
        );
    }

    BENCHMARK_OPTIMA
        .iter()
        .map(|&(name, cost, length)| {
            let dir = root.join(name);
            assert!(
                dir.join("domain.pddl").is_file(),
                "missing domain.pddl in {dir:?}"
            );
            (name, Solution { cost, length }, dir)
        })
        .collect()
}

/// The benchmarks that are translated by the tests below, with their tasks.
fn translated_benchmarks() -> Vec<(&'static str, Solution, NumericRootTask)> {
    pinned_benchmarks()
        .into_iter()
        .filter(|(name, _, _)| !TOO_SLOW_TO_TRANSLATE_IN_TEST_BUILDS.contains(name))
        .map(|(name, optimum, dir)| {
            let scratch = Scratch::new(&format!("numeric_corpus_{name}"));
            let task = translate_to_disk(&dir.join("domain.pddl"), &problem_file(&dir), &scratch);
            (name, optimum, task)
        })
        .collect()
}

/// Reads the blind-A* reference capture stored next to a problem file.
fn recorded_solution(bench_dir: &Path) -> Solution {
    let stats_file = single_file(bench_dir, |path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".fd_blind.json"))
    });
    let content = std::fs::read_to_string(&stats_file)
        .unwrap_or_else(|e| panic!("read_to_string failed for {stats_file:?}: {e}"));
    let json: serde_json::Value = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("invalid json {stats_file:?}: {e}"));
    let stats = json
        .get("stats")
        .unwrap_or_else(|| panic!("missing `stats` object in {stats_file:?}"));

    assert_eq!(
        stats.get("solution_found").and_then(|v| v.as_bool()),
        Some(true),
        "expected stats.solution_found=true in {stats_file:?}"
    );

    Solution {
        cost: stats
            .get("plan_cost")
            .and_then(|v| v.as_f64())
            .unwrap_or_else(|| panic!("missing numeric stats.plan_cost in {stats_file:?}")),
        length: stats
            .get("plan_length")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| panic!("missing integer stats.plan_length in {stats_file:?}")),
    }
}

/// Guards the recorded `.fd_blind.json` captures against silent drift: they must
/// agree with the optima pinned in this file.
#[test]
fn recorded_blind_stats_match_known_optima() {
    let mut mismatches: Vec<String> = Vec::new();

    for (name, optimum, dir) in pinned_benchmarks() {
        let recorded = recorded_solution(&dir);
        if !recorded.matches(&optimum) {
            mismatches.push(format!(
                "{name}: recorded {recorded:?}, expected {optimum:?}"
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "recorded blind-A* stats disagree with BENCHMARK_OPTIMA:\n{}",
        mismatches.join("\n")
    );
}

/// The real correctness gate: a planner regression changes a plan cost, and
/// this fails.
#[test]
fn blind_astar_reproduces_every_pinned_optimum() {
    let mut mismatches: Vec<String> = Vec::new();

    for (name, optimum, task) in translated_benchmarks() {
        match blind_astar(&task) {
            Some(actual) if actual.matches(&optimum) => eprintln!("{name}: {actual:?} (optimal)"),
            Some(actual) => {
                mismatches.push(format!("{name}: expected {optimum:?}, got {actual:?}"))
            }
            None => mismatches.push(format!(
                "{name}: expected {optimum:?}, but search did not return a solved plan"
            )),
        }
    }

    assert!(
        mismatches.is_empty(),
        "blind A* did not reproduce the known optima:\n{}",
        mismatches.join("\n")
    );
}

/// The layering contract the axiom evaluator relies on, checked on every
/// translated benchmark rather than on one hand-built task.
///
/// `AxiomEvaluator` evaluates arithmetic axioms bottom-up by numeric layer,
/// then all comparison axioms at one layer, then propositional axioms above
/// that. Nothing re-checks that the translated task actually has that shape, so
/// a translator or preprocessor change could silently produce a task the
/// evaluator reads in the wrong order.
#[test]
fn translated_tasks_satisfy_the_axiom_layering_contract() {
    for (name, _, task) in translated_benchmarks() {
        assert_axiom_layering_contract(name, &task);
    }
}

pub fn assert_axiom_layering_contract(name: &str, task: &dyn AbstractNumericTask) {
    let layer_of = |var: usize| -> Option<usize> {
        task.get_variable_axiom_layer(var)
            .unwrap_or_else(|e| panic!("{name}: variable {var} out of range: {e}"))
    };
    let numeric_count = task.numeric_variables().len();

    // Derived numeric variables are exactly the assignment-axiom targets, one
    // axiom each, and each sits strictly above every derived operand it reads.
    let mut numeric_layer_of: Vec<Option<usize>> = Vec::with_capacity(numeric_count);
    for variable in task.numeric_variables() {
        let layer = variable.axiom_layer();
        assert_eq!(
            layer.is_some(),
            matches!(variable.get_type(), NumericType::Derived),
            "{name}: numeric variable {:?} of type {:?} has axiom layer {layer:?}",
            variable.name(),
            variable.get_type()
        );
        numeric_layer_of.push(layer);
    }

    let mut written_by: BTreeMap<usize, usize> = BTreeMap::new();
    for (index, axiom) in task.assignment_axioms().iter().enumerate() {
        let target = axiom.get_affected_var_id();
        assert!(
            target < numeric_count,
            "{name}: assignment axiom {index} targets numeric variable {target} of {numeric_count}"
        );
        assert!(
            written_by.insert(target, index).is_none(),
            "{name}: numeric variable {target} is written by two assignment axioms"
        );

        let target_layer = numeric_layer_of[target].unwrap_or_else(|| {
            panic!("{name}: assignment axiom {index} writes non-derived numeric variable {target}")
        });
        for operand in [axiom.get_left_var_id(), axiom.get_right_var_id()] {
            assert!(
                operand < numeric_count,
                "{name}: assignment axiom {index} reads numeric variable {operand} of {numeric_count}"
            );
            if let Some(operand_layer) = numeric_layer_of[operand] {
                assert!(
                    operand_layer < target_layer,
                    "{name}: assignment axiom {index} reads derived variable {operand} at layer \
                     {operand_layer} to define layer {target_layer}; the evaluator would read a \
                     stale value"
                );
            }
        }
    }
    for (id, layer) in numeric_layer_of.iter().enumerate() {
        assert_eq!(
            layer.is_some(),
            written_by.contains_key(&id),
            "{name}: derived numeric variable {id} has no defining assignment axiom"
        );
    }

    // Comparison axioms all sit on one layer, which is the lowest derived
    // propositional layer and one above the last arithmetic layer.
    let comparison_layers: BTreeSet<usize> = task
        .comparison_axioms()
        .iter()
        .map(|axiom| {
            let head = axiom.get_affected_var_id();
            // A compiled numeric condition is two-valued and defaults to
            // `False`, so a condition the evaluator has not reached yet can
            // never look satisfied. Interval-based abstractions rely on exactly
            // that, and on there being no third value to mean anything else.
            assert_eq!(
                task.get_variable_domain_size(head),
                Ok(ConditionValue::DOMAIN_SIZE),
                "{name}: comparison-axiom head {head} is not a two-valued condition variable"
            );
            assert_eq!(
                task.get_variable_default_axiom_value(head),
                Ok(ConditionValue::False.as_usize()),
                "{name}: comparison-axiom head {head} must default to `False`"
            );
            for operand in [axiom.get_left_var_id(), axiom.get_right_var_id()] {
                assert!(
                    operand < numeric_count,
                    "{name}: comparison axiom reads numeric variable {operand} of {numeric_count}"
                );
            }
            layer_of(head).unwrap_or_else(|| {
                panic!("{name}: comparison axiom writes non-derived variable {head}")
            })
        })
        .collect();
    assert!(
        comparison_layers.len() <= 1,
        "{name}: comparison axioms are spread over layers {comparison_layers:?}, but the \
         evaluator applies all of them at a single layer"
    );

    let derived_propositional_layers: BTreeSet<usize> =
        (0..task.get_num_variables()).filter_map(layer_of).collect();
    if let Some(&comparison_layer) = comparison_layers.iter().next() {
        assert_eq!(
            derived_propositional_layers.iter().next(),
            Some(&comparison_layer),
            "{name}: comparison axioms must occupy the lowest derived propositional layer"
        );
        let last_arithmetic_layer = numeric_layer_of.iter().flatten().max().copied();
        assert_eq!(
            comparison_layer,
            last_arithmetic_layer.map_or(0, |layer| layer + 1),
            "{name}: the comparison layer must sit directly above the last arithmetic layer"
        );
        for axiom in task.axioms() {
            let layer = layer_of(axiom.var_id()).unwrap_or_else(|| {
                panic!(
                    "{name}: propositional axiom writes non-derived variable {}",
                    axiom.var_id()
                )
            });
            assert!(
                layer > comparison_layer,
                "{name}: propositional axiom on variable {} sits at layer {layer}, at or below \
                 the comparison layer {comparison_layer}",
                axiom.var_id()
            );
        }
    } else {
        for axiom in task.axioms() {
            assert!(
                layer_of(axiom.var_id()).is_some(),
                "{name}: propositional axiom writes non-derived variable {}",
                axiom.var_id()
            );
        }
    }

    assert_every_axiom_proves_its_head(name, task);
    assert_negation_by_failure_reads_a_settled_layer(name, task, &comparison_layers);

    // Derived variables are the axioms' business alone: no operator may write
    // one, or the axiom layer it was computed at would be silently invalidated.
    for operator in task.get_operators() {
        for effect in operator.effects() {
            assert!(
                layer_of(effect.var_id()).is_none(),
                "{name}: operator {:?} writes derived variable {}",
                operator.name(),
                effect.var_id()
            );
        }
        for effect in operator.assignment_effects() {
            let affected = effect.affected_var_id();
            assert!(
                affected < numeric_count,
                "{name}: operator {:?} writes numeric variable {affected} of {numeric_count}",
                operator.name()
            );
            assert!(
                matches!(
                    task.numeric_variables()[affected].get_type(),
                    NumericType::Regular | NumericType::Cost
                ),
                "{name}: operator {:?} writes {:?} numeric variable {:?}",
                operator.name(),
                task.numeric_variables()[affected].get_type(),
                task.numeric_variables()[affected].name()
            );
        }
    }
}

/// Every axiom *proves* its head; nothing in the task refutes a derived variable.
///
/// This is the issue454 contract, and it is load-bearing in three places. The
/// axiom evaluator refutes a derived variable by finding it unproven at the end of
/// its layer, and a rule writing the default value would announce that same
/// literal a second time. `planforge_sas::default_value_axioms` computes the
/// refuting rules and asserts this of its input. And the sites that recover a
/// derived goal's hidden conditions - the CEGAR flaw searches, the domain
/// abstraction goal maps, the projected-task goal expansion - keep one axiom per
/// variable, so a refuting rule could win the slot and hand them the negation of
/// the conditions they wanted.
fn assert_every_axiom_proves_its_head(name: &str, task: &dyn AbstractNumericTask) {
    for (axiom_id, axiom) in task.axioms().iter().enumerate() {
        let head = axiom.var_id();
        let default = task
            .get_variable_default_axiom_value(head)
            .unwrap_or_else(|e| panic!("{name}: variable {head} out of range: {e}"));
        assert_ne!(
            axiom.effect_value(),
            default,
            "{name}: axiom {axiom_id} sets derived variable {head} to its default value {default}; \
             the rules that do that belong to the heuristic that wants them, not to the task"
        );
        assert_eq!(
            axiom.precondition_value(),
            default,
            "{name}: axiom {axiom_id} on derived variable {head} claims the variable held \
             {} before it fired, but a proving rule fires on the default {default}",
            axiom.precondition_value()
        );
    }
}

/// The reason axiom layers exist, checked on the translated task.
///
/// An axiom that reads a derived variable at the value that variable's own
/// axioms *prove* is monotone: one layer's fixpoint closes it, so the two may
/// share a layer. An axiom that reads a derived variable at its *default* value
/// is negation by failure — it asks whether the variable stayed unproven — and
/// that answer only exists once the reader's layer is done, which is why
/// `AxiomEvaluator` admits those literals only between layers. So such a
/// reading must come from a strictly lower layer, or the evaluator answers it
/// before the evidence is in.
fn assert_negation_by_failure_reads_a_settled_layer(
    name: &str,
    task: &dyn AbstractNumericTask,
    comparison_layers: &BTreeSet<usize>,
) {
    let derived_layer = |var: usize| -> Option<usize> {
        let layer = task
            .get_variable_axiom_layer(var)
            .unwrap_or_else(|e| panic!("{name}: variable {var} out of range: {e}"))?;
        // A comparison head also carries a layer, but its value comes from the
        // numeric pass rather than from Horn rules, and its `unknown` default is
        // never derived, so it is not a negation-by-failure literal.
        (!comparison_layers.contains(&layer)).then_some(layer)
    };

    for axiom in task.axioms() {
        let head = axiom.var_id();
        let Some(head_layer) = derived_layer(head) else {
            continue;
        };
        for condition in axiom.conditions() {
            let read = condition.var();
            let Some(read_layer) = derived_layer(read) else {
                continue;
            };
            let default = task
                .get_variable_default_axiom_value(read)
                .unwrap_or_else(|e| panic!("{name}: variable {read} out of range: {e}"));
            if condition.value() != default {
                continue;
            }
            assert!(
                read_layer < head_layer,
                "{name}: the axiom on variable {head} at layer {head_layer} reads variable \
                 {read} at its default value {default}, but {read} is derived at layer \
                 {read_layer}; that negation by failure is answered before {read}'s layer has \
                 settled"
            );
        }
    }
}

/// Pins which benchmarks exercise the two numeric-effect shapes that are easy
/// to lose in translation. Both sets are compared set-wise, so a fixture that
/// stops producing a guarded or repeated effect fails here.
#[test]
fn numeric_effect_shapes_are_pinned_per_benchmark() {
    let mut with_repeated_target: Vec<String> = Vec::new();
    let mut with_conditional: Vec<String> = Vec::new();

    for (name, _, task) in translated_benchmarks() {
        let mut repeated = false;
        let mut conditional = false;
        for operator in task.get_operators() {
            let mut targets = BTreeSet::new();
            for effect in operator.assignment_effects() {
                repeated |= !targets.insert(effect.affected_var_id());
                conditional |= effect.is_conditional();
                assert_eq!(
                    effect.is_conditional(),
                    !effect.conditions().is_empty(),
                    "{name}: operator {:?} has an effect flagged conditional={} with {} \
                     conditions",
                    operator.name(),
                    effect.is_conditional(),
                    effect.conditions().len()
                );
            }
        }
        if repeated {
            with_repeated_target.push(name.to_string());
        }
        if conditional {
            with_conditional.push(name.to_string());
        }
    }

    assert_fixture_set_is_pinned(
        "benchmarks with a repeated numeric effect target",
        &with_repeated_target,
        WITH_REPEATED_NUMERIC_EFFECT_TARGET,
    );
    assert_fixture_set_is_pinned(
        "benchmarks with a conditional numeric effect",
        &with_conditional,
        WITH_CONDITIONAL_NUMERIC_EFFECT,
    );
}

/// `lmcutnumeric` on Plant Watering, end to end.
///
/// One translation feeds three checks that used to be three tests translating
/// the same instance: the initial-state value is finite and admissible, it
/// stays finite along an optimal blind plan, and a full A* with the heuristic
/// still returns the optimum without marking anything a dead end.
#[test]
fn plant_watering_lmcutnumeric_is_admissible_finite_and_solves_optimally() {
    const OPTIMAL_COST: f64 = 13.0;

    let dir = benchmarks_root().join("plant-watering");
    let scratch = Scratch::new("plant_watering_lmcut");
    let task = translate_to_disk(&dir.join("domain.pddl"), &problem_file(&dir), &scratch);

    let blind_plan = {
        let registry = StateRegistry::for_task(Arc::new(&task));
        let mut search = AStarSearch::new(Arc::new(&task), registry, None, None, None);
        let result = search.search().expect("blind A* search failed");
        match (&result.status, result.plan) {
            (SearchStatus::Solved(_), Some(plan)) => plan,
            (status, _) => {
                panic!("blind Plant Watering search should solve the task, got {status:?}")
            }
        }
    };

    // The initial state and every state along the optimal blind plan must have
    // a finite, admissible LM-cut value. A zero-cost plateau used to make some
    // of them look like dead ends.
    let mut registry = StateRegistry::for_task(Arc::new(&task));
    let mut state = registry.get_initial_state();
    let mut landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());
    let mut propositional_values = Vec::new();
    let mut numeric_values = Vec::new();

    for (step, operator) in std::iter::once(None)
        .chain(blind_plan.iter().map(Some))
        .enumerate()
    {
        registry
            .fill_state_and_numeric_vars(&state, &mut propositional_values, &mut numeric_values)
            .unwrap_or_else(|e| {
                panic!("failed to unpack Plant Watering state at step {step}: {e:?}")
            });
        let (dead_end, total_cost, _cuts) = landmarks
            .compute_landmarks(
                &propositional_values,
                state.buffer(&registry).len(),
                &numeric_values,
                false,
            )
            .unwrap_or_else(|e| panic!("LM-cut evaluation failed at step {step}: {e}"));

        let last_operator = operator.map(|op| op.name());
        assert!(
            !dead_end,
            "Plant Watering state at step {step} must not be a dead end; last operator: \
             {last_operator:?}"
        );
        assert!(
            total_cost.is_finite(),
            "Plant Watering LM-cut value at step {step} must be finite, got {total_cost}; last \
             operator: {last_operator:?}"
        );
        assert!(
            total_cost <= OPTIMAL_COST + 1e-6,
            "Plant Watering LM-cut value at step {step} must be admissible against \
             {OPTIMAL_COST}, got {total_cost}"
        );

        if let Some(operator) = operator {
            state = registry
                .get_successor_state(&state, operator)
                .unwrap_or_else(|e| {
                    panic!("failed to apply blind-plan operator at step {step}: {e:?}")
                });
        }
    }

    // The full search with the heuristic must still return the optimum.
    let registry = StateRegistry::for_task(Arc::new(&task));
    let heuristic = LandmarkCutNumericHeuristic::from_config(
        &task as &dyn AbstractNumericTask,
        LmCutNumericConfig::default(),
    )
    .expect("default lmcutnumeric config should be supported");
    let mut search = AStarSearch::new(
        Arc::new(&task),
        registry,
        Some(Box::new(heuristic)),
        None,
        None,
    );
    let result = search.search().expect("LM-cut A* search failed");

    let plan = match (&result.status, &result.plan) {
        (SearchStatus::Solved(_), Some(plan)) => plan,
        (status, _) => {
            panic!("Plant Watering lmcutnumeric search should solve the task, got {status:?}")
        }
    };
    let solution_cost = result
        .solution_cost
        .unwrap_or_else(|| plan.iter().map(|op| op.cost() as f64).sum());

    assert_eq!(
        result.dead_ends, 0,
        "Plant Watering lmcutnumeric search must not mark any state as a dead end"
    );
    assert_eq!(
        plan.len() as u64,
        blind_plan.len() as u64,
        "lmcutnumeric returned a plan of a different length than blind A*"
    );
    assert!(
        (solution_cost - OPTIMAL_COST).abs() <= 1e-6,
        "Plant Watering lmcutnumeric must keep the optimal cost {OPTIMAL_COST}, got {solution_cost}"
    );
}
