//! Main translation from STRIPS/PDDL ground representation to SAS+ finite-domain representation.
//!
//! This module assembles the task: it encodes facts and numeric fluents as
//! variables, finds the mutex groups and builds the [`SASTask`];
//! [`conditions`] translates conditions and [`operators`] the operators and
//! axioms.
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{NumericValue, ZERO_VALUE};

use tracing::info;

mod conditions;
mod operators;

use conditions::*;
use operators::*;

use super::axiom_rules;
use super::fact_groups;
use super::numeric_axiom_rules;
use super::options;
use super::pddl::actions::PropositionalAction;
use super::pddl::axioms::{InstantiatedNumericAxiom, PropositionalAxiom};
use super::pddl::conditions::*;
use super::pddl::f_expression::*;
use super::sas_tasks::*;
use super::simplify;
use super::tools;

/// Everything the translation of one task carries from step to step: the SAS
/// encoding it is filling in, the parallel encoding it tests conditions
/// against mutexes with, the comparison axioms it invents for numeric
/// conditions, and the two counters it reports at the end.
///
/// These travelled as seven to fourteen parameters through every function
/// below, always in the same order and always all together.
struct Translation {
    ranges: Vec<usize>,
    dictionary: HashMap<Atom, Vec<SasFact>>,
    numeric: HashMap<PrimitiveNumericExpression, usize>,
    /// One variable per mutex group, including the groups no SAS variable was
    /// built from, so that a condition can be tested against all of them.
    mutex_ranges: Vec<usize>,
    mutex_dictionary: HashMap<Atom, Vec<SasFact>>,
    /// The variable a `(comparator, operands)` comparison was given, so that
    /// the same comparison is not encoded twice.
    comparison_axioms: HashMap<(String, Vec<usize>), Condition>,
    sas_comparison_axioms: Vec<SASCompareAxiom>,
    /// The expression each SAS numeric variable is named after, one per
    /// variable. Its length is the number of SAS numeric variables.
    numeric_names: Vec<PrimitiveNumericExpression>,
    /// The human-readable name of each value of each variable.
    translation_key: Vec<Vec<String>>,
    /// One entry per mutex group, in SAS facts.
    mutex_key: Vec<Vec<SasFact>>,
    /// The value of each numeric variable that a constant numeric axiom
    /// defines, so that a comparison of two constants is decided here, as
    /// instantiation decides a static atom, instead of becoming a variable.
    constants: Vec<Option<NumericValue>>,
    /// The initial value of every numeric variable, for static costs.
    num_vals: Vec<NumericValue>,
    relevant_numeric: Vec<usize>,
    simplified_effect_conditions: usize,
}

// ============================================================
// the SAS encoding of facts and numeric fluents
// ============================================================

/// One SAS variable per group of mutually exclusive facts, plus a value for
/// "none of those", and the map from a fact to the variable/value pairs that
/// express it.
struct FactEncoding {
    ranges: Vec<usize>,
    dictionary: HashMap<Atom, Vec<SasFact>>,
}

fn encode_facts(groups: &[Vec<Atom>], assert_partial: bool) -> FactEncoding {
    let mut dictionary: HashMap<Atom, Vec<SasFact>> = HashMap::new();
    for (var_no, group) in groups.iter().enumerate() {
        for (val_no, atom) in group.iter().enumerate() {
            dictionary
                .entry(atom.clone())
                .or_default()
                .push((var_no, val_no));
        }
    }

    if assert_partial {
        for (atom, sas_pairs) in &dictionary {
            assert_eq!(
                sas_pairs.len(),
                1,
                "partial encoding covers {atom:?} by {} variables",
                sas_pairs.len()
            );
        }
    }

    FactEncoding {
        ranges: groups.iter().map(|group| group.len() + 1).collect(),
        dictionary,
    }
}

/// One SAS numeric variable per axiom effect that no equivalent axiom already
/// defines, then one per fluent none of them defines. An axiom made redundant
/// by an equivalent one shares that one's variable.
struct NumericEncoding {
    /// The expression a SAS numeric variable is named after: the effect of the
    /// axiom that defines it, or the fluent it stands for. Several expressions
    /// can map onto the same variable, so the name cannot be recovered from
    /// `variables` -- it is fixed here, where the variable is created.
    names: Vec<PrimitiveNumericExpression>,
    variables: HashMap<PrimitiveNumericExpression, usize>,
}

fn encode_numeric_fluents(
    num_axioms: &[InstantiatedNumericAxiom],
    equivalent: &HashMap<PrimitiveNumericExpression, PrimitiveNumericExpression>,
    num_fluents: &[PrimitiveNumericExpression],
) -> NumericEncoding {
    let mut variables: HashMap<PrimitiveNumericExpression, usize> = HashMap::new();
    let mut names: Vec<PrimitiveNumericExpression> = Vec::new();

    let mut redundant = vec![];
    for axiom in num_axioms {
        if equivalent.contains_key(&axiom.effect) {
            redundant.push(axiom.effect.clone());
        } else {
            let previous = variables.insert(axiom.effect.clone(), names.len());
            assert!(
                previous.is_none(),
                "numeric axiom effect {} defines a SAS variable twice",
                axiom.effect
            );
            names.push(axiom.effect.clone());
        }
    }
    for effect in &redundant {
        if let Some(kept) = equivalent.get(effect)
            && let Some(&variable) = variables.get(kept)
        {
            variables.insert(effect.clone(), variable);
        }
    }

    let mut fluents: Vec<PrimitiveNumericExpression> = num_fluents.to_vec();
    fluents.sort_by_cached_key(PrimitiveNumericExpression::to_string);
    for fluent in fluents {
        if let Entry::Vacant(entry) = variables.entry(fluent) {
            names.push(entry.key().clone());
            entry.insert(names.len() - 1);
        }
    }

    NumericEncoding { names, variables }
}

// ============================================================
// translate_task
// ============================================================

/// The grounded task an SAS+ encoding is built from.
struct GroundedTask<'a> {
    init: &'a [Atom],
    num_init: &'a [FunctionAssignment],
    goal_list: &'a [Condition],
    global_constraint: &'a Condition,
    actions: &'a [PropositionalAction],
    axioms: Vec<PropositionalAxiom>,
    metric: &'a (String, PrimitiveNumericExpression),
}

fn translate_task(
    translation: &mut Translation,
    task: GroundedTask,
    numeric_axioms: &numeric_axiom_rules::NumericAxioms,
    layer_strategy: options::LayerStrategy,
) -> Result<SASTask, String> {
    // Process axioms. Derived variables are not listed here: every one of them
    // defaults to false, which the closed-world assumption below already gives
    // them, and the axiom closure computes the real value on top of that.
    let (processed_axioms, axiom_layer_dict) = axiom_rules::handle_axioms(
        task.actions,
        task.axioms,
        task.goal_list,
        task.global_constraint,
        layer_strategy,
    )?;

    // Initialize init_values: Closed World Assumption
    let mut init_values: Vec<i32> = translation.ranges.iter().map(|&r| (r as i32) - 1).collect();
    for fact in task.init {
        if let Some(pairs) = translation.dictionary.get(fact) {
            for &(var, val) in pairs {
                let curr_val = init_values[var];
                if curr_val != (translation.ranges[var] as i32 - 1) && curr_val != val as i32 {
                    return Err(format!("Inconsistent init facts! [fact = {:?}]", fact));
                }
                init_values[var] = val as i32;
            }
        }
    }

    // A goal fact with no SAS variable is one the relaxation cannot reach, so the
    // task has no plan. This is checked here rather than inside
    // `translate_strips_conditions`, because that function's invariant holds for
    // an operator or an axiom and not for the goal: instantiation removes the
    // statically true and false atoms from a *condition*, and the goal is never
    // instantiated, so the goal is the one site that can carry an unreachable
    // atom this far.
    //
    // Fast Downward answers the same way. On `citycar-opt14-adl/p2-2-2-1-2` its
    // translator reports 2 facts and 0 operators, having found the same goal out
    // of reach.
    if let Some(unreachable) = task.goal_list.iter().find_map(|condition| match condition {
        Condition::Atom(atom) if !translation.dictionary.contains_key(atom) => Some(atom),
        _ => None,
    }) {
        return Ok(trivial_task(
            false,
            &format!("Goal fact {unreachable:?} is unreachable"),
        ));
    }

    // The goal, operators and axioms may compare constants, which are decided
    // while their conditions are translated.
    translation.constants = vec![None; translation.numeric_names.len()];
    for axiom in &numeric_axioms.constant {
        if let Some(&var) = translation.numeric.get(&axiom.effect) {
            let Some(FunctionalExpression::NumericConstant(nc)) = axiom.parts.first() else {
                unreachable!(
                    "constant numeric axiom {} must hold a folded numeric constant",
                    axiom.name
                );
            };
            translation.constants[var] = Some(nc.value);
        }
    }

    // Translate goal
    let goal_dict_list = translate_strips_conditions(translation, task.goal_list);

    // Translate global constraint
    let gc_as_list = vec![task.global_constraint.clone()];
    let global_constraint_dict_list = translate_strips_conditions(translation, &gc_as_list);

    let Some(goal_dict_list) = goal_dict_list else {
        return Ok(trivial_task(
            false,
            "Goal violates a mutex or a comparison of constants",
        ));
    };
    // A goal literal names one value of one variable: a positive one its own
    // fact's value, a negative one the only other value of the binary variable
    // that `compute_groups` reserved for it.
    assert_eq!(goal_dict_list.len(), 1, "the goal is a single assignment");

    let goal_pairs: Vec<(usize, usize)> = goal_dict_list[0].iter().collect();

    if goal_pairs.is_empty() {
        return Ok(trivial_task(true, "Empty goal"));
    }

    let sas_goal = SASGoal::new(goal_pairs);

    let global_constraint_dict_list =
        global_constraint_dict_list.expect("the global constraint translates to SAS+ conditions");
    assert_eq!(
        global_constraint_dict_list.len(),
        1,
        "the global constraint is one conjunction"
    );

    // Numeric init values.
    //
    // A fluent without a SAS numeric variable is skipped: it holds a constant,
    // which every expression mentioning it was folded against, so there is no
    // variable left for its initial value to name.
    let mut num_init_values = vec![ZERO_VALUE; translation.numeric_names.len()];

    let mut relevant_numeric: Vec<usize> = vec![];
    for fact in task.num_init {
        let Some(&var) = translation.numeric.get(&fact.fluent) else {
            continue;
        };
        let FunctionalExpression::NumericConstant(nc) = &fact.expression else {
            return Err(format!(
                "numeric init fact for {} must assign a numeric constant, got {}",
                fact.fluent, fact.expression
            ));
        };
        num_init_values[var] = nc.value;
        if fact.fluent.kind == FluentKind::Regular {
            relevant_numeric.push(var);
        }
    }

    // Fold the constant numeric axioms into the initial values before the
    // operators are translated: `sas_operator_cost` evaluates a non-constant
    // action cost in the initial state, and normalization rewrites every
    // compound cost expression into a derived function backed by such an
    // axiom. Constant axiom effects are derived fluents and therefore never
    // collide with the `:init` facts above.
    for (value, constant) in num_init_values.iter_mut().zip(&translation.constants) {
        if let Some(constant) = constant {
            *value = *constant;
        }
    }

    // Translate operators
    translation.num_vals = num_init_values;
    translation.relevant_numeric = relevant_numeric;
    let operators = translate_strips_operators(translation, task.actions);

    // Translate axioms
    let sas_axioms = translate_strips_axioms(translation, &processed_axioms);

    // Translate numeric axioms
    let const_num_axiom_effects: HashSet<PrimitiveNumericExpression> = numeric_axioms
        .constant
        .iter()
        .map(|ax| ax.effect.clone())
        .collect();
    let sas_num_axioms: Vec<SASNumericAxiom> = numeric_axioms
        .axioms
        .iter()
        .filter(|ax| {
            !const_num_axiom_effects.contains(&ax.effect)
                && !numeric_axioms.equivalent.contains_key(&ax.effect)
        })
        .filter_map(|ax| translate_numeric_axiom(ax, &translation.dictionary, &translation.numeric))
        .collect();

    // Compute axiom layers
    let mut axiom_layers: Vec<i32> = vec![-1; translation.ranges.len()];
    let mut num_axiom_layers: Vec<i32> = vec![-1; translation.numeric_names.len()];
    let mut num_axiom_layer = 0i32;

    for (&layer, layer_axioms) in &numeric_axioms.by_layer {
        let mut sorted_axioms = layer_axioms.clone();
        sorted_axioms.sort_by(|a, b| a.name.cmp(&b.name));
        for axiom in &sorted_axioms {
            if !numeric_axioms.equivalent.contains_key(&axiom.effect)
                && let Some(&var) = translation.numeric.get(&axiom.effect)
            {
                if layer == -1 {
                    num_axiom_layers[var] = -1;
                } else {
                    num_axiom_layers[var] = num_axiom_layer;
                    num_axiom_layer += 1;
                }
            }
        }
    }

    // A comparison variable's initial-state entry is its axiom default, and the
    // comparison axiom overwrites it with a verdict before anything reads it.
    // `False` is the default that says what the closure has not yet proven.
    let comp_axiom_default = ConditionValue::False.as_usize() as i32;
    init_values.extend(std::iter::repeat_n(
        comp_axiom_default,
        translation.sas_comparison_axioms.len(),
    ));

    for axiom in &translation.sas_comparison_axioms {
        axiom_layers[axiom.effect] = num_axiom_layer;
    }

    for (atom, &layer) in &axiom_layer_dict {
        assert!(layer >= 0);
        // An atom with no SAS variable was pruned as static or unreachable.
        // There is no variable to carry a layer, so there is nothing to record.
        if let Some(pairs) = translation.dictionary.get(atom) {
            let (var, _val) = pairs[0];
            axiom_layers[var] = layer + num_axiom_layer + 1;
        }
    }

    // Extend axiom_layers for comparison axiom variables
    while axiom_layers.len() < translation.ranges.len() {
        axiom_layers.push(num_axiom_layer);
    }

    add_key_to_comp_axioms(
        &translation.sas_comparison_axioms.clone(),
        &mut translation.translation_key,
    );

    let variables = SASVariables::new(
        translation.ranges.clone(),
        axiom_layers,
        translation.translation_key.clone(),
        num_axiom_layer,
    );

    // Name each numeric variable after the expression it was created for. The
    // `numeric` map cannot serve here: equivalent axiom effects share a
    // variable, so it holds several names per variable.
    let num_variables: Vec<String> = translation
        .numeric_names
        .iter()
        .map(|entry| format!("{}", entry))
        .collect();
    let num_var_types: Vec<String> = translation
        .numeric_names
        .iter()
        .map(|entry| entry.kind.sas_letter().to_string())
        .collect();

    let numeric_variables =
        SASNumericVariables::new(num_variables, num_axiom_layers, num_var_types);

    let mutexes: Vec<SASMutexGroup> = translation
        .mutex_key
        .iter()
        .map(|group| SASMutexGroup::new(group.clone()))
        .collect();

    let sas_init = SASInit::new(init_values, translation.num_vals.clone());

    // Look up task.metric fluent
    let sas_metric = if task.metric.1.symbol.is_empty() {
        // Unit cost
        (task.metric.0.clone(), -1i64)
    } else {
        let idx = translation.numeric.get(&task.metric.1).unwrap_or_else(|| {
            panic!(
                "metric fluent {} has no translated numeric variable",
                task.metric.1
            )
        });
        (task.metric.0.clone(), *idx as i64)
    };

    // The global constraint is a single atom (asserted by the caller) and
    // `USE_PARTIAL_ENCODING` maps every atom to exactly one SAS fact, so the
    // translated condition holds exactly one pair. Picking an arbitrary pair
    // out of a larger map would silently install a different constraint, and
    // an empty map has no pair to pick at all.
    let global_constraint_facts = &global_constraint_dict_list[0];
    assert_eq!(
        global_constraint_facts.len(),
        1,
        "global constraint must translate to exactly one SAS fact, got {global_constraint_facts:?}"
    );
    let gc_pair = global_constraint_facts
        .iter()
        .next()
        .expect("length checked above");

    let mut sas_task = SASTask {
        variables,
        numeric_variables,
        mutexes,
        init: sas_init,
        goal: sas_goal,
        operators,
        axioms: sas_axioms,
        comp_axioms: std::mem::take(&mut translation.sas_comparison_axioms),
        numeric_axioms: sas_num_axioms,
        global_constraint: gc_pair,
        metric: sas_metric,
    };
    sas_task.canonicalize();
    Ok(sas_task)
}

// ============================================================
// trivial_task
// ============================================================

fn trivial_task(solvable: bool, msg: &str) -> SASTask {
    if solvable {
        info!("{}! Generating solvable task...", msg);
    } else {
        info!("{}! Generating unsolvable task...", msg);
    }
    simplify::trivial_task(solvable)
}

// ============================================================
// build_mutex_key
// ============================================================

fn build_mutex_key(
    strips_to_sas: &HashMap<Atom, Vec<(usize, usize)>>,
    groups: &[Vec<Atom>],
) -> Vec<Vec<(usize, usize)>> {
    let mut group_keys = vec![];
    for group in groups {
        let mut group_key = vec![];
        for fact in group {
            if let Some(pairs) = strips_to_sas.get(fact) {
                for &(var, val) in pairs {
                    group_key.push((var, val));
                }
            } else {
                // PARITY(numeric-fd): mutex facts absent from the strips-to-SAS
                // map are unreachable and are intentionally omitted.
                info!("not in strips_to_sas, left out: {:?}", fact);
            }
        }
        group_keys.push(group_key);
    }
    group_keys
}

// ============================================================
// Main entry point: pddl_to_sas / translate_task_from_grounded_internal
// ============================================================

/// Called from main.rs as translate_task_from_grounded_internal
/// Encodes a grounded task as SAS+.
///
/// `singleton_groups` skips invariant synthesis and gives every fact its own
/// variable: a larger encoding of the same task.
pub fn translate_grounded_task(
    explored: &crate::instantiate::ExploreResult,
    task: &crate::pddl::tasks::Task,
    singleton_groups: bool,
    layer_strategy: options::LayerStrategy,
) -> Result<SASTask, String> {
    let crate::instantiate::ExploreResult {
        atoms,
        num_fluents,
        grounded_ops,
        grounded_axioms,
        numeric_axioms: num_axioms,
        reachable_action_params,
        ..
    } = explored;
    let goal = &task.goal;

    fn merge_numeric_fluent_type(
        merged: &mut HashMap<(String, Vec<String>), PrimitiveNumericExpression>,
        pne: PrimitiveNumericExpression,
    ) {
        let key = (pne.symbol.clone(), pne.args.clone());
        match merged.get(&key) {
            Some(existing) if existing.kind.merge_rank() >= pne.kind.merge_rank() => {}
            _ => {
                merged.insert(key, pne);
            }
        }
    }

    let mut merged_num_fluents: HashMap<(String, Vec<String>), PrimitiveNumericExpression> =
        HashMap::new();
    for fluent in num_fluents {
        merge_numeric_fluent_type(&mut merged_num_fluents, fluent.clone());
    }
    merge_numeric_fluent_type(&mut merged_num_fluents, task.metric.1.clone());

    let mut num_fluents_vec: Vec<PrimitiveNumericExpression> =
        merged_num_fluents.into_values().collect();
    num_fluents_vec.sort_by(|left, right| {
        left.symbol
            .cmp(&right.symbol)
            .then_with(|| left.args.cmp(&right.args))
            .then_with(|| left.kind.cmp(&right.kind))
    });

    // Build goal list
    let goal_list: Vec<Condition> = match goal {
        Condition::Conjunction(conj) => conj.parts.clone(),
        other => vec![other.clone()],
    };

    // `substitute_complicated_goal` has left a conjunction of facts, so every
    // conjunct is a propositional literal or a numeric comparison. Only a
    // negative literal needs anything here: its atom must keep a variable with a
    // value standing for "absent", which is what `compute_groups` uses this set
    // for. A comparison already has a two-valued variable of its own.
    let mut negative_in_goal: HashSet<Atom> = HashSet::new();
    for item in &goal_list {
        match item {
            Condition::Atom(_)
            | Condition::FunctionComparison(_)
            | Condition::NegatedFunctionComparison(_) => {}
            Condition::NegatedAtom(negated) => {
                negative_in_goal.insert(Atom::new(negated.predicate.clone(), negated.args.clone()));
            }
            _ => return Err(format!("Non-literal goal: {:?}", item)),
        }
    }

    // Compute fact groups
    let atoms_set: HashSet<Atom> = atoms.iter().cloned().collect();
    let fact_groups::FactGroups {
        groups,
        mutex_groups,
        translation_key,
    } = if singleton_groups {
        // Fast path: skip invariant finding / mutex discovery. Every fact
        // already gets a binary variable, so a negative goal needs nothing
        // extra. This preserves semantics but produces a less compact encoding.
        fact_groups::compute_singleton_groups(&atoms_set)
    } else {
        fact_groups::compute_groups(task, &atoms_set, reachable_action_params, &negative_in_goal)
    };

    let numeric_axioms = numeric_axiom_rules::handle_axioms(num_axioms);

    let facts = encode_facts(&groups, true);
    let numeric = encode_numeric_fluents(
        &numeric_axioms.axioms,
        &numeric_axioms.equivalent,
        &num_fluents_vec,
    );
    // The mutex groups overlap, so they get their own variables, used only to
    // test whether a condition violates a mutex.
    let mutex_facts = encode_facts(&mutex_groups, false);

    let mutex_key = build_mutex_key(&facts.dictionary, &mutex_groups);

    let mut translation = Translation {
        ranges: facts.ranges,
        dictionary: facts.dictionary,
        numeric: numeric.variables,
        numeric_names: numeric.names,
        mutex_ranges: mutex_facts.ranges,
        mutex_dictionary: mutex_facts.dictionary,
        mutex_key,
        translation_key,
        comparison_axioms: HashMap::new(),
        sas_comparison_axioms: Vec::new(),
        constants: Vec::new(),
        num_vals: Vec::new(),
        relevant_numeric: Vec::new(),
        simplified_effect_conditions: 0,
    };

    let gc = &task.global_constraint;
    assert!(
        matches!(gc, Condition::Atom(_)),
        "Global constraint must be an atom literal"
    );

    let sas_task = translate_task(
        &mut translation,
        GroundedTask {
            init: &task.init,
            num_init: &task.num_init,
            goal_list: &goal_list,
            global_constraint: gc,
            actions: grounded_ops,
            axioms: grounded_axioms.to_vec(),
            metric: &task.metric,
        },
        &numeric_axioms,
        layer_strategy,
    )?;

    info!(
        "{} effect conditions simplified",
        translation.simplified_effect_conditions
    );

    let mut sas_task = sas_task;
    match simplify::filter_unreachable_propositions(&mut sas_task) {
        Ok(()) => Ok(sas_task),
        // Naming every variant rather than using a wildcard makes a future
        // variant a compile error instead of a silent fallthrough:
        // `filter_unreachable_propositions` renames the task in place, so
        // returning `sas_task` for an unhandled error would hand back a
        // half-rewritten task.
        Err(simplify::SimplifyError::Impossible) => Ok(simplify::trivial_task(false)),
        Err(simplify::SimplifyError::TriviallySolvable) => Ok(simplify::trivial_task(true)),
    }
}

/// Extension trait for FunctionalExpression
impl FunctionalExpression {
    pub fn as_pne(&self) -> Option<&PrimitiveNumericExpression> {
        match self {
            FunctionalExpression::PrimitiveNumericExpression(pne) => Some(pne),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pne(symbol: &str) -> PrimitiveNumericExpression {
        PrimitiveNumericExpression::new(symbol.to_string(), vec![])
    }

    fn translation_with_numeric(
        numeric: impl IntoIterator<Item = PrimitiveNumericExpression>,
    ) -> Translation {
        let numeric: HashMap<_, _> = numeric
            .into_iter()
            .enumerate()
            .map(|(index, expression)| (expression, index))
            .collect();
        let numeric_count = numeric.len();
        Translation {
            ranges: vec![],
            dictionary: HashMap::new(),
            numeric,
            mutex_ranges: vec![],
            mutex_dictionary: HashMap::new(),
            comparison_axioms: HashMap::new(),
            sas_comparison_axioms: vec![],
            numeric_names: vec![],
            translation_key: vec![],
            mutex_key: vec![],
            constants: vec![None; numeric_count],
            num_vals: vec![ZERO_VALUE; numeric_count],
            relevant_numeric: (0..numeric_count).collect(),
            simplified_effect_conditions: 0,
        }
    }

    fn action_with_assignment(assignment: FunctionAssignment) -> PropositionalAction {
        PropositionalAction {
            name: "broken-action".to_string(),
            precondition: vec![],
            add_effects: vec![],
            del_effects: vec![],
            assign_effects: vec![(vec![], assignment)],
            cost: None,
        }
    }

    #[test]
    #[should_panic(expected = "numeric effect of operator broken-action has non-PNE expression")]
    fn rejects_non_normalized_numeric_effect_expression() {
        let fluent = pne("fuel");
        let mut translation = translation_with_numeric([fluent.clone()]);
        let assignment = FunctionAssignment::new(
            "+".to_string(),
            fluent,
            FunctionalExpression::ArithmeticExpression(ArithmeticExpression::new(
                "+".to_string(),
                vec![FunctionalExpression::NumericConstant(NumericConstant::new(
                    1.0,
                ))],
            )),
        );

        translate_strips_operator(&mut translation, &action_with_assignment(assignment));
    }

    #[test]
    #[should_panic(expected = "numeric effect expression PNE missing() of operator broken-action")]
    fn rejects_missing_numeric_effect_expression_variable() {
        let fluent = pne("fuel");
        let mut translation = translation_with_numeric([fluent.clone()]);
        let assignment = FunctionAssignment::new(
            "+".to_string(),
            fluent,
            FunctionalExpression::PrimitiveNumericExpression(pne("missing")),
        );

        translate_strips_operator(&mut translation, &action_with_assignment(assignment));
    }

    #[test]
    #[should_panic(expected = "numeric effect fluent PNE missing() of operator broken-action")]
    fn rejects_missing_numeric_effect_fluent_variable() {
        let expression = pne("amount");
        let mut translation = translation_with_numeric([expression.clone()]);
        let assignment = FunctionAssignment::new(
            "+".to_string(),
            pne("missing"),
            FunctionalExpression::PrimitiveNumericExpression(expression),
        );

        translate_strips_operator(&mut translation, &action_with_assignment(assignment));
    }

    #[test]
    fn permits_pruned_non_metric_cost_fluent() {
        let expression = pne("amount");
        let mut translation = translation_with_numeric([expression.clone()]);
        let action = PropositionalAction {
            name: "broken-action".to_string(),
            precondition: vec![],
            add_effects: vec![],
            del_effects: vec![],
            assign_effects: vec![],
            cost: Some(FunctionAssignment::new(
                "+".to_string(),
                PrimitiveNumericExpression::total_cost(),
                FunctionalExpression::PrimitiveNumericExpression(expression),
            )),
        };

        assert!(translate_strips_operator(&mut translation, &action).is_empty());
    }
}
