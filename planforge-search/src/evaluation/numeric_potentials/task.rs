use std::collections::{BTreeMap, BTreeSet};

use planforge_sas::axioms::{CalOperator, ComparisonOperator};
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentOperation, ExplicitFact, NumericType,
    metric_operator_cost_from_initial_values,
};
use planforge_sas::state_registry::{ConcreteState, StateRegistry};

use super::BoundsProvider;
use crate::evaluation::numeric_landmarks::numeric_helper::{
    LinearNumericCondition, NumericTaskHelper,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureBounds {
    pub lower: f64,
    pub upper: f64,
}

impl Default for FeatureBounds {
    fn default() -> Self {
        Self {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NumericFeature {
    pub name: String,
    /// Sorted `(numeric_variable_id, coefficient)` entries.
    pub coefficients: Vec<(usize, f64)>,
    pub constant: f64,
    pub affine: bool,
    pub source_numeric_var: usize,
}

#[derive(Debug, Clone)]
pub struct PotentialOperator {
    pub preconditions: Vec<ExplicitFact>,
    pub effects: Vec<(usize, usize)>,
    /// Sorted `(feature_id, delta)` entries. Zero deltas are omitted.
    pub numeric_deltas: Vec<(usize, f64)>,
    /// Sorted `(feature_id, bounds)` entries. Unbounded intervals are omitted.
    pub numeric_precondition_bounds: Vec<(usize, FeatureBounds)>,
    pub cost: f64,
    pub reachable: bool,
}

impl PotentialOperator {
    pub fn numeric_delta(&self, feature_id: usize) -> f64 {
        self.numeric_deltas
            .binary_search_by_key(&feature_id, |(id, _)| *id)
            .map(|index| self.numeric_deltas[index].1)
            .unwrap_or(0.0)
    }
}

#[derive(Debug, Clone)]
pub struct LinearEquality {
    pub coefficients: Vec<(usize, f64)>,
    pub rhs: f64,
}

#[derive(Debug, Clone)]
pub struct PotentialTask {
    pub numeric_variable_count: usize,
    pub domain_sizes: Vec<usize>,
    pub derived_propositional: Vec<bool>,
    pub propositional_goals: Vec<ExplicitFact>,
    pub features: Vec<NumericFeature>,
    pub feature_goal_bounds: Vec<FeatureBounds>,
    pub ray_feature_goal_bounds: Vec<FeatureBounds>,
    pub assignment_target_features: Vec<bool>,
    pub global_linear_equalities: Vec<LinearEquality>,
    pub global_feature_bounds: Vec<FeatureBounds>,
    pub reachable_facts: Option<Vec<Vec<bool>>>,
    pub operators: Vec<PotentialOperator>,
    pub goal_unsatisfiable: bool,
}

impl PotentialTask {
    pub fn build(
        task: &dyn AbstractNumericTask,
        precision: f64,
        epsilon: f64,
        ignore_numeric_variables: bool,
        bounds_provider: BoundsProvider,
        simple_action_bounds: bool,
    ) -> Result<Self, String> {
        for (operator_id, operator) in task.get_operators().iter().enumerate() {
            for effect in operator.effects() {
                if !effect.conditions().is_empty() {
                    return Err(format!(
                        "numeric_potential does not support conditional propositional effects (operator {operator_id}, `{}`)",
                        operator.name()
                    ));
                }
            }
            for effect in operator.assignment_effects() {
                if effect.is_conditional() || !effect.conditions().is_empty() {
                    return Err(format!(
                        "numeric_potential does not support conditional numeric effects (operator {operator_id}, `{}`)",
                        operator.name()
                    ));
                }
            }
        }

        let helper =
            NumericTaskHelper::new_potentials(task, precision, epsilon, simple_action_bounds);
        let domain_sizes = task
            .variables()
            .iter()
            .map(|var| var.domain_size())
            .collect();
        let derived_propositional: Vec<bool> = task
            .variables()
            .iter()
            .map(|var| var.axiom_layer().is_some())
            .collect();

        let mut propositional_goals = Vec::new();
        let mut numeric_conditions = Vec::new();
        let mut expanded_goal_facts = BTreeSet::new();
        for goal_id in 0..task.get_num_goals() {
            let goal = task.get_goal_fact(goal_id);
            expand_cpp_goal_fact(
                task,
                &helper,
                *goal,
                &derived_propositional,
                &mut expanded_goal_facts,
                &mut propositional_goals,
                &mut numeric_conditions,
            )?;
        }
        propositional_goals.sort_unstable();
        propositional_goals.dedup();
        // A propositional goal that is mutex with the false value of a
        // comparison axiom implies that comparison in every concrete goal
        // state. Preserve this numeric-fd goal invariant as an additional
        // numeric goal interval.
        for prop_goal in &propositional_goals {
            for comparison in task.comparison_axioms() {
                let false_fact = ExplicitFact::propositional(comparison.get_affected_var_id(), 1);
                if prop_goal.var() != false_fact.var()
                    && task.are_facts_mutex(prop_goal, &false_fact)
                {
                    numeric_conditions.extend(helper.comparison_fact_materialized_conditions(
                        comparison.get_affected_var_id(),
                        0,
                    ));
                }
            }
        }

        let numeric_count = task.numeric_variables().len();
        // Classical-only mode keeps the complete C++ matrix shape: numeric
        // weight columns and the numeric goal-awareness row still exist, but
        // all weights are fixed to zero because no numeric goal interval is
        // registered below.
        let features = build_cpp_numeric_features(task)?;
        let mut feature_keys: BTreeMap<Vec<(usize, u64)>, usize> = BTreeMap::new();
        for (feature_id, feature) in features.iter().enumerate() {
            if !feature.affine || feature.coefficients.is_empty() {
                continue;
            }
            let (canonical, _) = canonical_feature(&feature.coefficients)
                .expect("nonconstant affine feature must be canonicalizable");
            feature_keys.insert(feature_key(&canonical), feature_id);
        }

        let mut feature_goal_bounds = vec![FeatureBounds::default(); features.len()];
        let mut goal_unsatisfiable = false;
        if !ignore_numeric_variables {
            let integral = all_values_integral(
                task,
                &features,
                &feature_keys,
                &numeric_conditions,
                precision,
            )?;
            for condition in &numeric_conditions {
                register_condition_bound(
                    condition,
                    &features,
                    &feature_keys,
                    &mut feature_goal_bounds,
                    integral,
                    epsilon,
                    &mut goal_unsatisfiable,
                )?;
            }
        }

        let mut assignment_targets = vec![false; numeric_count];
        let mut primitive_deltas = Vec::with_capacity(task.get_operators().len());
        for (operator_id, operator) in task.get_operators().iter().enumerate() {
            let linearized = task
                .linearized_assignment_effects(operator_id)
                .map_err(|error| {
                    format!(
                        "numeric_potential cannot linearize numeric effects of operator {operator_id} (`{}`): {error}",
                        operator.name()
                    )
                })?;
            let mut deltas = BTreeMap::<usize, f64>::new();
            for effect in linearized {
                match effect.operation {
                    AssignmentOperation::Assign => {
                        assignment_targets[effect.affected_var_id] = true;
                    }
                    AssignmentOperation::Plus | AssignmentOperation::Minus => {
                        if !effect.delta.is_constant() {
                            return Err(format!(
                                "numeric_potential requires state-independent additive numeric effects; operator {operator_id} (`{}`) has a state-dependent effect on numeric variable {}",
                                operator.name(),
                                effect.affected_var_id
                            ));
                        }
                        *deltas.entry(effect.affected_var_id).or_default() += effect.delta.constant;
                    }
                    AssignmentOperation::Times | AssignmentOperation::Divide => {
                        return Err(format!(
                            "numeric_potential does not support nonlinear numeric effects (operator {operator_id}, `{}`)",
                            operator.name()
                        ));
                    }
                }
            }
            primitive_deltas.push(
                deltas
                    .into_iter()
                    .filter(|(_, delta)| *delta != 0.0)
                    .collect::<Vec<_>>(),
            );
        }

        let assignment_target_features: Vec<bool> = features
            .iter()
            .map(|feature| {
                !feature.affine
                    || feature
                        .coefficients
                        .iter()
                        .any(|(variable_id, _)| assignment_targets[*variable_id])
            })
            .collect();
        let initial_numeric = task.get_initial_numeric_state_values();
        let mut global_linear_equalities = Vec::new();
        for lhs in 0..numeric_count {
            if task.numeric_variables()[lhs].get_type() != &NumericType::Regular
                || assignment_targets[lhs]
            {
                continue;
            }
            for rhs in (lhs + 1)..numeric_count {
                if task.numeric_variables()[rhs].get_type() != &NumericType::Regular
                    || assignment_targets[rhs]
                {
                    continue;
                }
                let identical = primitive_deltas
                    .iter()
                    .all(|deltas| sparse_value(deltas, lhs) == sparse_value(deltas, rhs));
                let opposite = primitive_deltas
                    .iter()
                    .all(|deltas| sparse_value(deltas, lhs) == -sparse_value(deltas, rhs));
                let has_nonzero_delta = primitive_deltas.iter().any(|deltas| {
                    sparse_value(deltas, lhs) != 0.0 || sparse_value(deltas, rhs) != 0.0
                });
                if !has_nonzero_delta || (!identical && !opposite) {
                    continue;
                }
                let rhs_coefficient = if identical { -1.0 } else { 1.0 };
                let coefficients = vec![(lhs, 1.0), (rhs, rhs_coefficient)];
                global_linear_equalities.push(LinearEquality {
                    rhs: initial_numeric[lhs] + rhs_coefficient * initial_numeric[rhs],
                    coefficients,
                });
            }
        }

        let operators = task
            .get_operators()
            .iter()
            .enumerate()
            .map(|(operator_id, operator)| {
                let mut numeric_precondition_bounds =
                    vec![FeatureBounds::default(); features.len()];
                for precondition in operator.preconditions() {
                    for condition in helper.comparison_fact_materialized_conditions(
                        precondition.var(),
                        precondition.value(),
                    ) {
                        register_closed_condition_bound(
                            &condition,
                            &features,
                            &feature_keys,
                            &mut numeric_precondition_bounds,
                        )?;
                    }
                }
                Ok(PotentialOperator {
                    preconditions: operator.preconditions().clone(),
                    effects: operator
                        .effects()
                        .iter()
                        .map(|effect| (effect.var_id(), effect.value()))
                        .collect(),
                    numeric_deltas: features
                        .iter()
                        .enumerate()
                        .filter_map(|(feature_id, feature)| {
                            if feature.affine {
                                let delta = dot_sparse(
                                    &feature.coefficients,
                                    &primitive_deltas[operator_id],
                                );
                                (delta != 0.0).then_some((feature_id, delta))
                            } else {
                                None
                            }
                        })
                        .collect(),
                    numeric_precondition_bounds: sparse_bounds(numeric_precondition_bounds),
                    cost: metric_operator_cost_from_initial_values(task, operator),
                    reachable: true,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        // Rays keep explicit goal endpoints unchanged. For a goal-free
        // resource, however, C++ falls back to the merged reachable bounds so
        // the homogeneous system can retain the corresponding kappa
        // certificate instead of pinning its weight to zero.
        let raw_ray_feature_goal_bounds = feature_goal_bounds.clone();
        let mut global_feature_bounds = vec![FeatureBounds::default(); features.len()];
        if simple_action_bounds {
            let mut primitive_bounds = vec![FeatureBounds::default(); numeric_count];
            for (local_id, task_id) in task.regular_numeric_variable_ids().into_iter().enumerate() {
                if assignment_targets[task_id] {
                    continue;
                }
                if let Some((lower, upper)) = helper.numeric_variable_bounds(local_id) {
                    primitive_bounds[task_id] = FeatureBounds {
                        lower: if lower > -9_999_999.0 {
                            lower
                        } else {
                            f64::NEG_INFINITY
                        },
                        upper: if upper < 9_999_999.0 {
                            upper
                        } else {
                            f64::INFINITY
                        },
                    };
                }
            }
            for (feature_id, feature) in features.iter().enumerate() {
                if feature.affine
                    && feature.coefficients.len() == 1
                    && feature.constant == 0.0
                    && task.numeric_variables()[feature.source_numeric_var].get_type()
                        == &NumericType::Regular
                {
                    global_feature_bounds[feature_id] =
                        primitive_bounds[feature.source_numeric_var];
                }
            }
        }
        if matches!(
            bounds_provider,
            BoundsProvider::Monotone | BoundsProvider::All
        ) {
            let initial = task.get_initial_numeric_state_values();
            for (feature_id, feature) in features.iter().enumerate() {
                if assignment_target_features[feature_id] {
                    continue;
                }
                let all_nonnegative = primitive_deltas
                    .iter()
                    .all(|deltas| dot_sparse(&feature.coefficients, deltas) >= 0.0);
                let all_nonpositive = primitive_deltas
                    .iter()
                    .all(|deltas| dot_sparse(&feature.coefficients, deltas) <= 0.0);
                let initial_value = if feature.affine {
                    dot_sparse_dense(&feature.coefficients, &initial) + feature.constant
                } else {
                    initial[feature.source_numeric_var]
                };
                if all_nonnegative {
                    global_feature_bounds[feature_id].lower = initial_value;
                }
                if all_nonpositive {
                    global_feature_bounds[feature_id].upper = initial_value;
                }
            }
        }
        let mut reachable_facts = None;
        if matches!(bounds_provider, BoundsProvider::Aibr | BoundsProvider::All) {
            let closure = compute_aibr_closure(task)?;
            let mut contribution = vec![FeatureBounds::default(); features.len()];
            map_primitive_bounds_to_features(&features, &closure.numeric_bounds, &mut contribution);
            intersect_bounds(&mut global_feature_bounds, &contribution);
            reachable_facts = Some(closure.reachable_facts);
        }
        intersect_bounds(&mut feature_goal_bounds, &global_feature_bounds);
        let ray_feature_goal_bounds = raw_ray_feature_goal_bounds
            .into_iter()
            .zip(&feature_goal_bounds)
            .map(|(raw, merged)| {
                if raw.lower.is_finite() || raw.upper.is_finite() {
                    raw
                } else {
                    *merged
                }
            })
            .collect();
        goal_unsatisfiable |= feature_goal_bounds
            .iter()
            .any(|bounds| bounds.lower > bounds.upper);
        let mut operators = operators;
        if !matches!(bounds_provider, BoundsProvider::None) || simple_action_bounds {
            for operator in &mut operators {
                operator.reachable = operator.preconditions.iter().all(|fact| {
                    reachable_facts.as_ref().is_none_or(|facts| {
                        facts
                            .get(fact.var())
                            .and_then(|values| values.get(fact.value()))
                            .copied()
                            .unwrap_or(true)
                    })
                }) && operator.numeric_precondition_bounds.iter().all(
                    |(feature_id, precondition)| {
                        let global = global_feature_bounds[*feature_id];
                        precondition.lower <= global.upper && precondition.upper >= global.lower
                    },
                );
                operator.numeric_precondition_bounds = intersect_sparse_bounds(
                    &operator.numeric_precondition_bounds,
                    &global_feature_bounds,
                );
            }
        }
        if let Some(facts) = &reachable_facts {
            goal_unsatisfiable |= propositional_goals.iter().any(|goal| {
                !facts
                    .get(goal.var())
                    .and_then(|values| values.get(goal.value()))
                    .copied()
                    .unwrap_or(true)
            });
        }

        Ok(Self {
            numeric_variable_count: numeric_count,
            domain_sizes,
            derived_propositional,
            propositional_goals,
            features,
            feature_goal_bounds,
            ray_feature_goal_bounds,
            assignment_target_features,
            global_linear_equalities,
            global_feature_bounds,
            reachable_facts,
            operators,
            goal_unsatisfiable,
        })
    }

    pub fn feature_values(
        &self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
        numeric_scratch: &mut Vec<f64>,
    ) -> Result<Vec<f64>, String> {
        registry
            .fill_numeric_vars(state, numeric_scratch)
            .map_err(|error| format!("failed to unpack numeric state: {error:?}"))?;
        Ok(self
            .features
            .iter()
            .map(|feature| {
                if feature.affine {
                    dot_sparse_dense(&feature.coefficients, numeric_scratch) + feature.constant
                } else {
                    numeric_scratch[feature.source_numeric_var]
                }
            })
            .collect())
    }
}

#[derive(Clone)]
struct ParsedNumericExpression {
    coefficients: Vec<(usize, f64)>,
    constant: f64,
    affine: bool,
    is_constant: bool,
}

fn expand_cpp_goal_fact(
    task: &dyn AbstractNumericTask,
    helper: &NumericTaskHelper,
    fact: ExplicitFact,
    derived_propositional: &[bool],
    expanded: &mut BTreeSet<(usize, usize)>,
    propositional_goals: &mut Vec<ExplicitFact>,
    numeric_conditions: &mut Vec<LinearNumericCondition>,
) -> Result<(), String> {
    if !expanded.insert((fact.var(), fact.value())) {
        return Ok(());
    }
    let direct = helper.comparison_fact_materialized_conditions(fact.var(), fact.value());
    if !direct.is_empty() {
        numeric_conditions.extend(direct);
        return Ok(());
    }
    if !derived_propositional
        .get(fact.var())
        .copied()
        .unwrap_or(false)
    {
        propositional_goals.push(fact);
        return Ok(());
    }

    let achievers = task
        .axioms()
        .iter()
        .filter(|axiom| {
            axiom.var_id() == fact.var()
                && axiom.effect_value() == fact.value()
                && !axiom.conditions().is_empty()
        })
        .collect::<Vec<_>>();
    if achievers.len() != 1 {
        return Err(format!(
            "numeric_potential requires a derived goal helper to have exactly one nonempty defining axiom; fact {}={} has {}",
            fact.var(),
            fact.value(),
            achievers.len()
        ));
    }
    for condition in achievers[0].conditions() {
        expand_cpp_goal_fact(
            task,
            helper,
            *condition,
            derived_propositional,
            expanded,
            propositional_goals,
            numeric_conditions,
        )?;
    }
    Ok(())
}

fn build_cpp_numeric_features(
    task: &dyn AbstractNumericTask,
) -> Result<Vec<NumericFeature>, String> {
    let mut features = Vec::new();
    for (numeric_var_id, variable) in task.numeric_variables().iter().enumerate() {
        if variable.get_type() != &NumericType::Regular {
            continue;
        }
        features.push(NumericFeature {
            name: variable.name().to_string(),
            coefficients: vec![(numeric_var_id, 1.0)],
            constant: 0.0,
            affine: true,
            source_numeric_var: numeric_var_id,
        });
    }

    let comparison_by_var: BTreeMap<usize, usize> = task
        .comparison_axioms()
        .iter()
        .enumerate()
        .map(|(comparison_id, comparison)| (comparison.get_affected_var_id(), comparison_id))
        .collect();
    let mut used_comparison_vars = Vec::new();
    let mut seen_comparisons = BTreeSet::new();
    {
        let mut record_fact = |fact: &ExplicitFact| {
            if comparison_by_var.contains_key(&fact.var()) && seen_comparisons.insert(fact.var()) {
                used_comparison_vars.push(fact.var());
            }
        };
        // This is the C++ numeric_pdb_helper construction order: operator
        // preconditions first, then the non-dummy propositional goal axioms.
        for operator in task.get_operators() {
            for precondition in operator.preconditions() {
                record_fact(precondition);
            }
        }
        for axiom in task.axioms() {
            if !axiom.conditions().is_empty() {
                for condition in axiom.conditions() {
                    record_fact(condition);
                }
            }
        }
    }
    // The optimizer also materializes comparison facts implied by a
    // propositional-goal mutex.
    for goal_id in 0..task.get_num_goals() {
        let goal = task.get_goal_fact(goal_id);
        for comparison in task.comparison_axioms() {
            let false_fact = ExplicitFact::propositional(comparison.get_affected_var_id(), 1);
            if goal.var() != false_fact.var()
                && task.are_facts_mutex(goal, &false_fact)
                && seen_comparisons.insert(comparison.get_affected_var_id())
            {
                used_comparison_vars.push(comparison.get_affected_var_id());
            }
        }
    }

    let assignment_by_var: BTreeMap<usize, usize> = task
        .assignment_axioms()
        .iter()
        .enumerate()
        .map(|(axiom_id, axiom)| (axiom.affected_var_id, axiom_id))
        .collect();
    let mut auxiliary_by_name = BTreeMap::new();
    let mut recursion_stack = BTreeSet::new();
    for comparison_var in used_comparison_vars {
        let comparison_id = comparison_by_var[&comparison_var];
        let comparison = &task.comparison_axioms()[comparison_id];
        let left = parse_cpp_numeric_expression(
            task,
            comparison.get_left_var_id(),
            &assignment_by_var,
            &mut auxiliary_by_name,
            &mut features,
            &mut recursion_stack,
        )?;
        let right = parse_cpp_numeric_expression(
            task,
            comparison.get_right_var_id(),
            &assignment_by_var,
            &mut auxiliary_by_name,
            &mut features,
            &mut recursion_stack,
        )?;
        if !left.is_constant && !right.is_constant {
            insert_cpp_auxiliary_feature(
                task,
                comparison.get_left_var_id(),
                left,
                &mut auxiliary_by_name,
                &mut features,
            );
            insert_cpp_auxiliary_feature(
                task,
                comparison.get_right_var_id(),
                right,
                &mut auxiliary_by_name,
                &mut features,
            );
        }
    }
    Ok(features)
}

fn parse_cpp_numeric_expression(
    task: &dyn AbstractNumericTask,
    numeric_var_id: usize,
    assignment_by_var: &BTreeMap<usize, usize>,
    auxiliary_by_name: &mut BTreeMap<String, usize>,
    features: &mut Vec<NumericFeature>,
    recursion_stack: &mut BTreeSet<usize>,
) -> Result<ParsedNumericExpression, String> {
    let variable = task
        .numeric_variables()
        .get(numeric_var_id)
        .ok_or_else(|| {
            format!(
                "numeric_potential expression references unknown numeric variable {numeric_var_id}"
            )
        })?;
    match variable.get_type() {
        NumericType::Regular => Ok(ParsedNumericExpression {
            coefficients: vec![(numeric_var_id, 1.0)],
            constant: 0.0,
            affine: true,
            is_constant: false,
        }),
        NumericType::Constant => Ok(ParsedNumericExpression {
            coefficients: Vec::new(),
            constant: task.get_initial_numeric_state_values()[numeric_var_id],
            affine: true,
            is_constant: true,
        }),
        NumericType::Derived => {
            if !recursion_stack.insert(numeric_var_id) {
                return Err(format!(
                    "numeric_potential found a cyclic numeric assignment expression at variable {numeric_var_id}"
                ));
            }
            let axiom_id = assignment_by_var.get(&numeric_var_id).ok_or_else(|| {
                format!(
                    "numeric_potential found no assignment axiom for derived numeric variable {numeric_var_id}"
                )
            })?;
            let axiom = &task.assignment_axioms()[*axiom_id];
            let left = parse_cpp_numeric_expression(
                task,
                axiom.left_hand_side,
                assignment_by_var,
                auxiliary_by_name,
                features,
                recursion_stack,
            )?;
            let right = parse_cpp_numeric_expression(
                task,
                axiom.right_hand_side,
                assignment_by_var,
                auxiliary_by_name,
                features,
                recursion_stack,
            )?;
            recursion_stack.remove(&numeric_var_id);
            let expression = combine_numeric_expression(&axiom.operator, &left, &right);
            if !left.is_constant && !right.is_constant {
                insert_cpp_auxiliary_feature(
                    task,
                    numeric_var_id,
                    expression.clone(),
                    auxiliary_by_name,
                    features,
                );
            }
            Ok(expression)
        }
        other => Err(format!(
            "numeric_potential does not support numeric variable {numeric_var_id} (`{}`) of type {other:?} in a comparison",
            variable.name()
        )),
    }
}

fn combine_numeric_expression(
    operator: &CalOperator,
    left: &ParsedNumericExpression,
    right: &ParsedNumericExpression,
) -> ParsedNumericExpression {
    let mut result = ParsedNumericExpression {
        coefficients: Vec::new(),
        constant: 0.0,
        affine: left.affine && right.affine,
        is_constant: left.is_constant && right.is_constant,
    };
    match operator {
        CalOperator::Sum | CalOperator::Difference => {
            let right_scale = if matches!(operator, CalOperator::Sum) {
                1.0
            } else {
                -1.0
            };
            result.coefficients = add_sparse(&left.coefficients, &right.coefficients, right_scale);
            result.constant = left.constant + right_scale * right.constant;
        }
        CalOperator::Product if left.is_constant && right.affine => {
            result.coefficients = scale_sparse(&right.coefficients, left.constant);
            result.constant = left.constant * right.constant;
        }
        CalOperator::Product if right.is_constant && left.affine => {
            result.coefficients = scale_sparse(&left.coefficients, right.constant);
            result.constant = right.constant * left.constant;
        }
        CalOperator::Division if right.is_constant && right.constant != 0.0 && left.affine => {
            result.coefficients = scale_sparse(&left.coefficients, 1.0 / right.constant);
            result.constant = left.constant / right.constant;
        }
        CalOperator::Product | CalOperator::Division => {
            result.affine = false;
        }
    }
    result
}

fn insert_cpp_auxiliary_feature(
    task: &dyn AbstractNumericTask,
    source_numeric_var: usize,
    expression: ParsedNumericExpression,
    auxiliary_by_name: &mut BTreeMap<String, usize>,
    features: &mut Vec<NumericFeature>,
) -> usize {
    let name = task.numeric_variables()[source_numeric_var]
        .name()
        .to_string();
    if let Some(feature_id) = auxiliary_by_name.get(&name) {
        return *feature_id;
    }
    if !expression.affine {
        auxiliary_by_name.insert(name, usize::MAX);
        return usize::MAX;
    }
    let feature_id = features.len();
    features.push(NumericFeature {
        name: name.clone(),
        coefficients: expression.coefficients,
        constant: expression.constant,
        affine: expression.affine,
        source_numeric_var,
    });
    auxiliary_by_name.insert(name, feature_id);
    feature_id
}

struct AibrClosure {
    numeric_bounds: Vec<FeatureBounds>,
    reachable_facts: Vec<Vec<bool>>,
}

fn compute_aibr_closure(task: &dyn AbstractNumericTask) -> Result<AibrClosure, String> {
    let (initial_props, initial_numeric) = task
        .evaluated_initial_abstract_state_values()
        .map_err(|error| format!("numeric bounds could not evaluate initial state: {error}"))?;
    let mut reachable_facts: Vec<Vec<bool>> = task
        .variables()
        .iter()
        .map(|variable| vec![false; variable.domain_size()])
        .collect();
    for (var_id, value) in initial_props.into_iter().enumerate() {
        reachable_facts[var_id][value] = true;
    }
    let mut intervals: Vec<FeatureBounds> = initial_numeric
        .into_iter()
        .map(|value| FeatureBounds {
            lower: value,
            upper: value,
        })
        .collect();

    loop {
        let previous_facts = reachable_facts.clone();
        let previous_intervals = intervals.clone();

        for axiom in task.assignment_axioms() {
            let left = intervals[axiom.get_left_var_id()];
            let right = intervals[axiom.get_right_var_id()];
            let value = match axiom.get_operator() {
                CalOperator::Sum => interval_add(left, right),
                CalOperator::Difference => interval_subtract(left, right),
                CalOperator::Product => interval_product(left, right),
                CalOperator::Division => interval_divide(left, right),
            };
            hull_into(&mut intervals[axiom.get_affected_var_id()], value);
        }
        for axiom in task.comparison_axioms() {
            if comparison_may_hold(
                intervals[axiom.get_left_var_id()],
                intervals[axiom.get_right_var_id()],
                axiom.get_operator(),
            ) {
                reachable_facts[axiom.get_affected_var_id()][0] = true;
            }
        }
        for axiom in task.axioms() {
            if axiom
                .conditions()
                .iter()
                .all(|fact| reachable_facts[fact.var()][fact.value()])
            {
                reachable_facts[axiom.var_id()][axiom.effect_value()] = true;
            }
        }
        for operator in task.get_operators() {
            if !operator
                .preconditions()
                .iter()
                .all(|fact| reachable_facts[fact.var()][fact.value()])
            {
                continue;
            }
            for effect in operator.effects() {
                reachable_facts[effect.var_id()][effect.value()] = true;
            }
            for effect in operator.assignment_effects() {
                let source = intervals[effect.var_id()];
                let target = &mut intervals[effect.affected_var_id()];
                match effect.operation() {
                    AssignmentOperation::Assign => hull_into(target, source),
                    AssignmentOperation::Plus => {
                        if source.lower < 0.0 {
                            target.lower = f64::NEG_INFINITY;
                        }
                        if source.upper > 0.0 {
                            target.upper = f64::INFINITY;
                        }
                    }
                    AssignmentOperation::Minus => {
                        if source.upper > 0.0 {
                            target.lower = f64::NEG_INFINITY;
                        }
                        if source.lower < 0.0 {
                            target.upper = f64::INFINITY;
                        }
                    }
                    AssignmentOperation::Times | AssignmentOperation::Divide => {
                        *target = FeatureBounds::default();
                    }
                }
            }
        }

        if reachable_facts == previous_facts && intervals == previous_intervals {
            break;
        }
    }
    Ok(AibrClosure {
        numeric_bounds: intervals,
        reachable_facts,
    })
}

fn hull_into(target: &mut FeatureBounds, value: FeatureBounds) {
    target.lower = target.lower.min(value.lower);
    target.upper = target.upper.max(value.upper);
}

fn interval_add(left: FeatureBounds, right: FeatureBounds) -> FeatureBounds {
    FeatureBounds {
        lower: left.lower + right.lower,
        upper: left.upper + right.upper,
    }
}

fn interval_subtract(left: FeatureBounds, right: FeatureBounds) -> FeatureBounds {
    FeatureBounds {
        lower: left.lower - right.upper,
        upper: left.upper - right.lower,
    }
}

fn interval_product(left: FeatureBounds, right: FeatureBounds) -> FeatureBounds {
    let values = [
        left.lower * right.lower,
        left.lower * right.upper,
        left.upper * right.lower,
        left.upper * right.upper,
    ];
    if values.iter().any(|value| value.is_nan()) {
        return FeatureBounds::default();
    }
    FeatureBounds {
        lower: values.iter().copied().fold(f64::INFINITY, f64::min),
        upper: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

fn interval_divide(left: FeatureBounds, right: FeatureBounds) -> FeatureBounds {
    if right.lower <= 0.0 && right.upper >= 0.0 {
        FeatureBounds::default()
    } else {
        interval_product(
            left,
            FeatureBounds {
                lower: 1.0 / right.upper,
                upper: 1.0 / right.lower,
            },
        )
    }
}

fn comparison_may_hold(
    left: FeatureBounds,
    right: FeatureBounds,
    operator: &ComparisonOperator,
) -> bool {
    match operator {
        ComparisonOperator::LessThan => left.lower < right.upper,
        ComparisonOperator::LessThanOrEqual => left.lower <= right.upper,
        ComparisonOperator::Equal => left.lower <= right.upper && right.lower <= left.upper,
        ComparisonOperator::GreaterThanOrEqual => left.upper >= right.lower,
        ComparisonOperator::GreaterThan => left.upper > right.lower,
        ComparisonOperator::UnEqual => {
            left.lower != left.upper || right.lower != right.upper || left.lower != right.lower
        }
    }
}

fn intersect_bounds(target: &mut [FeatureBounds], contribution: &[FeatureBounds]) {
    assert_eq!(target.len(), contribution.len());
    for (target, contribution) in target.iter_mut().zip(contribution) {
        target.lower = target.lower.max(contribution.lower);
        target.upper = target.upper.min(contribution.upper);
    }
}

fn sparse_bounds(bounds: Vec<FeatureBounds>) -> Vec<(usize, FeatureBounds)> {
    bounds
        .into_iter()
        .enumerate()
        .filter(|(_, bounds)| bounds.lower.is_finite() || bounds.upper.is_finite())
        .collect()
}

fn intersect_sparse_bounds(
    local: &[(usize, FeatureBounds)],
    global: &[FeatureBounds],
) -> Vec<(usize, FeatureBounds)> {
    let mut result = Vec::new();
    let mut local_index = 0;
    for (feature_id, global_bounds) in global.iter().copied().enumerate() {
        let local_bounds = if local
            .get(local_index)
            .is_some_and(|(id, _)| *id == feature_id)
        {
            let bounds = local[local_index].1;
            local_index += 1;
            bounds
        } else {
            FeatureBounds::default()
        };
        let bounds = FeatureBounds {
            lower: local_bounds.lower.max(global_bounds.lower),
            upper: local_bounds.upper.min(global_bounds.upper),
        };
        if bounds.lower.is_finite() || bounds.upper.is_finite() {
            result.push((feature_id, bounds));
        }
    }
    debug_assert_eq!(local_index, local.len());
    result
}

fn map_primitive_bounds_to_features(
    features: &[NumericFeature],
    primitive_bounds: &[FeatureBounds],
    target: &mut [FeatureBounds],
) {
    for (feature_id, feature) in features.iter().enumerate() {
        if !feature.affine {
            continue;
        }
        let mut lower = feature.constant;
        let mut upper = feature.constant;
        for &(variable_id, coefficient) in &feature.coefficients {
            let bounds = primitive_bounds[variable_id];
            if coefficient >= 0.0 {
                lower += coefficient * bounds.lower;
                upper += coefficient * bounds.upper;
            } else {
                lower += coefficient * bounds.upper;
                upper += coefficient * bounds.lower;
            }
        }
        target[feature_id].lower = target[feature_id].lower.max(lower);
        target[feature_id].upper = target[feature_id].upper.min(upper);
    }
}

fn register_closed_condition_bound(
    condition: &LinearNumericCondition,
    features: &[NumericFeature],
    feature_keys: &BTreeMap<Vec<(usize, u64)>, usize>,
    bounds: &mut [FeatureBounds],
) -> Result<(), String> {
    if condition
        .coefficients
        .iter()
        .all(|coefficient| *coefficient == 0.0)
    {
        return Ok(());
    }
    let (canonical, condition_scale) = canonical_dense_feature(&condition.coefficients)
        .expect("nonconstant condition must have a canonical feature");
    let key = feature_key(&canonical);
    let feature_id = *feature_keys.get(&key).ok_or_else(|| {
        format!(
            "numeric_potential internal error: no feature for action condition `{}`",
            condition.name
        )
    })?;
    let (_, feature_scale) = canonical_feature(&features[feature_id].coefficients)
        .expect("matched feature must be nonconstant");
    let ratio = condition_scale / feature_scale;
    let threshold = features[feature_id].constant - condition.constant / ratio;
    if ratio > 0.0 {
        bounds[feature_id].lower = bounds[feature_id].lower.max(threshold);
    } else {
        bounds[feature_id].upper = bounds[feature_id].upper.min(threshold);
    }
    Ok(())
}

fn register_condition_bound(
    condition: &LinearNumericCondition,
    features: &[NumericFeature],
    feature_keys: &BTreeMap<Vec<(usize, u64)>, usize>,
    bounds: &mut [FeatureBounds],
    integral: bool,
    epsilon: f64,
    goal_unsatisfiable: &mut bool,
) -> Result<(), String> {
    if condition
        .coefficients
        .iter()
        .all(|coefficient| *coefficient == 0.0)
    {
        let required = if condition.is_strictly_greater {
            epsilon
        } else {
            0.0
        };
        *goal_unsatisfiable |= condition.constant < required;
        return Ok(());
    }
    let (canonical, condition_scale) = canonical_dense_feature(&condition.coefficients)
        .expect("nonconstant condition must have a canonical feature");
    let key = feature_key(&canonical);
    let feature_id = *feature_keys.get(&key).ok_or_else(|| {
        format!(
            "numeric_potential internal error: no feature for goal condition `{}`",
            condition.name
        )
    })?;
    let (_, feature_scale) = canonical_feature(&features[feature_id].coefficients)
        .expect("matched feature must be nonconstant");
    let ratio = condition_scale / feature_scale;
    let mut threshold = features[feature_id].constant - condition.constant / ratio;
    if ratio > 0.0 {
        if condition.is_strictly_greater {
            threshold = if integral {
                threshold.floor() + 1.0
            } else {
                threshold + epsilon / ratio
            };
        }
        bounds[feature_id].lower = bounds[feature_id].lower.max(threshold);
    } else {
        if condition.is_strictly_greater {
            threshold = if integral {
                threshold.ceil() - 1.0
            } else {
                threshold - epsilon / -ratio
            };
        }
        bounds[feature_id].upper = bounds[feature_id].upper.min(threshold);
    }
    if bounds[feature_id].lower > bounds[feature_id].upper {
        *goal_unsatisfiable = true;
    }
    Ok(())
}

fn canonical_feature(coefficients: &[(usize, f64)]) -> Option<(Vec<(usize, f64)>, f64)> {
    let scale = coefficients
        .iter()
        .map(|(_, coefficient)| *coefficient)
        .find(|coefficient| *coefficient != 0.0)?;
    let canonical = coefficients
        .iter()
        .filter_map(|&(variable_id, coefficient)| {
            let value = coefficient / scale;
            (value != 0.0).then_some((variable_id, value))
        })
        .collect();
    Some((canonical, scale))
}

fn canonical_dense_feature(coefficients: &[f64]) -> Option<(Vec<(usize, f64)>, f64)> {
    let sparse = coefficients
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, coefficient)| *coefficient != 0.0)
        .collect::<Vec<_>>();
    canonical_feature(&sparse)
}

fn feature_key(coefficients: &[(usize, f64)]) -> Vec<(usize, u64)> {
    coefficients
        .iter()
        .map(|&(variable_id, value)| (variable_id, value.to_bits()))
        .collect()
}

fn all_values_integral(
    task: &dyn AbstractNumericTask,
    features: &[NumericFeature],
    feature_keys: &BTreeMap<Vec<(usize, u64)>, usize>,
    goal_conditions: &[LinearNumericCondition],
    precision: f64,
) -> Result<bool, String> {
    let integral = |value: f64| (value - value.round()).abs() <= precision;
    let initial = task.get_initial_numeric_state_values();
    if !task
        .numeric_variables()
        .iter()
        .enumerate()
        .filter(|(_, variable)| {
            matches!(
                variable.get_type(),
                NumericType::Regular | NumericType::Constant
            )
        })
        .all(|(variable_id, _)| integral(initial[variable_id]))
        || !features.iter().all(|feature| {
            integral(dot_sparse_dense(&feature.coefficients, &initial) + feature.constant)
        })
    {
        return Ok(false);
    }
    for operator_id in 0..task.get_num_operators() {
        let effects = task
            .linearized_assignment_effects(operator_id)
            .map_err(|error| error.to_string())?;
        let mut primitive_delta = vec![0.0; task.numeric_variables().len()];
        for effect in effects {
            if effect.operation != AssignmentOperation::Assign && !effect.delta.is_constant() {
                return Ok(false);
            }
            if matches!(
                effect.operation,
                AssignmentOperation::Plus | AssignmentOperation::Minus
            ) {
                primitive_delta[effect.affected_var_id] += effect.delta.constant;
            }
        }
        if features
            .iter()
            .any(|feature| !integral(dot_sparse_dense(&feature.coefficients, &primitive_delta)))
        {
            return Ok(false);
        }
    }
    for condition in goal_conditions {
        let Some((canonical, condition_scale)) = canonical_dense_feature(&condition.coefficients)
        else {
            continue;
        };
        let Some(&feature_id) = feature_keys.get(&feature_key(&canonical)) else {
            continue;
        };
        let (_, feature_scale) = canonical_feature(&features[feature_id].coefficients)
            .expect("matched feature must be nonconstant");
        let ratio = condition_scale / feature_scale;
        let threshold = features[feature_id].constant - condition.constant / ratio;
        if !integral(threshold) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn dot_sparse_dense(sparse: &[(usize, f64)], dense: &[f64]) -> f64 {
    sparse
        .iter()
        .map(|&(index, value)| dense[index] * value)
        .sum()
}

fn dot_sparse(left: &[(usize, f64)], right: &[(usize, f64)]) -> f64 {
    let mut result = 0.0;
    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left.len() && right_index < right.len() {
        let (left_id, left_value) = left[left_index];
        let (right_id, right_value) = right[right_index];
        match left_id.cmp(&right_id) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                result += left_value * right_value;
                left_index += 1;
                right_index += 1;
            }
        }
    }
    result
}

fn add_sparse(
    left: &[(usize, f64)],
    right: &[(usize, f64)],
    right_scale: f64,
) -> Vec<(usize, f64)> {
    let mut coefficients = BTreeMap::new();
    for &(variable_id, coefficient) in left {
        coefficients.insert(variable_id, coefficient);
    }
    for &(variable_id, coefficient) in right {
        *coefficients.entry(variable_id).or_default() += right_scale * coefficient;
    }
    coefficients
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != 0.0)
        .collect()
}

fn scale_sparse(coefficients: &[(usize, f64)], scale: f64) -> Vec<(usize, f64)> {
    coefficients
        .iter()
        .filter_map(|&(variable_id, coefficient)| {
            let coefficient = coefficient * scale;
            (coefficient != 0.0).then_some((variable_id, coefficient))
        })
        .collect()
}

fn sparse_value(sparse: &[(usize, f64)], index: usize) -> f64 {
    sparse
        .binary_search_by_key(&index, |(id, _)| *id)
        .map(|position| sparse[position].1)
        .unwrap_or(0.0)
}
