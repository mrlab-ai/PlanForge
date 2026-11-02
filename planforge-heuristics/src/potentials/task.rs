use planforge_sas::numeric_task::Task;
use planforge_sas::numeric_task::VariableKind;
use std::collections::{BTreeMap, BTreeSet};
use tracing::info;

use crate::abstractions::common::coordinates::Coordinates;
use planforge_sas::axioms::{CalOperator, ComparisonOperator, NumericExpression};
use planforge_sas::default_value_axioms::{DefaultValueAxiomMode, default_value_axioms};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AssignmentOperation, ExplicitFact, ExplicitValueIndex, NumericType, NumericValue,
    OperatorIndex, VariableIndex, ZERO_VALUE,
};
use planforge_sas::state_registry::ConcreteStateView;
use planforge_sas::utils::value_lattice::ValueLattice;

use super::BoundsProvider;
use crate::landmarks::numeric_helper::{LinearNumericCondition, NumericTaskHelper};
use planforge_search::task_transformations::projection::validate_abstractable_goal;

mod reachable;

use reachable::{
    ReachableBounds, global_feature_bounds, intersect_bounds, restrict_operators_to_reachable,
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
    pub coefficients: Vec<(VariableIndex, f64)>,
    pub constant: f64,
    pub affine: bool,
    pub source_numeric_var: VariableIndex,
}

#[derive(Debug, Clone)]
pub struct PotentialOperator {
    pub preconditions: Vec<ExplicitFact>,
    pub effects: Vec<(VariableIndex, ExplicitValueIndex)>,
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
    pub coefficients: Vec<(VariableIndex, f64)>,
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
        task: &Task,
        precision: f64,
        epsilon: f64,
        ignore_numeric_variables: bool,
        bounds_provider: BoundsProvider,
        simple_action_bounds: bool,
        goal_strengthening: bool,
    ) -> Result<Self, String> {
        validate_abstractable_goal(task)?;
        reject_conditional_effects(task)?;
        let helper =
            NumericTaskHelper::new_potentials(task, precision, epsilon, simple_action_bounds);
        let domain_sizes = task
            .variables()
            .iter()
            .map(|var| var.domain_size())
            .collect();
        let derived_propositional: Vec<bool> = (0..task.variables().len())
            .map(|var| task.variable_kind(VariableIndex::from_usize(var)) != VariableKind::Regular)
            .collect();
        let goal = goal_conditions(task, &helper, goal_strengthening)?;

        // Classical-only mode keeps the complete C++ matrix shape: numeric
        // weight columns and the numeric goal-awareness row still exist, but
        // all weights are fixed to zero because no numeric goal interval is
        // registered below.
        let features = build_cpp_numeric_features(task, &goal.implied_comparisons)?;
        let feature_keys = affine_feature_keys(&features);
        let mut feature_goal_bounds = vec![FeatureBounds::default(); features.len()];
        let mut goal_unsatisfiable = false;
        if !ignore_numeric_variables {
            let lattices = Coordinates::new(task).value_lattices();
            for condition in &goal.numeric_conditions {
                register_condition_bound(
                    condition,
                    &features,
                    &feature_keys,
                    &mut feature_goal_bounds,
                    &lattices,
                    &mut goal_unsatisfiable,
                )?;
            }
        }

        let effects = primitive_effects(task)?;
        let assignment_target_features: Vec<bool> = features
            .iter()
            .map(|feature| {
                !feature.affine
                    || feature
                        .coefficients
                        .iter()
                        .any(|(variable_id, _)| effects.assignment_targets[variable_id.index()])
            })
            .collect();
        let global_linear_equalities = mine_linear_equalities(task, &effects);
        let mut operators =
            potential_operators(task, &helper, &features, &feature_keys, &effects.deltas)?;

        // Rays keep explicit goal endpoints unchanged. For a goal-free
        // resource, however, C++ falls back to the merged reachable bounds so
        // the homogeneous system can retain the corresponding kappa
        // certificate instead of pinning its weight to zero.
        let raw_ray_feature_goal_bounds = feature_goal_bounds.clone();
        let ReachableBounds {
            features: global_feature_bounds,
            facts: reachable_facts,
        } = global_feature_bounds(
            task,
            &helper,
            &features,
            &assignment_target_features,
            &effects,
            bounds_provider,
            simple_action_bounds,
        )?;
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
        if !matches!(bounds_provider, BoundsProvider::None) || simple_action_bounds {
            restrict_operators_to_reachable(
                &mut operators,
                &global_feature_bounds,
                reachable_facts.as_deref(),
            );
        }
        if let Some(facts) = &reachable_facts {
            goal_unsatisfiable |= goal
                .propositional_goals
                .iter()
                .any(|goal| !facts[goal.var()][goal.value()]);
        }

        Ok(Self {
            numeric_variable_count: task.numeric_variables().len(),
            domain_sizes,
            derived_propositional,
            propositional_goals: goal.propositional_goals,
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
        state: ConcreteStateView<'_>,
        numeric_scratch: &mut Vec<NumericValue>,
    ) -> Result<Vec<f64>, String> {
        state
            .fill_numeric(numeric_scratch)
            .map_err(|error| format!("failed to unpack numeric state: {error:?}"))?;
        Ok(self
            .features
            .iter()
            .map(|feature| {
                if feature.affine {
                    NumericValue::affine(
                        feature
                            .coefficients
                            .iter()
                            .map(|&(var, coefficient)| (coefficient, numeric_scratch[var.index()])),
                        feature.constant,
                    )
                    .value()
                } else {
                    numeric_scratch[feature.source_numeric_var.index()].value()
                }
            })
            .collect())
    }
}

/// Fails on conditional effects, which the potential LP cannot express.
fn reject_conditional_effects(task: &Task) -> Result<(), String> {
    for (operator_id, operator) in task.operators().iter().enumerate() {
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
    Ok(())
}

/// The goal as the LP reads it.
struct GoalConditions {
    /// The propositional goal facts, sorted and without duplicates.
    propositional_goals: Vec<ExplicitFact>,
    /// The linear conditions of the comparison goals and of the comparisons
    /// the goal implies.
    numeric_conditions: Vec<LinearNumericCondition>,
    /// The comparisons the propositional goals imply.
    implied_comparisons: Vec<VariableIndex>,
}

/// A comparison goal becomes the linear conditions materializing it; every
/// other goal fact is propositional, since `validate_abstractable_goal` has
/// refused a goal on a derived variable. That case used to be rewritten into
/// the conditions of its one defining axiom, and rejected when there was not
/// exactly one -- a rewrite that had to be equivalent to the goal, not merely
/// implied by it, to keep the LP bound admissible. With `goal_strengthening`,
/// a comparison every reachable goal state satisfies joins the goal.
fn goal_conditions(
    task: &Task,
    helper: &NumericTaskHelper,
    goal_strengthening: bool,
) -> Result<GoalConditions, String> {
    let mut propositional_goals = Vec::new();
    let mut numeric_conditions = Vec::new();
    for &goal in task.goals() {
        let materialized =
            helper.comparison_fact_materialized_conditions(goal.var_index(), goal.value_index());
        if materialized.is_empty() {
            propositional_goals.push(goal);
        } else {
            numeric_conditions.extend(materialized);
        }
    }
    propositional_goals.sort_unstable();
    propositional_goals.dedup();
    let implied_comparisons = if goal_strengthening {
        goal_implied_comparisons(task, helper, &propositional_goals)?
    } else {
        Vec::new()
    };
    info!(
        "Numeric potential goal-implied comparisons: {}",
        implied_comparisons.len()
    );
    for &comparison_var in &implied_comparisons {
        numeric_conditions.extend(helper.comparison_fact_materialized_conditions(
            comparison_var,
            ExplicitValueIndex::from_usize(ConditionValue::True.as_usize()),
        ));
    }
    Ok(GoalConditions {
        propositional_goals,
        numeric_conditions,
        implied_comparisons,
    })
}

/// The id of every nonconstant affine feature, by its canonical
/// coefficients.
fn affine_feature_keys(
    features: &[NumericFeature],
) -> BTreeMap<Vec<(VariableIndex, NumericValue)>, usize> {
    let mut feature_keys = BTreeMap::new();
    for (feature_id, feature) in features.iter().enumerate() {
        if !feature.affine || feature.coefficients.is_empty() {
            continue;
        }
        let (canonical, _) = canonical_feature(&feature.coefficients)
            .expect("nonconstant affine feature must be canonicalizable");
        feature_keys.insert(feature_key(&canonical), feature_id);
    }
    feature_keys
}

/// What the operators do to the numeric variables.
struct PrimitiveEffects {
    /// Per numeric variable, whether some operator assigns it.
    assignment_targets: Vec<bool>,
    /// Per operator, its nonzero constant change of each numeric variable,
    /// ascending by variable.
    deltas: Vec<Vec<(VariableIndex, f64)>>,
}

/// The operators' numeric effects, which must be constant increases and
/// decreases or assignments.
fn primitive_effects(task: &Task) -> Result<PrimitiveEffects, String> {
    let mut assignment_targets = vec![false; task.numeric_variables().len()];
    let mut primitive_deltas = Vec::with_capacity(task.operators().len());
    for (operator_id, operator) in task.operators().iter().enumerate() {
        let linearized = task
            .linearized_assignment_effects(OperatorIndex::from_usize(operator_id))
            .map_err(|error| {
                format!(
                    "numeric_potential cannot linearize numeric effects of operator {operator_id} (`{}`): {error}",
                    operator.name()
                )
            })?;
        let mut deltas = BTreeMap::<VariableIndex, f64>::new();
        for effect in linearized {
            match effect.operation {
                AssignmentOperation::Assign => {
                    assignment_targets[effect.affected_var_id.index()] = true;
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
    Ok(PrimitiveEffects {
        assignment_targets,
        deltas: primitive_deltas,
    })
}

/// The pairwise invariants of the effect profile: two regular variables no
/// operator assigns, whose changes coincide or cancel in every operator and
/// that some operator changes, keep their difference or sum from the initial
/// state.
fn mine_linear_equalities(task: &Task, effects: &PrimitiveEffects) -> Vec<LinearEquality> {
    let numeric_count = task.numeric_variables().len();
    let initial_numeric = task.initial_numeric_values();
    let candidate = |var: usize| {
        task.numeric_variables()[var].numeric_type() == NumericType::Regular
            && !effects.assignment_targets[var]
    };
    let mut equalities = Vec::new();
    for lhs in (0..numeric_count).filter(|&var| candidate(var)) {
        let lhs_id = VariableIndex::from_usize(lhs);
        for rhs in ((lhs + 1)..numeric_count).filter(|&var| candidate(var)) {
            let rhs_id = VariableIndex::from_usize(rhs);
            let identical = effects
                .deltas
                .iter()
                .all(|deltas| sparse_value(deltas, lhs_id) == sparse_value(deltas, rhs_id));
            let opposite = effects
                .deltas
                .iter()
                .all(|deltas| sparse_value(deltas, lhs_id) == -sparse_value(deltas, rhs_id));
            let has_nonzero_delta = effects.deltas.iter().any(|deltas| {
                sparse_value(deltas, lhs_id) != 0.0 || sparse_value(deltas, rhs_id) != 0.0
            });
            if !has_nonzero_delta || (!identical && !opposite) {
                continue;
            }
            let rhs_coefficient = if identical { -1.0 } else { 1.0 };
            equalities.push(LinearEquality {
                rhs: NumericValue::affine(
                    [
                        (1.0, initial_numeric[lhs]),
                        (rhs_coefficient, initial_numeric[rhs]),
                    ],
                    0.0,
                )
                .value(),
                coefficients: vec![(lhs_id, 1.0), (rhs_id, rhs_coefficient)],
            });
        }
    }
    equalities
}

/// Every operator as the LP reads it: its propositional preconditions and
/// effects, its change of every affine feature, the feature bounds its
/// numeric preconditions require, and its cost.
fn potential_operators(
    task: &Task,
    helper: &NumericTaskHelper,
    features: &[NumericFeature],
    feature_keys: &BTreeMap<Vec<(VariableIndex, NumericValue)>, usize>,
    primitive_deltas: &[Vec<(VariableIndex, f64)>],
) -> Result<Vec<PotentialOperator>, String> {
    task.operators()
        .iter()
        .enumerate()
        .map(|(operator_id, operator)| {
            let mut numeric_precondition_bounds = vec![FeatureBounds::default(); features.len()];
            for precondition in operator.preconditions() {
                for condition in helper.comparison_fact_materialized_conditions(
                    precondition.var_index(),
                    precondition.value_index(),
                ) {
                    register_closed_condition_bound(
                        &condition,
                        features,
                        feature_keys,
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
                            let delta =
                                dot_sparse(&feature.coefficients, &primitive_deltas[operator_id]);
                            (delta != 0.0).then_some((feature_id, delta))
                        } else {
                            None
                        }
                    })
                    .collect(),
                numeric_precondition_bounds: sparse_bounds(numeric_precondition_bounds),
                cost: task.operator_cost(OperatorIndex::from_usize(operator_id)),
                reachable: true,
            })
        })
        .collect()
}

/// The comparisons every reachable state satisfying some propositional goal
/// satisfies too, which the goal may therefore be strengthened by (the
/// mutex-based goal strengthening of Fišer, Horčík, and Komenda 2020).
///
/// The translator's mutex groups never name a comparison, so the implication
/// "goal fact `g` implies comparison `c`" is proven an inductive invariant
/// from the operators: it holds initially (`g` false or `c` true), an
/// operator that makes `g` true requires `c` and cannot lower it, and an
/// operator that can lower `c` requires `g` false or makes it false.
fn goal_implied_comparisons(
    task: &Task,
    helper: &NumericTaskHelper,
    propositional_goals: &[ExplicitFact],
) -> Result<Vec<VariableIndex>, String> {
    let numeric_count = task.numeric_variables().len();
    // Per operator, the constant change of every numeric variable, and the
    // variables it assigns or changes by a state-dependent amount: those can
    // take any value afterwards.
    let mut deltas = Vec::with_capacity(task.operators().len());
    let mut unbounded_changes = Vec::with_capacity(task.operators().len());
    for (operator_id, operator) in task.operators().iter().enumerate() {
        let mut operator_deltas = vec![0.0; numeric_count];
        let mut operator_unbounded = vec![false; numeric_count];
        let effects = task
            .linearized_assignment_effects(OperatorIndex::from_usize(operator_id))
            .map_err(|error| {
                format!(
                    "numeric_potential cannot linearize numeric effects of operator {operator_id} (`{}`): {error}",
                    operator.name()
                )
            })?;
        for effect in effects {
            let var = effect.affected_var_id.index();
            match effect.operation {
                AssignmentOperation::Plus | AssignmentOperation::Minus
                    if effect.delta.is_constant() =>
                {
                    operator_deltas[var] += effect.delta.constant;
                }
                _ => operator_unbounded[var] = true,
            }
        }
        deltas.push(operator_deltas);
        unbounded_changes.push(operator_unbounded);
    }
    // Whether the operator can make a linear condition of `c` false.
    let may_lower = |operator_id: usize, conditions: &[LinearNumericCondition]| {
        conditions.iter().any(|condition| {
            let coefficients = condition.coefficients.iter().enumerate();
            coefficients.clone().any(|(var, &coefficient)| {
                coefficient != 0.0 && unbounded_changes[operator_id][var]
            }) || coefficients
                .map(|(var, &coefficient)| coefficient * deltas[operator_id][var])
                .sum::<f64>()
                < 0.0
        })
    };
    let true_value = ExplicitValueIndex::from_usize(ConditionValue::True.as_usize());
    let initial = task.initial_propositional_values();
    let mut implied = Vec::new();
    for comparison in task.comparison_axioms() {
        let comparison_var = comparison.get_affected_var_id();
        let conditions = helper.comparison_fact_materialized_conditions(comparison_var, true_value);
        if conditions.is_empty() {
            continue;
        }
        let initially_true = initial[comparison_var.index()] == true_value;
        let implied_by = |goal: &ExplicitFact| {
            let holds_initially = initially_true || initial[goal.var()] != goal.value_index();
            holds_initially
                && task
                    .operators()
                    .iter()
                    .enumerate()
                    .all(|(operator_id, operator)| {
                        let precondition_on_goal_var = operator
                            .preconditions()
                            .iter()
                            .find(|precondition| precondition.var_index() == goal.var_index());
                        let effect_on_goal_var = operator
                            .effects()
                            .iter()
                            .find(|effect| effect.var_id() == goal.var_index());
                        let requires_goal = precondition_on_goal_var == Some(goal);
                        let goal_false_after = effect_on_goal_var
                            .map(|effect| effect.value() != goal.value_index())
                            .unwrap_or_else(|| {
                                precondition_on_goal_var.is_some() && !requires_goal
                            });
                        let makes_goal_true = effect_on_goal_var
                            .is_some_and(|effect| effect.value() == goal.value_index())
                            && !requires_goal;
                        let lowers = may_lower(operator_id, &conditions);
                        if makes_goal_true {
                            !lowers
                                && operator.preconditions().iter().any(|precondition| {
                                    precondition.var_index() == comparison_var
                                        && precondition.value_index() == true_value
                                })
                        } else {
                            !lowers || goal_false_after
                        }
                    })
        };
        if propositional_goals.iter().any(implied_by) {
            implied.push(comparison_var);
        }
    }
    Ok(implied)
}

#[derive(Clone)]
struct ParsedNumericExpression {
    coefficients: Vec<(VariableIndex, f64)>,
    constant: f64,
    affine: bool,
    is_constant: bool,
}

fn build_cpp_numeric_features(
    task: &Task,
    implied_comparisons: &[VariableIndex],
) -> Result<Vec<NumericFeature>, String> {
    let mut features = Vec::new();
    for (numeric_var_id, variable) in task.numeric_variables().iter().enumerate() {
        if variable.numeric_type() != NumericType::Regular {
            continue;
        }
        let numeric_var_id = VariableIndex::from_usize(numeric_var_id);
        features.push(NumericFeature {
            name: variable.name().to_string(),
            coefficients: vec![(numeric_var_id, 1.0)],
            constant: 0.0,
            affine: true,
            source_numeric_var: numeric_var_id,
        });
    }

    let comparison_by_var: BTreeMap<VariableIndex, usize> = task
        .comparison_axioms()
        .iter()
        .enumerate()
        .map(|(comparison_id, comparison)| (comparison.get_affected_var_id(), comparison_id))
        .collect();
    let mut used_comparison_vars = Vec::new();
    let mut seen_comparisons = BTreeSet::new();
    {
        let mut record_fact = |fact: &ExplicitFact| {
            let var = fact.var_index();
            if comparison_by_var.contains_key(&var) && seen_comparisons.insert(var) {
                used_comparison_vars.push(var);
            }
        };
        // The C++ numeric_pdb_helper construction order: operator
        // preconditions, then the axiom bodies, then the goal. The C++
        // translator's goal was a derived atom whose axiom body named the
        // goal comparisons; here the goal names them itself.
        //
        // Only the variable of a body fact matters here, and every variable a
        // rule refuting `v` names comes from a body of a rule proving `v` --
        // so refuting rules never contribute a variable, nor a position in
        // the order, that the proving rules have not already contributed.
        for operator in task.operators() {
            for precondition in operator.preconditions() {
                record_fact(precondition);
            }
        }
        for axiom in task.axioms() {
            for condition in axiom.conditions() {
                record_fact(condition);
            }
        }
        for goal in task.goals() {
            record_fact(goal);
        }
    }
    // The comparisons the goal implies enter the goal, so they need features
    // too.
    for &comparison_var in implied_comparisons {
        if seen_comparisons.insert(comparison_var) {
            used_comparison_vars.push(comparison_var);
        }
    }

    let mut auxiliary_by_name = BTreeMap::new();
    for comparison_var in used_comparison_vars {
        let comparison_id = comparison_by_var[&comparison_var];
        let comparison = &task.comparison_axioms()[comparison_id];
        let left = parse_cpp_numeric_expression(
            task,
            comparison.get_left_var_id(),
            &mut auxiliary_by_name,
            &mut features,
        )?;
        let right = parse_cpp_numeric_expression(
            task,
            comparison.get_right_var_id(),
            &mut auxiliary_by_name,
            &mut features,
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
    task: &Task,
    numeric_var_id: VariableIndex,
    auxiliary_by_name: &mut BTreeMap<String, usize>,
    features: &mut Vec<NumericFeature>,
) -> Result<ParsedNumericExpression, String> {
    let expression = match task.linearize_numeric_var(numeric_var_id) {
        Ok(expression) => ParsedNumericExpression {
            is_constant: expression.is_constant(),
            coefficients: expression
                .coefficients
                .into_iter()
                .enumerate()
                .filter(|(_, value)| *value != 0.0)
                .map(|(id, coefficient)| (VariableIndex::from_usize(id), coefficient))
                .collect(),
            constant: expression.constant,
            affine: true,
        },
        Err(
            planforge_sas::utils::linear_effects::LinearizationError::NonLinearAssignmentAxiom {
                ..
            },
        ) => ParsedNumericExpression {
            coefficients: Vec::new(),
            constant: 0.0,
            affine: false,
            is_constant: false,
        },
        Err(error) => {
            return Err(format!(
                "numeric_potential cannot expand variable {numeric_var_id}: {error}"
            ));
        }
    };
    if task.numeric_type(numeric_var_id) == NumericType::Derived {
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

fn insert_cpp_auxiliary_feature(
    task: &Task,
    source_numeric_var: VariableIndex,
    expression: ParsedNumericExpression,
    auxiliary_by_name: &mut BTreeMap<String, usize>,
    features: &mut Vec<NumericFeature>,
) -> usize {
    let name = task.numeric_variables()[source_numeric_var.index()]
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
            let bounds = primitive_bounds[variable_id.index()];
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
    feature_keys: &BTreeMap<Vec<(VariableIndex, NumericValue)>, usize>,
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
    feature_keys: &BTreeMap<Vec<(VariableIndex, NumericValue)>, usize>,
    bounds: &mut [FeatureBounds],
    lattices: &[ValueLattice],
    goal_unsatisfiable: &mut bool,
) -> Result<(), String> {
    if condition
        .coefficients
        .iter()
        .all(|coefficient| *coefficient == 0.0)
    {
        let constant = NumericValue::new(condition.constant);
        *goal_unsatisfiable |= if condition.is_strictly_greater {
            constant <= ZERO_VALUE
        } else {
            constant < ZERO_VALUE
        };
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
            threshold = lattices[features[feature_id].source_numeric_var.index()]
                .split_bounds(NumericValue::new(threshold), true)
                .ok_or_else(|| {
                    format!("non-finite strict numeric potential threshold: {threshold}")
                })?
                .1
                .value();
        }
        bounds[feature_id].lower = bounds[feature_id].lower.max(threshold);
    } else {
        if condition.is_strictly_greater {
            threshold = lattices[features[feature_id].source_numeric_var.index()]
                .split_bounds(NumericValue::new(threshold), false)
                .ok_or_else(|| {
                    format!("non-finite strict numeric potential threshold: {threshold}")
                })?
                .0
                .value();
        }
        bounds[feature_id].upper = bounds[feature_id].upper.min(threshold);
    }
    if bounds[feature_id].lower > bounds[feature_id].upper {
        *goal_unsatisfiable = true;
    }
    Ok(())
}

fn canonical_feature(
    coefficients: &[(VariableIndex, f64)],
) -> Option<(Vec<(VariableIndex, f64)>, f64)> {
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

fn canonical_dense_feature(coefficients: &[f64]) -> Option<(Vec<(VariableIndex, f64)>, f64)> {
    let sparse = coefficients
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, coefficient)| *coefficient != 0.0)
        .map(|(variable_id, coefficient)| (VariableIndex::from_usize(variable_id), coefficient))
        .collect::<Vec<_>>();
    canonical_feature(&sparse)
}

fn feature_key(coefficients: &[(VariableIndex, f64)]) -> Vec<(VariableIndex, NumericValue)> {
    coefficients
        .iter()
        .map(|&(variable_id, value)| (variable_id, NumericValue::new(value)))
        .collect()
}

fn dot_sparse(left: &[(VariableIndex, f64)], right: &[(VariableIndex, f64)]) -> f64 {
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

fn sparse_value(sparse: &[(VariableIndex, f64)], index: VariableIndex) -> f64 {
    sparse
        .binary_search_by_key(&index, |(id, _)| *id)
        .map(|position| sparse[position].1)
        .unwrap_or(0.0)
}
