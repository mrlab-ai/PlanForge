//! The projection of a task to a pattern: the task over only the pattern's
//! variables, whose operators keep their preconditions and effects on those
//! variables and drop everything else.
//!
//! Projections are primarily the abstract tasks of pattern databases. A
//! pattern database for the pattern {truck location, package location} of a
//! delivery task explores the projection to those two variables: an operator
//! that drives the truck keeps its location precondition and effect but loses
//! its fuel precondition, so every plan of the task is a plan of the
//! projection, and the projection's goal distances are admissible estimates
//! of the task's. The projection is a task in its own right, though, so any
//! search or heuristic can run on it.

#[cfg(test)]
mod tests;

use planforge_sas::explicit_state::StateValues;
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use planforge_sas::axioms::{
    AssignmentAxiom, ComparisonAxiom, NumericExpression, PropositionalAxiom,
};
use planforge_sas::numeric_task::{
    AssignmentEffect, Effect, ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric,
    NumericType, NumericValue, NumericVariable, Operator, OperatorIndex, Task, TaskParts,
    VariableIndex,
};

use crate::task_transformations::TransformError;
use crate::task_transformations::restriction::validate_restricted_task;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pattern {
    pub regular: Vec<VariableIndex>,
    pub numeric: Vec<VariableIndex>,
}

impl Pattern {
    pub fn new(regular: Vec<VariableIndex>, numeric: Vec<VariableIndex>) -> Self {
        let mut pattern = Self { regular, numeric };
        pattern.normalize_in_place();
        pattern
    }

    pub fn normalized(&self) -> Self {
        let mut pattern = self.clone();
        pattern.normalize_in_place();
        pattern
    }

    pub fn normalize_in_place(&mut self) {
        self.regular.sort_unstable();
        self.regular.dedup();
        self.numeric.sort_unstable();
        self.numeric.dedup();
    }

    pub fn is_empty(&self) -> bool {
        self.regular.is_empty() && self.numeric.is_empty()
    }

    pub fn total_len(&self) -> usize {
        self.regular.len() + self.numeric.len()
    }

    pub fn add_regular_var(&mut self, var_id: VariableIndex) -> bool {
        match self.regular.binary_search(&var_id) {
            Ok(_) => false,
            Err(index) => {
                self.regular.insert(index, var_id);
                true
            }
        }
    }

    pub fn add_numeric_var(&mut self, var_id: VariableIndex) -> bool {
        match self.numeric.binary_search(&var_id) {
            Ok(_) => false,
            Err(index) => {
                self.numeric.insert(index, var_id);
                true
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProjectedTaskBuildError {
    UnrestrictedTask {
        reason: String,
    },
    /// The base task's goal names a variable only the axioms write; see
    /// [`validate_abstractable_goal`].
    UnabstractableGoal {
        reason: String,
    },
    InvalidRegularVarId {
        provided: usize,
        len: usize,
    },
    InvalidNumericVarId {
        provided: usize,
        len: usize,
    },
    UnsupportedPatternNumericVarType {
        numeric_var_id: VariableIndex,
        numeric_type: NumericType,
    },
}

impl fmt::Display for ProjectedTaskBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnrestrictedTask { reason } | Self::UnabstractableGoal { reason } => {
                write!(formatter, "{reason}")
            }
            Self::InvalidRegularVarId { provided, len } => write!(
                formatter,
                "invalid projected propositional variable {provided}; task has {len} variables"
            ),
            Self::InvalidNumericVarId { provided, len } => write!(
                formatter,
                "invalid projected numeric variable {provided}; task has {len} numeric variables"
            ),
            Self::UnsupportedPatternNumericVarType {
                numeric_var_id,
                numeric_type,
            } => write!(
                formatter,
                "pattern numeric variable {numeric_var_id} has unsupported type {:?}; PDB patterns over restricted tasks accept only regular numeric variables",
                numeric_type
            ),
        }
    }
}

impl std::error::Error for ProjectedTaskBuildError {}

/// A pattern database cannot be built on a task the projection does not
/// support; a pattern naming no variable of the task is a failure.
impl From<ProjectedTaskBuildError> for crate::heuristic_features::HeuristicBuildError {
    fn from(error: ProjectedTaskBuildError) -> Self {
        match error {
            ProjectedTaskBuildError::UnrestrictedTask { .. }
            | ProjectedTaskBuildError::UnabstractableGoal { .. }
            | ProjectedTaskBuildError::UnsupportedPatternNumericVarType { .. } => {
                Self::Unsupported(error.to_string())
            }
            ProjectedTaskBuildError::InvalidRegularVarId { .. }
            | ProjectedTaskBuildError::InvalidNumericVarId { .. } => {
                Self::Failure(error.to_string())
            }
        }
    }
}

/// The projection of a task onto a pattern: a task of its own over the
/// pattern's variables, and the ids that lead back to the base task.
///
/// The projected task keeps the base task's operator costs, so its goal
/// distances are those of the abstraction the pattern induces.
#[derive(Debug)]
pub struct ProjectedTask {
    task: Arc<Task>,
    base_operator_ids: Vec<OperatorIndex>,
    projected_var_to_original: Vec<VariableIndex>,
    projected_num_var_to_original: Vec<VariableIndex>,
    base_num_variables: usize,
    base_num_numeric_variables: usize,
}

impl ProjectedTask {
    pub fn new(base: &Task, pattern: &Pattern) -> Result<Self, ProjectedTaskBuildError> {
        validate_restricted_task(base).map_err(|reason| {
            ProjectedTaskBuildError::UnrestrictedTask {
                reason: reason.to_string(),
            }
        })?;
        validate_abstractable_goal(base).map_err(|reason| {
            ProjectedTaskBuildError::UnabstractableGoal {
                reason: reason.to_string(),
            }
        })?;
        let num_vars = base.variables().len();
        let num_numeric_vars = base.numeric_variables().len();

        let base_initial_numeric_values = base.initial_numeric_values();

        let mut projected_var_to_original: Vec<VariableIndex> = Vec::new();
        let mut projected_num_var_to_original: Vec<VariableIndex> = Vec::new();
        let mut original_var_to_projected = vec![None; num_vars];
        let mut original_num_var_to_projected = vec![None; num_numeric_vars];

        for &var_id in &pattern.regular {
            if var_id.index() >= num_vars {
                return Err(ProjectedTaskBuildError::InvalidRegularVarId {
                    provided: var_id.index(),
                    len: num_vars,
                });
            }
            push_unique_mapping(
                var_id,
                &mut projected_var_to_original,
                &mut original_var_to_projected,
            );
        }

        for &numeric_var_id in &pattern.numeric {
            if numeric_var_id.index() < num_numeric_vars {
                let numeric_type = base.numeric_type(numeric_var_id);
                if numeric_type != NumericType::Regular {
                    return Err(ProjectedTaskBuildError::UnsupportedPatternNumericVarType {
                        numeric_var_id,
                        numeric_type,
                    });
                }
                push_unique_mapping(
                    numeric_var_id,
                    &mut projected_num_var_to_original,
                    &mut original_num_var_to_projected,
                );
            } else {
                return Err(ProjectedTaskBuildError::InvalidNumericVarId {
                    provided: numeric_var_id.index(),
                    len: num_numeric_vars,
                });
            }
        }

        for original_var_id in projected_var_to_original.clone() {
            let Some(comparison_axiom_id) = base.numeric_conditions().id_for_var(original_var_id)
            else {
                continue;
            };
            include_restricted_comparison_operands(
                base,
                comparison_axiom_id,
                &mut projected_num_var_to_original,
                &mut original_num_var_to_projected,
            );
        }
        let goal_facts = collect_restricted_projected_goals(
            base,
            pattern,
            VariableProjection {
                projected_var_to_original: &mut projected_var_to_original,
                original_var_to_projected: &mut original_var_to_projected,
                projected_num_var_to_original: &mut projected_num_var_to_original,
                original_num_var_to_projected: &mut original_num_var_to_projected,
            },
        );
        let mut numeric_effect_sources_by_target = vec![Vec::new(); num_numeric_vars];
        for operator in base.operators() {
            for effect in operator.assignment_effects() {
                numeric_effect_sources_by_target[effect.affected_var_id().index()]
                    .push(effect.var_id());
            }
        }
        let mut closure_index = 0;
        while closure_index < projected_num_var_to_original.len() {
            let target_var_id = projected_num_var_to_original[closure_index];
            for &source_var_id in &numeric_effect_sources_by_target[target_var_id.index()] {
                push_unique_mapping(
                    source_var_id,
                    &mut projected_num_var_to_original,
                    &mut original_num_var_to_projected,
                );
            }
            closure_index += 1;
        }
        include_comparisons_over_projected_variables(
            base,
            VariableProjection {
                projected_var_to_original: &mut projected_var_to_original,
                original_var_to_projected: &mut original_var_to_projected,
                projected_num_var_to_original: &mut projected_num_var_to_original,
                original_num_var_to_projected: &mut original_num_var_to_projected,
            },
        );

        let mut variable_names: Vec<String> = Vec::with_capacity(projected_var_to_original.len());
        let mut fact_names: Vec<Vec<String>> = Vec::with_capacity(projected_var_to_original.len());
        let mut variable_domain_sizes: Vec<usize> =
            Vec::with_capacity(projected_var_to_original.len());
        let mut variable_default_values: Vec<ExplicitValueIndex> =
            Vec::with_capacity(projected_var_to_original.len());
        for &original_var_id in &projected_var_to_original {
            let variable_name = base.variable(original_var_id).name().to_string();
            let domain_size = base.variable(original_var_id).domain_size();

            let var_fact_names = (0..domain_size)
                .map(|value| {
                    let original_fact = base
                        .numeric_conditions()
                        .fact(original_var_id.index(), value);
                    let fact_name = base.fact_name(&original_fact);
                    if fact_name.is_empty() {
                        format!("{variable_name}={value}")
                    } else {
                        fact_name.to_string()
                    }
                })
                .collect();

            variable_domain_sizes.push(domain_size);
            variable_default_values.push(base.variable(original_var_id).axiom_default_value());
            variable_names.push(variable_name);
            fact_names.push(var_fact_names);
        }

        let initial_prop_values = base.initial_propositional_values();
        let projected_prop_values: Vec<ExplicitValueIndex> = projected_var_to_original
            .iter()
            .map(|&original| initial_prop_values[original.index()])
            .collect();

        let mut numeric_variables: Vec<NumericVariable> =
            Vec::with_capacity(projected_num_var_to_original.len());
        let mut projected_numeric_values: Vec<NumericValue> =
            Vec::with_capacity(projected_num_var_to_original.len());
        for &source_original in &projected_num_var_to_original {
            numeric_variables.push(base.numeric_variables()[source_original.index()].clone());
            projected_numeric_values.push(base_initial_numeric_values[source_original.index()]);
        }

        let goals: Vec<ExplicitFact> = goal_facts
            .iter()
            .filter_map(|goal| project_fact(goal, &original_var_to_projected))
            .collect();

        let mut operators: Vec<Operator> = Vec::new();
        let mut operator_costs: Vec<f64> = Vec::new();
        let mut base_operator_ids: Vec<OperatorIndex> = Vec::new();
        for (base_operator_id, operator) in base.operators().iter().enumerate() {
            let operator_cost = base.operator_cost(OperatorIndex::from_usize(base_operator_id));
            if let Some(projected_operator) = project_restricted_operator(
                operator,
                &original_var_to_projected,
                &original_num_var_to_projected,
            ) {
                operators.push(projected_operator);
                operator_costs.push(operator_cost);
                base_operator_ids.push(OperatorIndex::from_usize(base_operator_id));
            }
        }

        let axioms: Vec<PropositionalAxiom> = base
            .axioms()
            .iter()
            .filter_map(|axiom| project_propositional_axiom(axiom, &original_var_to_projected))
            .collect();

        let mut comparison_axioms: Vec<ComparisonAxiom> = Vec::new();
        for comparison_axiom_id in 0..base.comparison_axioms().len() {
            if let Some(projected_axiom) = project_restricted_comparison_axiom(
                base,
                comparison_axiom_id,
                &original_var_to_projected,
                &original_num_var_to_projected,
            ) {
                comparison_axioms.push(projected_axiom);
            }
        }

        let assignment_axioms: Vec<AssignmentAxiom> = base
            .assignment_axioms()
            .iter()
            .filter_map(|axiom| project_assignment_axiom(axiom, &original_num_var_to_projected))
            .collect();

        let variable_layers = normalize_projected_variable_layers(
            projected_var_to_original.len(),
            &numeric_variables,
            &comparison_axioms,
            &axioms,
        );
        let variables: Vec<ExplicitVariable> = variable_names
            .into_iter()
            .zip(fact_names.iter().cloned())
            .zip(variable_domain_sizes.iter().copied())
            .zip(variable_default_values.iter().copied())
            .zip(variable_layers.iter().copied())
            .map(
                |((((name, fact_names), domain_size), axiom_default_value), axiom_layer)| {
                    ExplicitVariable::new(
                        domain_size,
                        name,
                        fact_names,
                        axiom_layer,
                        axiom_default_value,
                    )
                },
            )
            .collect();

        let metric_var_id = if base.metric().var_id().is_none() {
            None
        } else {
            original_num_var_to_projected
                .get(base.metric().var_id().unwrap().index())
                .and_then(|mapped| *mapped)
                .map(VariableIndex::from_usize)
        };

        let mutexes = projected_mutex_groups(base, &original_var_to_projected);
        let task = Task::new(TaskParts {
            metric: Metric::new(base.metric().is_min(), metric_var_id),
            variables,
            numeric_variables,
            goals,
            mutexes,
            state: projected_prop_values,
            numeric_state: projected_numeric_values,
            operators,
            axioms,
            comparison_axioms,
            assignment_axioms,
            global_constraint: None,
        })
        .with_operator_costs(operator_costs);
        Ok(Self {
            task: Arc::new(task),
            base_operator_ids,
            projected_var_to_original,
            projected_num_var_to_original,
            base_num_variables: num_vars,
            base_num_numeric_variables: num_numeric_vars,
        })
    }

    /// The projected task.
    pub fn task(&self) -> &Arc<Task> {
        &self.task
    }

    /// Fill `projected_prop_values` and `projected_numeric_values` with the
    /// values the projected variables take in the state of the base task with
    /// `propositional_values` and `source_numeric_values`.
    pub fn project_state_values_into(
        &self,
        state: StateValues<'_>,
        projected_prop_values: &mut Vec<ExplicitValueIndex>,
        projected_numeric_values: &mut Vec<NumericValue>,
    ) {
        let source_numeric_values = state.numeric();
        assert_eq!(
            state.num_propositional(),
            self.base_num_variables,
            "a projected state needs one value per base propositional variable"
        );
        assert_eq!(
            source_numeric_values.len(),
            self.base_num_numeric_variables,
            "a projected state needs one value per base numeric variable"
        );

        projected_prop_values.clear();
        projected_prop_values.extend(
            self.projected_var_to_original
                .iter()
                .map(|&original_var_id| state.propositional(original_var_id.index())),
        );

        projected_numeric_values.clear();
        projected_numeric_values.reserve(self.projected_num_var_to_original.len());
        for &original_numeric_var in &self.projected_num_var_to_original {
            projected_numeric_values.push(source_numeric_values[original_numeric_var.index()]);
        }
    }

    /// The cheapest cost of an operator the projection keeps, the least cost
    /// of reaching the goal from a state that is no abstract goal; infinite
    /// when the projection keeps no operator.
    pub fn min_operator_cost(&self) -> f64 {
        let min_operator_cost = self
            .task
            .effective_operator_costs()
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_operator_cost >= 0.0,
            "operator costs are nonnegative, got {min_operator_cost}"
        );
        min_operator_cost
    }

    pub fn base_operator_id(&self, projected_operator_id: OperatorIndex) -> OperatorIndex {
        self.base_operator_ids[projected_operator_id.index()]
    }

    pub fn base_operator_ids(&self) -> &[OperatorIndex] {
        &self.base_operator_ids
    }

    /// The base task variable the projected propositional variable stands for.
    pub fn original_variable(&self, projected_var: usize) -> VariableIndex {
        self.projected_var_to_original[projected_var]
    }

    /// Whether a numeric variable that operators change, rather than a
    /// constant, cost or derived one, is projected.
    pub fn has_regular_numeric_variables(&self) -> bool {
        self.task
            .numeric_variables()
            .iter()
            .any(|variable| variable.numeric_type() == NumericType::Regular)
    }
}

/// The base task's mutex groups restricted to the projected variables and
/// renumbered into their ids; a group left with fewer than two facts says
/// nothing and is dropped.
fn projected_mutex_groups(
    base: &Task,
    original_var_to_projected: &[Option<usize>],
) -> Vec<Vec<ExplicitFact>> {
    base.mutexes()
        .iter()
        .map(|group| {
            group
                .iter()
                .filter_map(|fact| project_fact(fact, original_var_to_projected))
                .collect::<Vec<_>>()
        })
        .filter(|group| group.len() >= 2)
        .collect()
}

fn push_unique_mapping(
    original_id: VariableIndex,
    projected_to_original: &mut Vec<VariableIndex>,
    original_to_projected: &mut [Option<usize>],
) {
    if original_to_projected[original_id.index()].is_none() {
        original_to_projected[original_id.index()] = Some(projected_to_original.len());
        projected_to_original.push(original_id);
    }
}

fn include_restricted_comparison_operands(
    task: &Task,
    comparison_axiom_id: usize,
    projected_num_var_to_original: &mut Vec<VariableIndex>,
    original_num_var_to_projected: &mut [Option<usize>],
) {
    let comparison_axiom = &task.comparison_axioms()[comparison_axiom_id];
    for numeric_var_id in [
        comparison_axiom.left_var_id(),
        comparison_axiom.right_var_id(),
    ] {
        assert_ne!(
            task.numeric_type(numeric_var_id),
            NumericType::Derived,
            "restricted task validation excludes derived comparison operands"
        );
        push_unique_mapping(
            numeric_var_id,
            projected_num_var_to_original,
            original_num_var_to_projected,
        );
    }
}

/// Adds every comparison over the projected numeric variables: its variable,
/// and its constant operand.
///
/// A precondition on such a comparison constrains the projection, as numeric
/// Fast Downward checks numeric preconditions on the numeric variables of a
/// pattern. Without the comparison variable the projection drops the
/// precondition, and a numeric pattern variable no operator condition reads
/// adds nothing to the heuristic.
fn include_comparisons_over_projected_variables(task: &Task, projection: VariableProjection<'_>) {
    let is_constant =
        |numeric_var_id: VariableIndex| task.numeric_type(numeric_var_id) == NumericType::Constant;
    for (comparison_axiom_id, comparison_axiom) in task.comparison_axioms().iter().enumerate() {
        let operands = [
            comparison_axiom.left_var_id(),
            comparison_axiom.right_var_id(),
        ];
        let projected = |numeric_var_id: VariableIndex| {
            projection.original_num_var_to_projected[numeric_var_id.index()].is_some()
        };
        let covered = operands
            .iter()
            .all(|&operand| projected(operand) || is_constant(operand));
        let reads_projection = operands
            .iter()
            .any(|&operand| projected(operand) && !is_constant(operand));
        if !(covered && reads_projection) {
            continue;
        }
        push_unique_mapping(
            comparison_axiom.affected_var_id(),
            projection.projected_var_to_original,
            projection.original_var_to_projected,
        );
        include_restricted_comparison_operands(
            task,
            comparison_axiom_id,
            projection.projected_num_var_to_original,
            projection.original_num_var_to_projected,
        );
    }
}

/// The pattern as membership sets, which is the form the goal walk queries it
/// in.
struct PatternMembers {
    regular: BTreeSet<VariableIndex>,
    numeric: BTreeSet<VariableIndex>,
}

/// The variable index maps between a task and its projection, both kinds and
/// both directions: `projected_*_to_original` gives every projected id the
/// original variable it stands for, `original_*_to_projected` the inverse for
/// the variables the pattern reached. The goal walk extends them as it pulls in
/// the variables a restricted goal depends on.
struct VariableProjection<'a> {
    projected_var_to_original: &'a mut Vec<VariableIndex>,
    original_var_to_projected: &'a mut [Option<usize>],
    projected_num_var_to_original: &'a mut Vec<VariableIndex>,
    original_num_var_to_projected: &'a mut [Option<usize>],
}

/// The goal facts the projection keeps, restricted to the pattern.
///
/// A goal the pattern omits is dropped, which relaxes the projected task and keeps
/// its distances admissible. A goal on a *condition* variable is kept as soon as
/// the pattern holds either the variable itself or one of the comparison's
/// operands, and pulls the operands in with it: the projected task has to be able
/// to compute the comparison it is asked to reach.
///
/// A goal on a derived variable does not appear here. This used to walk the rule
/// deriving it and pull the body variables into the projection;
/// `validate_abstractable_goal` refuses such a task instead.
fn collect_restricted_projected_goals(
    task: &Task,
    pattern: &Pattern,
    projection: VariableProjection<'_>,
) -> Vec<ExplicitFact> {
    let members = PatternMembers {
        regular: pattern.regular.iter().copied().collect(),
        numeric: pattern.numeric.iter().copied().collect(),
    };
    let mut goals = Vec::new();

    for &fact in task.goals() {
        let comparison_axiom_id = task.numeric_conditions().id_for_var(fact.var_index());
        let keep = match comparison_axiom_id {
            Some(comparison_axiom_id) => {
                let comparison_axiom = &task.comparison_axioms()[comparison_axiom_id];
                members.regular.contains(&fact.var_index())
                    || [
                        comparison_axiom.left_var_id(),
                        comparison_axiom.right_var_id(),
                    ]
                    .iter()
                    .any(|id| members.numeric.contains(id))
            }
            None => members.regular.contains(&fact.var_index()),
        };
        if !keep {
            continue;
        }
        push_unique_mapping(
            fact.var_index(),
            projection.projected_var_to_original,
            projection.original_var_to_projected,
        );
        if let Some(comparison_axiom_id) = comparison_axiom_id {
            include_restricted_comparison_operands(
                task,
                comparison_axiom_id,
                projection.projected_num_var_to_original,
                projection.original_num_var_to_projected,
            );
        }
        goals.push(fact);
    }

    goals.sort();
    goals.dedup();
    goals
}

/// Projecting renames a variable; it never changes what kind of variable it is,
/// so the namespace carries over.
/// `fact` in the projected ids; `None` when its variable is not projected.
fn project_fact(fact: &ExplicitFact, var_map: &[Option<usize>]) -> Option<ExplicitFact> {
    var_map[fact.var()]
        .map(|mapped| ExplicitFact::in_namespace(fact.namespace(), mapped, fact.value()))
}

fn project_effect(effect: &Effect, var_map: &[Option<usize>]) -> Option<Effect> {
    let mapped_var = var_map
        .get(effect.var_id().index())
        .and_then(|mapped| *mapped)?;
    let conditions: Vec<ExplicitFact> = effect
        .conditions()
        .iter()
        .filter_map(|fact| project_fact(fact, var_map))
        .collect();
    Some(Effect::new(
        conditions,
        VariableIndex::from_usize(mapped_var),
        effect.precondition_value(),
        effect.value(),
    ))
}

fn project_assignment_effect(
    effect: &AssignmentEffect,
    var_map: &[Option<usize>],
    num_var_map: &[Option<usize>],
) -> Option<AssignmentEffect> {
    let affected = num_var_map
        .get(effect.affected_var_id().index())
        .and_then(|mapped| *mapped)?;
    let source = num_var_map
        .get(effect.var_id().index())
        .and_then(|mapped| *mapped)?;
    let conditions: Vec<ExplicitFact> = effect
        .conditions()
        .iter()
        .filter_map(|fact| project_fact(fact, var_map))
        .collect();
    Some(AssignmentEffect::new(
        VariableIndex::from_usize(affected),
        effect.operation().clone(),
        VariableIndex::from_usize(source),
        effect.is_conditional(),
        conditions,
    ))
}

fn project_restricted_operator(
    operator: &Operator,
    var_map: &[Option<usize>],
    num_var_map: &[Option<usize>],
) -> Option<Operator> {
    let effects: Vec<Effect> = operator
        .effects()
        .iter()
        .filter_map(|effect| project_effect(effect, var_map))
        .collect();
    let assignment_effects: Vec<AssignmentEffect> = operator
        .assignment_effects()
        .iter()
        .filter_map(|effect| project_assignment_effect(effect, var_map, num_var_map))
        .collect();
    if effects.is_empty() && assignment_effects.is_empty() {
        return None;
    }

    let preconditions = operator
        .preconditions()
        .iter()
        .filter_map(|fact| project_fact(fact, var_map))
        .collect();
    Some(Operator::new(
        operator.name().to_string(),
        preconditions,
        effects,
        assignment_effects,
        operator.cost(),
    ))
}

fn project_propositional_axiom(
    axiom: &PropositionalAxiom,
    var_map: &[Option<usize>],
) -> Option<PropositionalAxiom> {
    let mapped_var = var_map
        .get(axiom.var_id().index())
        .and_then(|mapped| *mapped)?;
    let conditions = axiom
        .conditions()
        .iter()
        .filter_map(|fact| project_fact(fact, var_map))
        .collect();
    Some(PropositionalAxiom::new(
        conditions,
        VariableIndex::from_usize(mapped_var),
        axiom.precondition_value(),
        axiom.effect_value(),
    ))
}

fn project_restricted_comparison_axiom(
    task: &Task,
    comparison_axiom_id: usize,
    var_map: &[Option<usize>],
    num_var_map: &[Option<usize>],
) -> Option<ComparisonAxiom> {
    let axiom = &task.comparison_axioms()[comparison_axiom_id];
    let affected = var_map
        .get(axiom.affected_var_id().index())
        .copied()
        .flatten()?;
    let left = num_var_map
        .get(axiom.left_var_id().index())
        .copied()
        .flatten()?;
    let right = num_var_map
        .get(axiom.right_var_id().index())
        .copied()
        .flatten()?;
    Some(ComparisonAxiom::new(
        VariableIndex::from_usize(affected),
        VariableIndex::from_usize(left),
        VariableIndex::from_usize(right),
        axiom.operator().clone(),
    ))
}

fn project_assignment_axiom(
    axiom: &AssignmentAxiom,
    num_var_map: &[Option<usize>],
) -> Option<AssignmentAxiom> {
    let affected = num_var_map
        .get(axiom.affected_var_id().index())
        .and_then(|mapped| *mapped)?;
    let mapped = |var: VariableIndex| {
        num_var_map
            .get(var.index())
            .copied()
            .flatten()
            .map(VariableIndex::from_usize)
    };
    let expression = match &axiom.expression {
        NumericExpression::Binary {
            operator,
            left,
            right,
        } => NumericExpression::Binary {
            operator: operator.clone(),
            left: mapped(*left)?,
            right: mapped(*right)?,
        },
        NumericExpression::Linear(terms) => NumericExpression::Linear(
            terms
                .iter()
                .map(|(var, coefficient)| Some((mapped(*var)?, *coefficient)))
                .collect::<Option<_>>()?,
        ),
    };
    Some(AssignmentAxiom {
        affected_var_id: VariableIndex::from_usize(affected),
        expression,
    })
}

fn normalize_projected_variable_layers(
    num_variables: usize,
    numeric_variables: &[NumericVariable],
    comparison_axioms: &[ComparisonAxiom],
    axioms: &[PropositionalAxiom],
) -> Vec<Option<usize>> {
    let last_arithmetic_layer = numeric_variables
        .iter()
        .map(NumericVariable::axiom_layer)
        .max()
        .map_or(-1, |x| x.map_or(-1, |y| y as i32));
    let comparison_layer = if comparison_axioms.is_empty() {
        None
    } else {
        Some((last_arithmetic_layer + 1) as usize)
    };
    let base_propositional_layer = if comparison_axioms.is_empty() {
        Some((last_arithmetic_layer + 1) as usize)
    } else {
        Some(comparison_layer.map_or(0, |x| x + 1))
    };

    let mut affects_comparison = vec![false; num_variables];
    for axiom in comparison_axioms {
        let var_id = axiom.affected_var_id();
        if var_id.index() < num_variables {
            affects_comparison[var_id.index()] = true;
        }
    }

    let mut axioms_by_var: Vec<Vec<&PropositionalAxiom>> = vec![Vec::new(); num_variables];
    for axiom in axioms {
        let affected_var = axiom.var_id();
        if affected_var.index() < num_variables {
            axioms_by_var[affected_var.index()].push(axiom);
        }
    }

    let mut layers = vec![None; num_variables];
    let mut visiting = vec![false; num_variables];

    fn compute_layer(
        var_id: VariableIndex,
        layers: &mut [Option<usize>],
        visiting: &mut [bool],
        affects_comparison: &[bool],
        axioms_by_var: &[Vec<&PropositionalAxiom>],
        comparison_layer: Option<usize>,
        base_propositional_layer: Option<usize>,
    ) -> Option<usize> {
        if layers[var_id.index()].is_some() {
            return layers[var_id.index()];
        }
        if affects_comparison[var_id.index()] {
            layers[var_id.index()] = comparison_layer;
            return comparison_layer;
        }
        if axioms_by_var[var_id.index()].is_empty() {
            return None;
        }
        if visiting[var_id.index()] {
            return base_propositional_layer;
        }

        visiting[var_id.index()] = true;
        let mut layer = base_propositional_layer;
        for axiom in &axioms_by_var[var_id.index()] {
            for condition in axiom.conditions() {
                let condition_var = condition.var();
                if condition_var >= layers.len() {
                    continue;
                }
                let dependency_layer = compute_layer(
                    VariableIndex::from_usize(condition_var),
                    layers,
                    visiting,
                    affects_comparison,
                    axioms_by_var,
                    comparison_layer,
                    base_propositional_layer,
                );
                if dependency_layer.is_some() {
                    layer = layer.max(dependency_layer.map(|x| x + 1));
                }
            }
        }
        visiting[var_id.index()] = false;
        layers[var_id.index()] = layer;
        layer
    }

    for var_id in 0..num_variables {
        if affects_comparison[var_id] || !axioms_by_var[var_id].is_empty() {
            compute_layer(
                VariableIndex::from_usize(var_id),
                &mut layers,
                &mut visiting,
                &affects_comparison,
                &axioms_by_var,
                comparison_layer,
                base_propositional_layer,
            );
        }
    }

    layers
}

/// Rejects a task whose goal an abstraction cannot express, naming the variable
/// that makes it one.
///
/// Every abstraction family here supports a *conjunctive* goal: a set of facts an
/// abstract operator can bring about. A propositional variable an axiom writes is
/// not one. No abstract operator ever writes it — abstract operators come from the
/// task's operators, and a derived variable has no operator writing it either — so
/// an abstraction over such a goal has nothing to reach, and reports the goal
/// unreachable or drops it. Both answers have been wrong in this tree before, and
/// both are silent, so the family refuses the task by name instead.
///
/// A numeric comparison is *not* affected, which is the point of the distinction:
/// its variable is derived in the SAS sense too, but it is
/// [`FactNamespace::Condition`](planforge_sas::numeric_task::FactNamespace::Condition),
/// and the abstraction machinery reasons about the comparison itself — refining
/// its operands is what the numeric CEGAR loop does all day.
///
/// What is left, then, is a `:derived` predicate the problem puts in its goal, and
/// the `@goal-reachable` predicate the translator invents for a disjunctive,
/// quantified or nested goal. Every non-abstraction configuration — blind, `ff`,
/// `lmcutnumeric` — still solves both, since it tests the goal fact in a state the
/// axiom evaluator has closed.
pub fn validate_abstractable_goal(task: &Task) -> std::result::Result<(), TransformError> {
    for fact in task.goals() {
        if fact.is_condition() {
            continue;
        }
        let variable = &task.variables()[fact.var()];
        if let Some(layer) = variable.axiom_layer() {
            return Err(TransformError::Unsupported(format!(
                "abstractions support conjunctive goals only: goal fact {fact:?} names the \
                 derived variable {:?} ({:?}) in axiom layer {layer}, which no operator writes",
                variable.name(),
                task.fact_name(fact)
            )));
        }
    }
    Ok(())
}
