use std::cell::{Ref, RefCell, RefMut};
use std::fmt;
use std::rc::Rc;

use anyhow::{Result, anyhow, bail};

use planners_sas::numeric::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, PropositionalAxiom,
};
use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitVariable,
    Metric, NumericType, NumericVariable, Operator,
};

#[derive(Debug, Clone)]
pub enum RestrictedTaskBuildError {
    NumericDomainSizeMismatch {
        provided: usize,
        expected: usize,
    },
    ConditionalNumericEffect {
        operator_id: usize,
        numeric_var_id: usize,
    },
    NonConstantNumericEffectRhs {
        operator_id: usize,
        rhs_var_id: usize,
    },
    UnsupportedNumericEffectOperation {
        operator_id: usize,
        numeric_var_id: usize,
        operation: AssignmentOperation,
    },
    UnsupportedDerivedRootAxiom {
        numeric_var_id: usize,
        operator: CalOperator,
    },
    MissingAssignmentAxiom {
        numeric_var_id: usize,
    },
}

impl fmt::Display for RestrictedTaskBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NumericDomainSizeMismatch { provided, expected } => {
                write!(
                    f,
                    "numeric domain size length mismatch for restricted task: {provided} != {expected}"
                )
            }
            Self::ConditionalNumericEffect {
                operator_id,
                numeric_var_id,
            } => write!(
                f,
                "restricted task requires unconditional numeric effects, got conditional effect on n{numeric_var_id} in op {operator_id}"
            ),
            Self::NonConstantNumericEffectRhs {
                operator_id,
                rhs_var_id,
            } => write!(
                f,
                "restricted task requires constant numeric effect RHS, got n{rhs_var_id} in op {operator_id}"
            ),
            Self::UnsupportedNumericEffectOperation {
                operator_id,
                numeric_var_id,
                operation,
            } => write!(
                f,
                "restricted task supports only +=/-= numeric effects, got {operation:?} on n{numeric_var_id} in op {operator_id}"
            ),
            Self::UnsupportedDerivedRootAxiom {
                numeric_var_id,
                operator,
            } => write!(
                f,
                "restricted task supports derived roots defined by sums/differences only, got {operator:?} for n{numeric_var_id}"
            ),
            Self::MissingAssignmentAxiom { numeric_var_id } => {
                write!(
                    f,
                    "restricted task could not find assignment axiom for derived root n{numeric_var_id}"
                )
            }
        }
    }
}

impl std::error::Error for RestrictedTaskBuildError {}

pub struct RestrictedTask<'task> {
    base: &'task dyn AbstractNumericTask,
    numeric_variables: Vec<NumericVariable>,
    operators: Vec<Operator>,
    state: Rc<RefCell<Vec<usize>>>,
    numeric_state: Rc<RefCell<Vec<f64>>>,
    promoted_numeric_roots: Vec<bool>,
    base_numeric_var_count: usize,
    synthetic_constant_values: Vec<f64>,
}

impl<'task> RestrictedTask<'task> {
    pub fn new(
        base: &'task dyn AbstractNumericTask,
        numeric_domain_sizes: &[usize],
    ) -> Result<Self, RestrictedTaskBuildError> {
        let num_numeric_vars = base.numeric_variables().len();
        if numeric_domain_sizes.len() != num_numeric_vars {
            return Err(RestrictedTaskBuildError::NumericDomainSizeMismatch {
                provided: numeric_domain_sizes.len(),
                expected: num_numeric_vars,
            });
        }

        validate_simple_numeric_task(base)?;

        let mut promoted_numeric_roots = vec![false; num_numeric_vars];
        for (numeric_var_id, numeric_var) in base.numeric_variables().iter().enumerate() {
            if numeric_domain_sizes[numeric_var_id] > 1
                && numeric_var.get_type() == &NumericType::Derived
            {
                promoted_numeric_roots[numeric_var_id] = true;
            }
        }

        let mut numeric_variables = base.numeric_variables().clone();
        let mut initial_numeric_values = base.get_initial_numeric_state_values().to_vec();
        for (numeric_var_id, promoted) in promoted_numeric_roots.iter().copied().enumerate() {
            if promoted {
                let old = &numeric_variables[numeric_var_id];
                numeric_variables[numeric_var_id] = NumericVariable::new(
                    old.name().to_string(),
                    NumericType::Regular,
                    old.axiom_layer(),
                );
            }
        }

        let base_numeric_var_count = numeric_variables.len();
        let mut synthetic_constant_values: Vec<f64> = Vec::new();
        let mut operators = Vec::with_capacity(base.get_operators().len());
        for (operator_id, op) in base.get_operators().iter().enumerate() {
            operators.push(build_restricted_operator(
                base,
                op,
                operator_id,
                &promoted_numeric_roots,
                &mut numeric_variables,
                &mut initial_numeric_values,
                base_numeric_var_count,
                &mut synthetic_constant_values,
            )?);
        }

        Ok(Self {
            base,
            numeric_variables,
            operators,
            state: Rc::new(RefCell::new(
                base.get_initial_propositional_state_values().to_vec(),
            )),
            numeric_state: Rc::new(RefCell::new(initial_numeric_values)),
            promoted_numeric_roots,
            base_numeric_var_count,
            synthetic_constant_values,
        })
    }

    pub fn promoted_numeric_roots(&self) -> &[bool] {
        &self.promoted_numeric_roots
    }

    pub fn base_numeric_var_count(&self) -> usize {
        self.base_numeric_var_count
    }

    pub fn synthetic_constant_values(&self) -> &[f64] {
        &self.synthetic_constant_values
    }
}

fn validate_simple_numeric_task(
    task: &dyn AbstractNumericTask,
) -> Result<(), RestrictedTaskBuildError> {
    for (operator_id, op) in task.get_operators().iter().enumerate() {
        for effect in op.assignment_effects() {
            if effect.is_conditional() || !effect.conditions().is_empty() {
                return Err(RestrictedTaskBuildError::ConditionalNumericEffect {
                    operator_id,
                    numeric_var_id: effect.affected_var_id(),
                });
            }
            let rhs_var_id = effect.var_id();
            if task
                .numeric_variables()
                .get(rhs_var_id)
                .is_none_or(|var| var.get_type() != &NumericType::Constant)
            {
                return Err(RestrictedTaskBuildError::NonConstantNumericEffectRhs {
                    operator_id,
                    rhs_var_id,
                });
            }
            match effect.operation() {
                AssignmentOperation::Plus | AssignmentOperation::Minus => {}
                other => {
                    return Err(
                        RestrictedTaskBuildError::UnsupportedNumericEffectOperation {
                            operator_id,
                            numeric_var_id: effect.affected_var_id(),
                            operation: other.clone(),
                        },
                    );
                }
            }
        }
    }
    Ok(())
}

fn build_restricted_operator(
    task: &dyn AbstractNumericTask,
    op: &Operator,
    operator_id: usize,
    promoted_numeric_roots: &[bool],
    numeric_variables: &mut Vec<NumericVariable>,
    initial_numeric_values: &mut Vec<f64>,
    base_numeric_var_count: usize,
    synthetic_constant_values: &mut Vec<f64>,
) -> Result<Operator, RestrictedTaskBuildError> {
    let mut assignment_effects = op.assignment_effects().clone();
    let effects_by_var = effects_by_var(op.assignment_effects(), task.numeric_variables().len());

    for (numeric_var_id, promoted) in promoted_numeric_roots.iter().copied().enumerate() {
        if !promoted {
            continue;
        }
        let delta = constant_delta_for_var(task, &effects_by_var, numeric_var_id)?;
        if delta == 0.0 {
            continue;
        }
        let (operation, rhs_var_id) = assignment_effect_for_delta(
            delta,
            numeric_variables,
            initial_numeric_values,
            base_numeric_var_count,
            synthetic_constant_values,
        );
        assignment_effects.push(AssignmentEffect::new(
            numeric_var_id,
            operation,
            rhs_var_id,
            false,
            vec![],
        ));
    }

    let _ = operator_id;
    Ok(Operator::new(
        op.name().to_string(),
        op.preconditions().clone(),
        op.effects().clone(),
        assignment_effects,
        op.cost(),
    ))
}

fn effects_by_var(
    effects: &[AssignmentEffect],
    num_numeric_vars: usize,
) -> Vec<Vec<&AssignmentEffect>> {
    let mut by_var = vec![Vec::new(); num_numeric_vars];
    for effect in effects {
        if effect.affected_var_id() < num_numeric_vars {
            by_var[effect.affected_var_id()].push(effect);
        }
    }
    by_var
}

fn constant_effect_delta(task: &dyn AbstractNumericTask, effect: &AssignmentEffect) -> Result<f64> {
    let rhs_value = *task
        .get_initial_numeric_state_values()
        .get(effect.var_id())
        .ok_or_else(|| anyhow!("numeric effect RHS out of bounds"))?;
    Ok(match effect.operation() {
        AssignmentOperation::Plus => rhs_value,
        AssignmentOperation::Minus => -rhs_value,
        other => bail!("unsupported numeric effect operation in restricted task: {other:?}"),
    })
}

fn constant_delta_for_var(
    task: &dyn AbstractNumericTask,
    effects_by_var: &[Vec<&AssignmentEffect>],
    numeric_var_id: usize,
) -> Result<f64, RestrictedTaskBuildError> {
    let mut visiting = vec![false; task.numeric_variables().len()];
    constant_delta_for_var_rec(task, effects_by_var, numeric_var_id, &mut visiting)
}

fn constant_delta_for_var_rec(
    task: &dyn AbstractNumericTask,
    effects_by_var: &[Vec<&AssignmentEffect>],
    numeric_var_id: usize,
    visiting: &mut [bool],
) -> Result<f64, RestrictedTaskBuildError> {
    let numeric_var = task
        .numeric_variables()
        .get(numeric_var_id)
        .ok_or(RestrictedTaskBuildError::MissingAssignmentAxiom { numeric_var_id })?;
    match numeric_var.get_type() {
        NumericType::Constant => Ok(0.0),
        NumericType::Regular | NumericType::Cost => {
            let mut delta = 0.0;
            for effect in effects_by_var.get(numeric_var_id).into_iter().flatten() {
                delta += constant_effect_delta(task, effect).map_err(|_| {
                    RestrictedTaskBuildError::UnsupportedNumericEffectOperation {
                        operator_id: 0,
                        numeric_var_id,
                        operation: effect.operation().clone(),
                    }
                })?;
            }
            Ok(delta)
        }
        NumericType::Derived => {
            if visiting[numeric_var_id] {
                return Ok(0.0);
            }
            visiting[numeric_var_id] = true;
            let axiom = task
                .assignment_axioms()
                .iter()
                .find(|axiom| axiom.get_affected_var_id() == numeric_var_id)
                .ok_or(RestrictedTaskBuildError::MissingAssignmentAxiom { numeric_var_id })?;
            let lhs = constant_delta_for_var_rec(
                task,
                effects_by_var,
                axiom.get_left_var_id(),
                visiting,
            )?;
            let rhs = constant_delta_for_var_rec(
                task,
                effects_by_var,
                axiom.get_right_var_id(),
                visiting,
            )?;
            visiting[numeric_var_id] = false;
            match axiom.get_operator() {
                CalOperator::Sum => Ok(lhs + rhs),
                CalOperator::Difference => Ok(lhs - rhs),
                other => Err(RestrictedTaskBuildError::UnsupportedDerivedRootAxiom {
                    numeric_var_id,
                    operator: other.clone(),
                }),
            }
        }
    }
}

fn assignment_effect_for_delta(
    delta: f64,
    numeric_variables: &mut Vec<NumericVariable>,
    initial_numeric_values: &mut Vec<f64>,
    base_numeric_var_count: usize,
    synthetic_constant_values: &mut Vec<f64>,
) -> (AssignmentOperation, usize) {
    let abs_delta = delta.abs();
    let rhs_var_id = numeric_variables
        .iter()
        .enumerate()
        .find_map(|(candidate_id, var)| {
            (var.get_type() == &NumericType::Constant
                && initial_numeric_values
                    .get(candidate_id)
                    .is_some_and(|&value| (value - abs_delta).abs() <= 1e-9))
            .then_some(candidate_id)
        });

    let rhs_var_id = rhs_var_id.unwrap_or_else(|| {
        let new_id = numeric_variables.len();
        numeric_variables.push(NumericVariable::new(
            format!("restricted!const({abs_delta})"),
            NumericType::Constant,
            None,
        ));
        initial_numeric_values.push(abs_delta);
        if new_id >= base_numeric_var_count {
            synthetic_constant_values.push(abs_delta);
        }
        new_id
    });

    let operation = if delta >= 0.0 {
        AssignmentOperation::Plus
    } else {
        AssignmentOperation::Minus
    };
    (operation, rhs_var_id)
}

impl AbstractNumericTask for RestrictedTask<'_> {
    fn variables(&self) -> &Vec<ExplicitVariable> {
        self.base.variables()
    }

    fn numeric_variables(&self) -> &Vec<NumericVariable> {
        &self.numeric_variables
    }

    fn assignment_axioms(&self) -> &Vec<AssignmentAxiom> {
        self.base.assignment_axioms()
    }

    fn comparison_axioms(&self) -> &Vec<ComparisonAxiom> {
        self.base.comparison_axioms()
    }

    fn axioms(&self) -> &Vec<PropositionalAxiom> {
        self.base.axioms()
    }

    fn metric(&self) -> &Metric {
        self.base.metric()
    }

    fn get_num_variables(&self) -> usize {
        self.base.get_num_variables()
    }

    fn get_variable_name(&self, index: usize) -> Result<&str, &str> {
        self.base.get_variable_name(index)
    }

    fn get_variable_domain_size(&self, index: usize) -> Result<usize, &str> {
        self.base.get_variable_domain_size(index)
    }

    fn get_variable_axiom_layer(&self, index: usize) -> Result<Option<usize>, &str> {
        self.base.get_variable_axiom_layer(index)
    }

    fn get_variable_default_axiom_value(&self, index: usize) -> Result<usize, &str> {
        self.base.get_variable_default_axiom_value(index)
    }

    fn get_fact_name(&self, fact: &ExplicitFact) -> &str {
        self.base.get_fact_name(fact)
    }

    fn are_facts_mutex(&self, fact1: &ExplicitFact, fact2: &ExplicitFact) -> bool {
        self.base.are_facts_mutex(fact1, fact2)
    }

    fn get_operators(&self) -> &Vec<Operator> {
        &self.operators
    }

    fn get_operator_cost(&self, index: usize, is_axiom: bool) -> u64 {
        if is_axiom {
            0
        } else {
            self.operators.get(index).map_or(0, Operator::cost)
        }
    }

    fn get_operator_name(&self, index: usize, is_axiom: bool) -> &str {
        if is_axiom {
            "<axiom>"
        } else {
            self.operators.get(index).map_or("", Operator::name)
        }
    }

    fn get_num_operators(&self) -> usize {
        self.operators.len()
    }

    fn get_num_operator_preconditions(&self, index: usize, is_axiom: bool) -> usize {
        if is_axiom {
            self.base.get_num_operator_preconditions(index, is_axiom)
        } else {
            self.operators
                .get(index)
                .map(|operator| operator.preconditions().len())
                .unwrap_or(0)
        }
    }

    fn get_operator_precondition(
        &self,
        index: usize,
        precond_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact {
        if is_axiom {
            self.base
                .get_operator_precondition(index, precond_index, is_axiom)
        } else {
            &self.operators[index].preconditions()[precond_index]
        }
    }

    fn get_num_operator_effects(&self, index: usize, is_axiom: bool) -> usize {
        if is_axiom {
            self.base.get_num_operator_effects(index, is_axiom)
        } else {
            self.operators
                .get(index)
                .map(|operator| operator.effects().len())
                .unwrap_or(0)
        }
    }

    fn get_num_operator_effect_conditions(
        &self,
        index: usize,
        eff_index: usize,
        is_axiom: bool,
    ) -> usize {
        if is_axiom {
            self.base
                .get_num_operator_effect_conditions(index, eff_index, is_axiom)
        } else {
            self.operators[index].effects()[eff_index]
                .conditions()
                .len()
        }
    }

    fn get_operator_effect_condition(
        &self,
        index: usize,
        eff_index: usize,
        cond_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact {
        if is_axiom {
            self.base
                .get_operator_effect_condition(index, eff_index, cond_index, is_axiom)
        } else {
            &self.operators[index].effects()[eff_index].conditions()[cond_index]
        }
    }

    fn get_operator_effect(&self, index: usize, eff_index: usize, is_axiom: bool) -> &ExplicitFact {
        if is_axiom {
            self.base.get_operator_effect(index, eff_index, is_axiom)
        } else {
            let effect = &self.operators[index].effects()[eff_index];
            // The trait exposes an ExplicitFact view of effects only for legacy callers. Domain
            // abstractions use Operator::effects directly, so delegating to the base preserves
            // stable storage here.
            let _ = effect;
            self.base.get_operator_effect(index, eff_index, is_axiom)
        }
    }

    fn convert_operator_index(&self, index: usize, ancestor_task: &dyn AbstractNumericTask) {
        self.base.convert_operator_index(index, ancestor_task)
    }

    fn get_num_axioms(&self) -> usize {
        self.base.get_num_axioms()
    }

    fn get_num_goals(&self) -> usize {
        self.base.get_num_goals()
    }

    fn get_goal_fact(&self, index: usize) -> &ExplicitFact {
        self.base.get_goal_fact(index)
    }

    fn get_initial_propositional_state_values(&self) -> Ref<'_, Vec<usize>> {
        self.state.borrow()
    }

    fn get_initial_numeric_state_values(&self) -> Ref<'_, Vec<f64>> {
        self.numeric_state.borrow()
    }

    fn get_initial_propositional_state_values_mut(&self) -> RefMut<'_, Vec<usize>> {
        self.state.borrow_mut()
    }

    fn get_initial_numeric_state_values_mut(&self) -> RefMut<'_, Vec<f64>> {
        self.numeric_state.borrow_mut()
    }

    fn set_initial_propositional_state_values(&self, values: Vec<usize>) {
        *self.state.borrow_mut() = values;
    }

    fn set_initial_numeric_state_values(&self, values: Vec<f64>) {
        *self.numeric_state.borrow_mut() = values;
    }

    fn convert_ancestor_state_values(
        &self,
        ancestor_state_values: &[usize],
        ancestor_task: &dyn AbstractNumericTask,
    ) -> Vec<usize> {
        self.base
            .convert_ancestor_state_values(ancestor_state_values, ancestor_task)
    }

    fn get_num_cmp_axioms(&self) -> usize {
        self.base.get_num_cmp_axioms()
    }

    fn evaluated_initial_abstract_state_values(&self) -> Result<(Vec<usize>, Vec<f64>), String> {
        self.base.evaluated_initial_abstract_state_values()
    }
}
