use std::cell::{Ref, RefCell, RefMut};
use std::collections::{BTreeSet, HashSet};
use std::fmt;
use std::rc::Rc;

use planners_sas::numeric::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, PropositionalAxiom,
};
use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentEffect, Effect, ExplicitVariable, Fact, Metric, NumericType,
    NumericVariable, Operator,
};

use super::super::comparison_expression::{ArithOp, ComparisonTree, ComparisonTreeNode};

#[derive(Debug, Clone)]
enum ArithmeticExpr {
    Var(usize),
    Const(f64),
    Op {
        lhs: Box<ArithmeticExpr>,
        op: CalOperator,
        rhs: Box<ArithmeticExpr>,
    },
}

impl ArithmeticExpr {
    fn constant(value: f64) -> Self {
        Self::Const(value)
    }

    fn op(lhs: ArithmeticExpr, op: CalOperator, rhs: ArithmeticExpr) -> Self {
        if lhs.is_constant() && rhs.is_constant() {
            let lhs_value = lhs.evaluate(&[]);
            let rhs_value = rhs.evaluate(&[]);
            return Self::Const(apply_cal_operator(&op, lhs_value, rhs_value));
        }
        Self::Op {
            lhs: Box::new(lhs),
            op,
            rhs: Box::new(rhs),
        }
    }

    fn is_constant(&self) -> bool {
        matches!(self, Self::Const(_))
    }

    fn evaluate(&self, values: &[f64]) -> f64 {
        match self {
            Self::Var(var_id) => values[*var_id],
            Self::Const(value) => *value,
            Self::Op { lhs, op, rhs } => {
                apply_cal_operator(op, lhs.evaluate(values), rhs.evaluate(values))
            }
        }
    }

    fn evaluate_ignore_additive_consts(&self, values: &[f64]) -> f64 {
        match self {
            Self::Var(var_id) => values[*var_id],
            Self::Const(value) => *value,
            Self::Op { lhs, op, rhs } => match op {
                CalOperator::Sum => {
                    if lhs.is_constant() {
                        rhs.evaluate_ignore_additive_consts(values)
                    } else if rhs.is_constant() {
                        lhs.evaluate_ignore_additive_consts(values)
                    } else {
                        lhs.evaluate_ignore_additive_consts(values)
                            + rhs.evaluate_ignore_additive_consts(values)
                    }
                }
                CalOperator::Difference => {
                    if lhs.is_constant() {
                        -rhs.evaluate_ignore_additive_consts(values)
                    } else if rhs.is_constant() {
                        lhs.evaluate_ignore_additive_consts(values)
                    } else {
                        lhs.evaluate_ignore_additive_consts(values)
                            - rhs.evaluate_ignore_additive_consts(values)
                    }
                }
                CalOperator::Product => {
                    lhs.evaluate_ignore_additive_consts(values)
                        * rhs.evaluate_ignore_additive_consts(values)
                }
                CalOperator::Division => {
                    lhs.evaluate_ignore_additive_consts(values)
                        / rhs.evaluate_ignore_additive_consts(values)
                }
            },
        }
    }
}

#[derive(Debug, Clone)]
struct AuxiliaryNumericVar {
    helper_id: usize,
    source_numeric_var_id: usize,
    expr: ArithmeticExpr,
    initial_value: f64,
}

fn apply_cal_operator(op: &CalOperator, lhs: f64, rhs: f64) -> f64 {
    match op {
        CalOperator::Sum => lhs + rhs,
        CalOperator::Difference => lhs - rhs,
        CalOperator::Product => lhs * rhs,
        CalOperator::Division => lhs / rhs,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pattern {
    pub regular: Vec<usize>,
    pub numeric: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProjectedTaskBuildError {
    InvalidRegularVarId {
        provided: usize,
        len: usize,
    },
    InvalidNumericVarId {
        provided: usize,
        len: usize,
    },
    InvalidAuxiliaryNumericVarId {
        provided: usize,
        len: usize,
    },
    UnsupportedPatternNumericVarType {
        numeric_var_id: usize,
        numeric_type: NumericType,
    },
    MissingAssignmentAxiom {
        numeric_var_id: usize,
    },
    UnsupportedAssignmentOperator {
        axiom_id: usize,
        numeric_var_id: usize,
        operator: &'static str,
    },
    UnsupportedComparisonTree {
        comparison_axiom_id: usize,
        reason: &'static str,
    },
    UnsupportedOperatorEffect {
        operator_name: String,
        reason: &'static str,
    },
}

impl fmt::Display for ProjectedTaskBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRegularVarId { provided, len } => write!(
                formatter,
                "invalid projected propositional variable {provided}; task has {len} variables"
            ),
            Self::InvalidNumericVarId { provided, len } => write!(
                formatter,
                "invalid projected numeric variable {provided}; task has {len} numeric variables"
            ),
            Self::InvalidAuxiliaryNumericVarId { provided, len } => write!(
                formatter,
                "invalid projected auxiliary numeric variable {provided}; helper space has {len} numeric variables"
            ),
            Self::UnsupportedPatternNumericVarType {
                numeric_var_id,
                numeric_type,
            } => write!(
                formatter,
                "pattern numeric variable {numeric_var_id} has unsupported type {:?}; only regular numeric variables are accepted directly, while numeric-fd-style lifted auxiliaries are not implemented here",
                numeric_type
            ),
            Self::MissingAssignmentAxiom { numeric_var_id } => write!(
                formatter,
                "derived numeric variable {numeric_var_id} has no defining assignment axiom"
            ),
            Self::UnsupportedAssignmentOperator {
                axiom_id,
                numeric_var_id,
                operator,
            } => write!(
                formatter,
                "assignment axiom {axiom_id} for numeric variable {numeric_var_id} uses unsupported operator {operator}"
            ),
            Self::UnsupportedComparisonTree {
                comparison_axiom_id,
                reason,
            } => write!(
                formatter,
                "comparison axiom {comparison_axiom_id} is unsupported for projected tasks: {reason}"
            ),
            Self::UnsupportedOperatorEffect {
                operator_name,
                reason,
            } => write!(
                formatter,
                "operator {operator_name} uses unsupported numeric effects for projected tasks: {reason}"
            ),
        }
    }
}

impl std::error::Error for ProjectedTaskBuildError {}

pub struct ProjectedTask<'task> {
    base: &'task dyn AbstractNumericTask,
    variables: Vec<ExplicitVariable>,
    numeric_variables: Vec<NumericVariable>,
    assignment_axioms: Vec<AssignmentAxiom>,
    comparison_axioms: Vec<ComparisonAxiom>,
    axioms: Vec<PropositionalAxiom>,
    metric: Metric,
    operators: Vec<Operator>,
    operator_effect_facts: Vec<Vec<Fact>>,
    goals: Vec<Fact>,
    axiom_effect_facts: Vec<Fact>,
    state: Rc<RefCell<Vec<i32>>>,
    numeric_state: Rc<RefCell<Vec<f64>>>,
    projected_var_to_original: Vec<usize>,
    projected_num_var_to_original: Vec<Option<usize>>,
    original_var_to_projected: Vec<Option<usize>>,
    original_num_var_to_projected: Vec<Option<usize>>,
    is_auxiliary_num_var: Vec<bool>,
    is_auxiliary_constant: Vec<bool>,
    auxiliary_exprs: Vec<Option<ArithmeticExpr>>,
    variable_names: Vec<String>,
    fact_names: Vec<Vec<String>>,
}

impl<'task> ProjectedTask<'task> {
    pub fn new(
        base: &'task dyn AbstractNumericTask,
        pattern: &Pattern,
    ) -> Result<Self, ProjectedTaskBuildError> {
        let num_vars = base.variables().len();
        let num_numeric_vars = base.numeric_variables().len();

        let affected_to_assignment_axiom = build_assignment_axiom_lookup(base);
        let base_initial_numeric_values = {
            let values = base.get_initial_numeric_state_values();
            values.to_vec()
        };
        let auxiliary_numeric_vars = build_auxiliary_numeric_vars(
            base,
            &affected_to_assignment_axiom,
            &base_initial_numeric_values,
        )?;
        let helper_space_len = num_numeric_vars + auxiliary_numeric_vars.len();

        let mut projected_var_to_original: Vec<usize> = Vec::new();
        let mut projected_num_var_to_original: Vec<Option<usize>> = Vec::new();
        let mut original_var_to_projected = vec![None; num_vars];
        let mut original_num_var_to_projected = vec![None; num_numeric_vars];
        let mut is_auxiliary_num_var: Vec<bool> = Vec::new();
        let mut is_auxiliary_constant: Vec<bool> = Vec::new();
        let mut auxiliary_exprs: Vec<Option<ArithmeticExpr>> = Vec::new();
        let mut projected_aux_initial_values: Vec<Option<f64>> = Vec::new();

        for &var_id in &pattern.regular {
            if var_id >= num_vars {
                return Err(ProjectedTaskBuildError::InvalidRegularVarId {
                    provided: var_id,
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
            if numeric_var_id < num_numeric_vars {
                let numeric_type = base.numeric_variables()[numeric_var_id].get_type().clone();
                if numeric_type != NumericType::Regular {
                    return Err(ProjectedTaskBuildError::UnsupportedPatternNumericVarType {
                        numeric_var_id,
                        numeric_type,
                    });
                }
                push_projected_base_numeric_var(
                    numeric_var_id,
                    &mut projected_num_var_to_original,
                    &mut original_num_var_to_projected,
                    &mut is_auxiliary_num_var,
                    &mut is_auxiliary_constant,
                    &mut auxiliary_exprs,
                    &mut projected_aux_initial_values,
                );
            } else if let Some(auxiliary_numeric_var) =
                auxiliary_numeric_vars.get(numeric_var_id - num_numeric_vars)
            {
                push_projected_auxiliary_numeric_var(
                    auxiliary_numeric_var,
                    &mut projected_num_var_to_original,
                    &mut original_num_var_to_projected,
                    &mut is_auxiliary_num_var,
                    &mut is_auxiliary_constant,
                    &mut auxiliary_exprs,
                    &mut projected_aux_initial_values,
                );
            } else {
                return Err(ProjectedTaskBuildError::InvalidAuxiliaryNumericVarId {
                    provided: numeric_var_id,
                    len: helper_space_len,
                });
            }
        }

        for (numeric_var_id, numeric_var) in base.numeric_variables().iter().enumerate() {
            if matches!(
                numeric_var.get_type(),
                NumericType::Constant | NumericType::Cost
            ) {
                push_projected_base_numeric_var(
                    numeric_var_id,
                    &mut projected_num_var_to_original,
                    &mut original_num_var_to_projected,
                    &mut is_auxiliary_num_var,
                    &mut is_auxiliary_constant,
                    &mut auxiliary_exprs,
                    &mut projected_aux_initial_values,
                );
            }
        }

        let mut changed = true;
        while changed {
            changed = false;
            for (axiom_id, axiom) in base.assignment_axioms().iter().enumerate() {
                let affected = axiom.get_affected_var_id() as usize;
                if affected >= num_numeric_vars || original_num_var_to_projected[affected].is_some()
                {
                    continue;
                }
                ensure_supported_assignment_operator(axiom_id, affected, axiom.get_operator())?;
                let deps = regular_numeric_dependencies(
                    base,
                    affected,
                    &affected_to_assignment_axiom,
                    &original_num_var_to_projected,
                    &is_auxiliary_num_var,
                )?;
                if deps
                    .iter()
                    .all(|&dep| original_num_var_to_projected[dep].is_some())
                {
                    push_projected_base_numeric_var(
                        affected,
                        &mut projected_num_var_to_original,
                        &mut original_num_var_to_projected,
                        &mut is_auxiliary_num_var,
                        &mut is_auxiliary_constant,
                        &mut auxiliary_exprs,
                        &mut projected_aux_initial_values,
                    );
                    changed = true;
                }
            }
        }

        for comparison_axiom_id in 0..base.comparison_axioms().len() {
            let comparison_axiom = &base.comparison_axioms()[comparison_axiom_id];
            let tree = ComparisonTree::from_task(base, comparison_axiom_id).map_err(|_| {
                ProjectedTaskBuildError::UnsupportedComparisonTree {
                    comparison_axiom_id,
                    reason: "failed to build comparison tree",
                }
            })?;
            ensure_supported_comparison_tree(base, &tree)?;

            let left = usize::try_from(comparison_axiom.get_left_var_id()).unwrap_or(usize::MAX);
            let right = usize::try_from(comparison_axiom.get_right_var_id()).unwrap_or(usize::MAX);
            if left < num_numeric_vars
                && right < num_numeric_vars
                && original_num_var_to_projected[left].is_some()
                && original_num_var_to_projected[right].is_some()
            {
                let affected_var =
                    usize::try_from(comparison_axiom.get_affected_var_id()).unwrap_or(usize::MAX);
                if affected_var < num_vars {
                    push_unique_mapping(
                        affected_var,
                        &mut projected_var_to_original,
                        &mut original_var_to_projected,
                    );
                }
            }
        }

        let mut changed = true;
        while changed {
            changed = false;
            for axiom in base.axioms() {
                let affected = axiom.var_id() as usize;
                if affected >= num_vars || original_var_to_projected[affected].is_some() {
                    continue;
                }
                if axiom.conditions().iter().any(|fact| {
                    usize::try_from(fact.var())
                        .ok()
                        .and_then(|var_id| original_var_to_projected.get(var_id))
                        .and_then(|mapped| *mapped)
                        .is_some()
                }) {
                    push_unique_mapping(
                        affected,
                        &mut projected_var_to_original,
                        &mut original_var_to_projected,
                    );
                    changed = true;
                }
            }
        }

        let variables: Vec<ExplicitVariable> = projected_var_to_original
            .iter()
            .map(|&original| base.variables()[original].clone())
            .collect();

        let mut variable_names: Vec<String> = Vec::with_capacity(projected_var_to_original.len());
        let mut fact_names: Vec<Vec<String>> = Vec::with_capacity(projected_var_to_original.len());
        for &original_var_id in &projected_var_to_original {
            let variable_name = base
                .get_variable_name(original_var_id as i32)
                .unwrap_or("<projected-var>")
                .to_string();
            let domain_size = base
                .get_variable_domain_size(original_var_id as i32)
                .unwrap_or(0)
                .max(0) as usize;

            let var_fact_names = (0..domain_size)
                .map(|value| {
                    let original_fact = Fact::new(original_var_id as u32, value as i32);
                    let fact_name = base.get_fact_name(&original_fact);
                    if fact_name.is_empty() {
                        format!("{variable_name}={value}")
                    } else {
                        fact_name.to_string()
                    }
                })
                .collect();

            variable_names.push(variable_name);
            fact_names.push(var_fact_names);
        }

        let initial_prop_values = base.get_initial_propositional_state_values();
        let projected_prop_values: Vec<i32> = projected_var_to_original
            .iter()
            .map(|&original| initial_prop_values[original])
            .collect();
        drop(initial_prop_values);

        let mut numeric_variables: Vec<NumericVariable> =
            Vec::with_capacity(projected_num_var_to_original.len());
        let mut projected_numeric_values: Vec<f64> =
            Vec::with_capacity(projected_num_var_to_original.len());
        for projected_index in 0..projected_num_var_to_original.len() {
            if is_auxiliary_constant[projected_index] {
                numeric_variables.push(NumericVariable::new(
                    format!("projected-const-{projected_index}"),
                    NumericType::Constant,
                    -1,
                ));
                projected_numeric_values
                    .push(projected_aux_initial_values[projected_index].unwrap_or(0.0));
                continue;
            }

            let source_original = projected_num_var_to_original[projected_index]
                .expect("non-constant projected numeric vars must map to a base numeric var");
            if is_auxiliary_num_var[projected_index] {
                numeric_variables.push(NumericVariable::new(
                    base.numeric_variables()[source_original].name().to_string(),
                    NumericType::Regular,
                    -1,
                ));
                projected_numeric_values
                    .push(projected_aux_initial_values[projected_index].unwrap_or(0.0));
            } else {
                numeric_variables.push(base.numeric_variables()[source_original].clone());
                projected_numeric_values.push(base_initial_numeric_values[source_original]);
            }
        }

        let goals: Vec<Fact> = (0..usize::try_from(base.get_num_goals().max(0)).unwrap_or(0))
            .filter_map(|goal_index| {
                let goal = base.get_goal_fact(goal_index as i32);
                project_fact(goal, &original_var_to_projected)
            })
            .collect();

        let mut operators: Vec<Operator> = Vec::new();
        for operator in base.get_operators().iter() {
            if let Some(projected_operator) = project_operator(
                base,
                operator,
                &base_initial_numeric_values,
                &auxiliary_numeric_vars,
                &original_var_to_projected,
                &original_num_var_to_projected,
                &mut projected_num_var_to_original,
                &mut is_auxiliary_num_var,
                &mut is_auxiliary_constant,
                &mut auxiliary_exprs,
                &mut projected_aux_initial_values,
                &mut numeric_variables,
                &mut projected_numeric_values,
            )? {
                operators.push(projected_operator);
            }
        }

        let axioms: Vec<PropositionalAxiom> = base
            .axioms()
            .iter()
            .filter_map(|axiom| project_propositional_axiom(axiom, &original_var_to_projected))
            .collect();

        let comparison_axioms: Vec<ComparisonAxiom> = base
            .comparison_axioms()
            .iter()
            .filter_map(|axiom| {
                project_comparison_axiom(
                    axiom,
                    &original_var_to_projected,
                    &original_num_var_to_projected,
                )
            })
            .collect();

        let assignment_axioms: Vec<AssignmentAxiom> = base
            .assignment_axioms()
            .iter()
            .filter_map(|axiom| {
                project_assignment_axiom(
                    axiom,
                    &original_num_var_to_projected,
                    &is_auxiliary_num_var,
                )
            })
            .collect();

        let operator_effect_facts: Vec<Vec<Fact>> = operators
            .iter()
            .map(|operator| {
                operator
                    .effects()
                    .iter()
                    .map(|effect| Fact::new(effect.var_id(), effect.value() as i32))
                    .collect()
            })
            .collect();
        let axiom_effect_facts: Vec<Fact> = axioms
            .iter()
            .map(|axiom| Fact::new(axiom.var_id(), axiom.effect_value() as i32))
            .collect();

        let metric_var_id = if base.metric().var_id() < 0 {
            -1
        } else {
            original_num_var_to_projected
                .get(base.metric().var_id() as usize)
                .and_then(|mapped| *mapped)
                .map(|mapped| mapped as i32)
                .unwrap_or(-1)
        };

        Ok(Self {
            base,
            variables,
            numeric_variables,
            assignment_axioms,
            comparison_axioms,
            axioms,
            metric: Metric::new(base.metric().is_min(), metric_var_id),
            operators,
            operator_effect_facts,
            goals,
            axiom_effect_facts,
            state: Rc::new(RefCell::new(projected_prop_values)),
            numeric_state: Rc::new(RefCell::new(projected_numeric_values)),
            projected_var_to_original,
            projected_num_var_to_original,
            original_var_to_projected,
            original_num_var_to_projected,
            is_auxiliary_num_var,
            is_auxiliary_constant,
            auxiliary_exprs,
            variable_names,
            fact_names,
        })
    }
}

impl AbstractNumericTask for ProjectedTask<'_> {
    fn variables(&self) -> &Vec<ExplicitVariable> {
        &self.variables
    }

    fn numeric_variables(&self) -> &Vec<NumericVariable> {
        &self.numeric_variables
    }

    fn assignment_axioms(&self) -> &Vec<AssignmentAxiom> {
        &self.assignment_axioms
    }

    fn comparison_axioms(&self) -> &Vec<ComparisonAxiom> {
        &self.comparison_axioms
    }

    fn axioms(&self) -> &Vec<PropositionalAxiom> {
        &self.axioms
    }

    fn metric(&self) -> &Metric {
        &self.metric
    }

    fn get_num_variables(&self) -> i32 {
        self.variables.len() as i32
    }

    fn get_variable_name(&self, index: i32) -> Result<&str, &str> {
        let index = usize::try_from(index).map_err(|_| "Index out of bounds")?;
        self.variable_names
            .get(index)
            .map(|name| name.as_str())
            .ok_or("Index out of bounds")
    }

    fn get_variable_domain_size(&self, index: i32) -> Result<i32, &str> {
        let index = usize::try_from(index).map_err(|_| "Index out of bounds")?;
        self.variables
            .get(index)
            .map(|var| var.domain_size() as i32)
            .ok_or("Index out of bounds")
    }

    fn get_variable_axiom_layer(&self, index: i32) -> Result<i32, &str> {
        let index = usize::try_from(index).map_err(|_| "Index out of bounds")?;
        self.variables
            .get(index)
            .map(ExplicitVariable::axiom_layer)
            .ok_or("Index out of bounds")
    }

    fn get_variable_default_axiom_value(&self, index: i32) -> Result<i32, &str> {
        let original_index = self
            .projected_var_to_original
            .get(usize::try_from(index).map_err(|_| "Index out of bounds")?)
            .copied()
            .ok_or("Index out of bounds")?;
        self.base
            .get_variable_default_axiom_value(original_index as i32)
    }

    fn get_fact_name(&self, fact: &Fact) -> &str {
        let Some(var_fact_names) = self.fact_names.get(fact.var() as usize) else {
            return "";
        };
        var_fact_names
            .get(fact.value() as usize)
            .map_or("", String::as_str)
    }

    fn are_facts_mutex(&self, fact1: &Fact, fact2: &Fact) -> bool {
        let Some(original_fact1) = restore_fact(fact1, &self.projected_var_to_original) else {
            return false;
        };
        let Some(original_fact2) = restore_fact(fact2, &self.projected_var_to_original) else {
            return false;
        };
        self.base.are_facts_mutex(&original_fact1, &original_fact2)
    }

    fn get_operators(&self) -> &Vec<Operator> {
        &self.operators
    }

    fn get_operator_cost(&self, index: i32, is_axiom: bool) -> i32 {
        if is_axiom {
            0
        } else {
            usize::try_from(index)
                .ok()
                .and_then(|index| self.operators.get(index))
                .map(|operator| operator.cost() as i32)
                .unwrap_or(0)
        }
    }

    fn get_operator_name(&self, index: i32, is_axiom: bool) -> &str {
        if is_axiom {
            "<axiom>"
        } else {
            usize::try_from(index)
                .ok()
                .and_then(|index| self.operators.get(index))
                .map_or("", Operator::name)
        }
    }

    fn get_num_operators(&self) -> i32 {
        self.operators.len() as i32
    }

    fn get_num_operator_preconditions(&self, index: i32, is_axiom: bool) -> i32 {
        if is_axiom {
            usize::try_from(index)
                .ok()
                .and_then(|index| self.axioms.get(index))
                .map(|axiom| axiom.conditions().len() as i32)
                .unwrap_or(0)
        } else {
            usize::try_from(index)
                .ok()
                .and_then(|index| self.operators.get(index))
                .map(|operator| operator.preconditions().len() as i32)
                .unwrap_or(0)
        }
    }

    fn get_operator_precondition(&self, index: i32, precond_index: i32, is_axiom: bool) -> &Fact {
        let precond_index =
            usize::try_from(precond_index).expect("precondition index must be >= 0");
        if is_axiom {
            &self.axioms[usize::try_from(index).expect("axiom index must be >= 0")].conditions()
                [precond_index]
        } else {
            &self.operators[usize::try_from(index).expect("operator index must be >= 0")]
                .preconditions()[precond_index]
        }
    }

    fn get_num_operator_effects(&self, index: i32, is_axiom: bool) -> i32 {
        if is_axiom {
            usize::try_from(index)
                .ok()
                .and_then(|index| self.axioms.get(index))
                .map(|_| 1)
                .unwrap_or(0)
        } else {
            usize::try_from(index)
                .ok()
                .and_then(|index| self.operators.get(index))
                .map(|operator| operator.effects().len() as i32)
                .unwrap_or(0)
        }
    }

    fn get_num_operator_effect_conditions(
        &self,
        index: i32,
        eff_index: i32,
        is_axiom: bool,
    ) -> i32 {
        if is_axiom {
            0
        } else {
            self.operators[usize::try_from(index).expect("operator index must be >= 0")].effects()
                [usize::try_from(eff_index).expect("effect index must be >= 0")]
            .conditions()
            .len() as i32
        }
    }

    fn get_operator_effect_condition(
        &self,
        index: i32,
        eff_index: i32,
        cond_index: i32,
        is_axiom: bool,
    ) -> &Fact {
        assert!(
            !is_axiom,
            "axioms do not expose conditional effects separately"
        );
        &self.operators[usize::try_from(index).expect("operator index must be >= 0")].effects()
            [usize::try_from(eff_index).expect("effect index must be >= 0")]
        .conditions()[usize::try_from(cond_index).expect("condition index must be >= 0")]
    }

    fn get_operator_effect(&self, index: i32, eff_index: i32, is_axiom: bool) -> &Fact {
        if is_axiom {
            let effect_index = usize::try_from(eff_index).expect("effect index must be >= 0");
            assert_eq!(effect_index, 0, "axioms expose exactly one effect");
            &self.axiom_effect_facts[usize::try_from(index).expect("axiom index must be >= 0")]
        } else {
            &self.operator_effect_facts
                [usize::try_from(index).expect("operator index must be >= 0")]
                [usize::try_from(eff_index).expect("effect index must be >= 0")]
        }
    }

    fn convert_operator_index(&self, _index: i32, _ancestor_task: &dyn AbstractNumericTask) {}

    fn get_num_axioms(&self) -> i32 {
        self.axioms.len() as i32
    }

    fn get_num_goals(&self) -> i32 {
        self.goals.len() as i32
    }

    fn get_goal_fact(&self, index: i32) -> &Fact {
        &self.goals[usize::try_from(index).expect("goal index must be >= 0")]
    }

    fn get_initial_propositional_state_values(&self) -> Ref<'_, Vec<i32>> {
        self.state.borrow()
    }

    fn get_initial_numeric_state_values(&self) -> Ref<'_, Vec<f64>> {
        self.numeric_state.borrow()
    }

    fn get_initial_propositional_state_values_mut(&self) -> RefMut<'_, Vec<i32>> {
        self.state.borrow_mut()
    }

    fn get_initial_numeric_state_values_mut(&self) -> RefMut<'_, Vec<f64>> {
        self.numeric_state.borrow_mut()
    }

    fn set_initial_numeric_state_values(&self, values: Vec<f64>) {
        *self.numeric_state.borrow_mut() = values;
    }

    fn set_initial_propositional_state_values(&self, values: Vec<i32>) {
        *self.state.borrow_mut() = values;
    }

    fn convert_ancestor_state_values(
        &self,
        ancestor_state_values: &Vec<i32>,
        _ancestor_task: &dyn AbstractNumericTask,
    ) -> Vec<i32> {
        if ancestor_state_values.len() == self.variables.len() {
            return ancestor_state_values.clone();
        }
        self.projected_var_to_original
            .iter()
            .map(|&original| {
                ancestor_state_values
                    .get(original)
                    .copied()
                    .unwrap_or_default()
            })
            .collect()
    }

    fn get_num_cmp_axioms(&self) -> i32 {
        self.comparison_axioms.len() as i32
    }
}

fn push_unique_mapping(
    original_id: usize,
    projected_to_original: &mut Vec<usize>,
    original_to_projected: &mut [Option<usize>],
) {
    if original_to_projected[original_id].is_none() {
        original_to_projected[original_id] = Some(projected_to_original.len());
        projected_to_original.push(original_id);
    }
}

fn push_projected_base_numeric_var(
    original_id: usize,
    projected_to_original: &mut Vec<Option<usize>>,
    original_to_projected: &mut [Option<usize>],
    is_auxiliary_num_var: &mut Vec<bool>,
    is_auxiliary_constant: &mut Vec<bool>,
    auxiliary_exprs: &mut Vec<Option<ArithmeticExpr>>,
    projected_aux_initial_values: &mut Vec<Option<f64>>,
) {
    if original_to_projected[original_id].is_none() {
        original_to_projected[original_id] = Some(projected_to_original.len());
        projected_to_original.push(Some(original_id));
        is_auxiliary_num_var.push(false);
        is_auxiliary_constant.push(false);
        auxiliary_exprs.push(None);
        projected_aux_initial_values.push(None);
    }
}

fn push_projected_auxiliary_numeric_var(
    auxiliary_numeric_var: &AuxiliaryNumericVar,
    projected_to_original: &mut Vec<Option<usize>>,
    original_to_projected: &mut [Option<usize>],
    is_auxiliary_num_var: &mut Vec<bool>,
    is_auxiliary_constant: &mut Vec<bool>,
    auxiliary_exprs: &mut Vec<Option<ArithmeticExpr>>,
    projected_aux_initial_values: &mut Vec<Option<f64>>,
) {
    let original_id = auxiliary_numeric_var.source_numeric_var_id;
    if original_to_projected[original_id].is_none() {
        original_to_projected[original_id] = Some(projected_to_original.len());
        projected_to_original.push(Some(original_id));
        is_auxiliary_num_var.push(true);
        is_auxiliary_constant.push(false);
        auxiliary_exprs.push(Some(auxiliary_numeric_var.expr.clone()));
        projected_aux_initial_values.push(Some(auxiliary_numeric_var.initial_value));
    }
}

fn build_auxiliary_numeric_vars(
    task: &dyn AbstractNumericTask,
    assignment_lookup: &[Option<usize>],
    base_initial_numeric_values: &[f64],
) -> Result<Vec<AuxiliaryNumericVar>, ProjectedTaskBuildError> {
    struct Builder<'a> {
        task: &'a dyn AbstractNumericTask,
        assignment_lookup: &'a [Option<usize>],
        base_initial_numeric_values: &'a [f64],
        auxiliary_numeric_vars: Vec<AuxiliaryNumericVar>,
        derived_to_helper_id: Vec<Option<usize>>,
    }

    impl<'a> Builder<'a> {
        fn parse_numeric_expression(
            &mut self,
            numeric_var_id: usize,
        ) -> Result<ArithmeticExpr, ProjectedTaskBuildError> {
            match self.task.numeric_variables()[numeric_var_id].get_type() {
                NumericType::Regular => Ok(ArithmeticExpr::Var(numeric_var_id)),
                NumericType::Constant | NumericType::Cost => Ok(ArithmeticExpr::constant(
                    self.base_initial_numeric_values[numeric_var_id],
                )),
                NumericType::Derived => {
                    if let Some(helper_id) = self.derived_to_helper_id[numeric_var_id] {
                        return Ok(ArithmeticExpr::Var(helper_id));
                    }
                    let Some(axiom_id) = self.assignment_lookup[numeric_var_id] else {
                        return Err(ProjectedTaskBuildError::MissingAssignmentAxiom {
                            numeric_var_id,
                        });
                    };
                    let axiom = &self.task.assignment_axioms()[axiom_id];
                    let lhs = self.parse_numeric_expression(axiom.get_left_var_id() as usize)?;
                    let rhs = self.parse_numeric_expression(axiom.get_right_var_id() as usize)?;
                    let expr =
                        ArithmeticExpr::op(lhs.clone(), axiom.get_operator().clone(), rhs.clone());

                    if !lhs.is_constant() && !rhs.is_constant() {
                        let helper_id =
                            self.task.numeric_variables().len() + self.auxiliary_numeric_vars.len();
                        let mut helper_values = self.base_initial_numeric_values.to_vec();
                        for auxiliary_numeric_var in &self.auxiliary_numeric_vars {
                            helper_values.push(auxiliary_numeric_var.initial_value);
                        }
                        let initial_value = expr.evaluate(&helper_values);
                        self.auxiliary_numeric_vars.push(AuxiliaryNumericVar {
                            helper_id,
                            source_numeric_var_id: numeric_var_id,
                            expr: expr.clone(),
                            initial_value,
                        });
                        self.derived_to_helper_id[numeric_var_id] = Some(helper_id);
                        Ok(ArithmeticExpr::Var(helper_id))
                    } else {
                        Ok(expr)
                    }
                }
            }
        }
    }

    let mut builder = Builder {
        task,
        assignment_lookup,
        base_initial_numeric_values,
        auxiliary_numeric_vars: Vec::new(),
        derived_to_helper_id: vec![None; task.numeric_variables().len()],
    };

    for numeric_var_id in 0..task.numeric_variables().len() {
        if task.numeric_variables()[numeric_var_id].get_type() == &NumericType::Derived {
            builder.parse_numeric_expression(numeric_var_id)?;
        }
    }

    Ok(builder.auxiliary_numeric_vars)
}

fn build_assignment_axiom_lookup(task: &dyn AbstractNumericTask) -> Vec<Option<usize>> {
    let mut lookup = vec![None; task.numeric_variables().len()];
    for (axiom_id, axiom) in task.assignment_axioms().iter().enumerate() {
        let affected = axiom.get_affected_var_id() as usize;
        if affected < lookup.len() {
            lookup[affected] = Some(axiom_id);
        }
    }
    lookup
}

fn regular_numeric_dependencies(
    task: &dyn AbstractNumericTask,
    numeric_var_id: usize,
    assignment_lookup: &[Option<usize>],
    original_num_var_to_projected: &[Option<usize>],
    is_auxiliary_num_var: &[bool],
) -> Result<BTreeSet<usize>, ProjectedTaskBuildError> {
    let mut out = BTreeSet::new();
    let mut seen = HashSet::new();
    regular_numeric_dependencies_recursive(
        task,
        numeric_var_id,
        assignment_lookup,
        original_num_var_to_projected,
        is_auxiliary_num_var,
        &mut seen,
        &mut out,
    )?;
    Ok(out)
}

fn regular_numeric_dependencies_recursive(
    task: &dyn AbstractNumericTask,
    numeric_var_id: usize,
    assignment_lookup: &[Option<usize>],
    original_num_var_to_projected: &[Option<usize>],
    is_auxiliary_num_var: &[bool],
    seen: &mut HashSet<usize>,
    out: &mut BTreeSet<usize>,
) -> Result<(), ProjectedTaskBuildError> {
    if !seen.insert(numeric_var_id) {
        return Ok(());
    }

    if let Some(projected_id) = original_num_var_to_projected[numeric_var_id] {
        if is_auxiliary_num_var[projected_id] {
            out.insert(numeric_var_id);
            return Ok(());
        }
    }

    match task.numeric_variables()[numeric_var_id].get_type() {
        NumericType::Regular => {
            out.insert(numeric_var_id);
            Ok(())
        }
        NumericType::Constant | NumericType::Cost => Ok(()),
        NumericType::Derived => {
            let Some(axiom_id) = assignment_lookup[numeric_var_id] else {
                return Err(ProjectedTaskBuildError::MissingAssignmentAxiom { numeric_var_id });
            };
            let axiom = &task.assignment_axioms()[axiom_id];
            ensure_supported_assignment_operator(axiom_id, numeric_var_id, axiom.get_operator())?;
            regular_numeric_dependencies_recursive(
                task,
                axiom.get_left_var_id() as usize,
                assignment_lookup,
                original_num_var_to_projected,
                is_auxiliary_num_var,
                seen,
                out,
            )?;
            regular_numeric_dependencies_recursive(
                task,
                axiom.get_right_var_id() as usize,
                assignment_lookup,
                original_num_var_to_projected,
                is_auxiliary_num_var,
                seen,
                out,
            )?;
            Ok(())
        }
    }
}

fn ensure_supported_assignment_operator(
    axiom_id: usize,
    numeric_var_id: usize,
    operator: &CalOperator,
) -> Result<(), ProjectedTaskBuildError> {
    match operator {
        CalOperator::Sum | CalOperator::Product => Ok(()),
        CalOperator::Difference => Err(ProjectedTaskBuildError::UnsupportedAssignmentOperator {
            axiom_id,
            numeric_var_id,
            operator: "-",
        }),
        CalOperator::Division => Err(ProjectedTaskBuildError::UnsupportedAssignmentOperator {
            axiom_id,
            numeric_var_id,
            operator: "/",
        }),
    }
}

fn ensure_supported_comparison_tree(
    task: &dyn AbstractNumericTask,
    tree: &ComparisonTree,
) -> Result<(), ProjectedTaskBuildError> {
    for node in &tree.nodes {
        if let ComparisonTreeNode::Arith { op, .. } = node {
            if !matches!(op, ArithOp::Add | ArithOp::Mul) {
                return Err(ProjectedTaskBuildError::UnsupportedComparisonTree {
                    comparison_axiom_id: tree.comparison_axiom_id,
                    reason: "comparison tree uses subtraction or division",
                });
            }
        }
    }

    let left_constant = is_constant_expression(task, tree, tree.left_root);
    let right_constant = is_constant_expression(task, tree, tree.right_root);
    if left_constant || right_constant {
        Ok(())
    } else {
        Err(ProjectedTaskBuildError::UnsupportedComparisonTree {
            comparison_axiom_id: tree.comparison_axiom_id,
            reason: "comparison is not of the form x comp c",
        })
    }
}

fn is_constant_expression(
    task: &dyn AbstractNumericTask,
    tree: &ComparisonTree,
    root: usize,
) -> bool {
    let mut stack = vec![root];
    while let Some(node_id) = stack.pop() {
        match &tree.nodes[node_id] {
            ComparisonTreeNode::Leaf { numeric_var_id } => {
                let Some(var_id) = usize::try_from(*numeric_var_id).ok() else {
                    return false;
                };
                if !matches!(
                    task.numeric_variables()[var_id].get_type(),
                    NumericType::Constant | NumericType::Cost
                ) {
                    return false;
                }
            }
            ComparisonTreeNode::Arith { left, right, .. } => {
                stack.push(*left);
                stack.push(*right);
            }
        }
    }
    true
}

fn project_fact(fact: &Fact, var_map: &[Option<usize>]) -> Option<Fact> {
    var_map
        .get(fact.var() as usize)
        .and_then(|mapped| *mapped)
        .map(|mapped| Fact::new(mapped as u32, fact.value()))
}

fn restore_fact(fact: &Fact, projected_to_original: &[usize]) -> Option<Fact> {
    projected_to_original
        .get(fact.var() as usize)
        .map(|&original| Fact::new(original as u32, fact.value()))
}

fn project_effect(effect: &Effect, var_map: &[Option<usize>]) -> Option<Effect> {
    let mapped_var = var_map
        .get(effect.var_id() as usize)
        .and_then(|mapped| *mapped)?;
    let conditions: Vec<Fact> = effect
        .conditions()
        .iter()
        .filter_map(|fact| project_fact(fact, var_map))
        .collect();
    Some(Effect::new(
        conditions,
        mapped_var as u32,
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
        .get(effect.affected_var_id() as usize)
        .and_then(|mapped| *mapped)?;
    let source = num_var_map
        .get(effect.var_id() as usize)
        .and_then(|mapped| *mapped)?;
    let conditions: Vec<Fact> = effect
        .conditions()
        .iter()
        .filter_map(|fact| project_fact(fact, var_map))
        .collect();
    Some(AssignmentEffect::new(
        affected as u32,
        effect.operation().clone(),
        source as u32,
        effect.is_conditional(),
        conditions,
    ))
}

fn project_operator(
    base: &dyn AbstractNumericTask,
    operator: &Operator,
    base_initial_numeric_values: &[f64],
    auxiliary_numeric_vars: &[AuxiliaryNumericVar],
    var_map: &[Option<usize>],
    num_var_map: &[Option<usize>],
    projected_num_var_to_original: &mut Vec<Option<usize>>,
    is_auxiliary_num_var: &mut Vec<bool>,
    is_auxiliary_constant: &mut Vec<bool>,
    auxiliary_exprs: &mut Vec<Option<ArithmeticExpr>>,
    projected_aux_initial_values: &mut Vec<Option<f64>>,
    numeric_variables: &mut Vec<NumericVariable>,
    projected_numeric_values: &mut Vec<f64>,
) -> Result<Option<Operator>, ProjectedTaskBuildError> {
    let preconditions: Vec<Fact> = operator
        .preconditions()
        .iter()
        .filter_map(|fact| project_fact(fact, var_map))
        .collect();
    let effects: Vec<Effect> = operator
        .effects()
        .iter()
        .filter_map(|effect| project_effect(effect, var_map))
        .collect();
    let mut assignment_effects: Vec<AssignmentEffect> = operator
        .assignment_effects()
        .iter()
        .filter_map(|effect| project_assignment_effect(effect, var_map, num_var_map))
        .collect();

    let mut relevant = !effects.is_empty() || !assignment_effects.is_empty();

    if !auxiliary_numeric_vars.is_empty() {
        let mut additive_effects =
            vec![0.0; base.numeric_variables().len() + auxiliary_numeric_vars.len()];
        let mut has_assign_effect = false;

        for effect in operator.assignment_effects() {
            if effect.is_conditional() {
                return Err(ProjectedTaskBuildError::UnsupportedOperatorEffect {
                    operator_name: operator.name().to_string(),
                    reason: "conditional numeric effects are not supported",
                });
            }

            let source_var_id = effect.var_id() as usize;
            let source_type = base.numeric_variables()[source_var_id].get_type();
            let source_value = base_initial_numeric_values[source_var_id];

            match effect.operation() {
                planners_sas::numeric::numeric_task::AssignmentOperation::Assign => {
                    has_assign_effect = true;
                }
                planners_sas::numeric::numeric_task::AssignmentOperation::Plus => {
                    if source_type != &NumericType::Constant {
                        return Err(ProjectedTaskBuildError::UnsupportedOperatorEffect {
                            operator_name: operator.name().to_string(),
                            reason: "lifted numeric effects require constant right-hand sides",
                        });
                    }
                    additive_effects[effect.affected_var_id() as usize] += source_value;
                }
                planners_sas::numeric::numeric_task::AssignmentOperation::Minus => {
                    if source_type != &NumericType::Constant {
                        return Err(ProjectedTaskBuildError::UnsupportedOperatorEffect {
                            operator_name: operator.name().to_string(),
                            reason: "lifted numeric effects require constant right-hand sides",
                        });
                    }
                    additive_effects[effect.affected_var_id() as usize] -= source_value;
                }
                planners_sas::numeric::numeric_task::AssignmentOperation::Times
                | planners_sas::numeric::numeric_task::AssignmentOperation::Divide => {
                    return Err(ProjectedTaskBuildError::UnsupportedOperatorEffect {
                        operator_name: operator.name().to_string(),
                        reason: "non-additive numeric effects are not supported",
                    });
                }
            }
        }

        if has_assign_effect {
            return Err(ProjectedTaskBuildError::UnsupportedOperatorEffect {
                operator_name: operator.name().to_string(),
                reason: "mixed assignment and lifted additive effects are not supported",
            });
        }

        for auxiliary_numeric_var in auxiliary_numeric_vars {
            additive_effects[auxiliary_numeric_var.helper_id] = auxiliary_numeric_var
                .expr
                .evaluate_ignore_additive_consts(&additive_effects);
        }

        for auxiliary_numeric_var in auxiliary_numeric_vars {
            let projected_target = num_var_map[auxiliary_numeric_var.source_numeric_var_id];
            let Some(projected_target) = projected_target else {
                continue;
            };

            let delta = additive_effects[auxiliary_numeric_var.helper_id];
            if delta != 0.0 {
                let projected_constant = get_or_add_projected_constant(
                    delta,
                    projected_num_var_to_original,
                    is_auxiliary_num_var,
                    is_auxiliary_constant,
                    auxiliary_exprs,
                    projected_aux_initial_values,
                    numeric_variables,
                    projected_numeric_values,
                );
                assignment_effects.push(AssignmentEffect::new(
                    projected_target as u32,
                    planners_sas::numeric::numeric_task::AssignmentOperation::Plus,
                    projected_constant as u32,
                    false,
                    vec![],
                ));
                relevant = true;
            }
        }
    }

    if !relevant {
        Ok(None)
    } else {
        Ok(Some(Operator::new(
            operator.name().to_string(),
            preconditions,
            effects,
            assignment_effects,
            operator.cost(),
        )))
    }
}

fn get_or_add_projected_constant(
    value: f64,
    projected_num_var_to_original: &mut Vec<Option<usize>>,
    is_auxiliary_num_var: &mut Vec<bool>,
    is_auxiliary_constant: &mut Vec<bool>,
    auxiliary_exprs: &mut Vec<Option<ArithmeticExpr>>,
    projected_aux_initial_values: &mut Vec<Option<f64>>,
    numeric_variables: &mut Vec<NumericVariable>,
    projected_numeric_values: &mut Vec<f64>,
) -> usize {
    for (projected_id, numeric_variable) in numeric_variables.iter().enumerate() {
        if numeric_variable.get_type() == &NumericType::Constant
            && projected_numeric_values.get(projected_id).copied() == Some(value)
        {
            return projected_id;
        }
    }

    let projected_id = numeric_variables.len();
    numeric_variables.push(NumericVariable::new(
        format!("projected-const-{projected_id}"),
        NumericType::Constant,
        -1,
    ));
    projected_numeric_values.push(value);
    projected_num_var_to_original.push(None);
    is_auxiliary_num_var.push(false);
    is_auxiliary_constant.push(true);
    auxiliary_exprs.push(None);
    projected_aux_initial_values.push(Some(value));
    projected_id
}

fn project_propositional_axiom(
    axiom: &PropositionalAxiom,
    var_map: &[Option<usize>],
) -> Option<PropositionalAxiom> {
    let mapped_var = var_map
        .get(axiom.var_id() as usize)
        .and_then(|mapped| *mapped)?;
    let conditions: Vec<Fact> = axiom
        .conditions()
        .iter()
        .filter_map(|fact| project_fact(fact, var_map))
        .collect();
    Some(PropositionalAxiom::new(
        conditions,
        mapped_var as u32,
        axiom.precondition_value(),
        axiom.effect_value(),
    ))
}

fn project_comparison_axiom(
    axiom: &ComparisonAxiom,
    var_map: &[Option<usize>],
    num_var_map: &[Option<usize>],
) -> Option<ComparisonAxiom> {
    let affected = var_map
        .get(usize::try_from(axiom.get_affected_var_id()).ok()?)
        .and_then(|mapped| *mapped)?;
    let left = num_var_map
        .get(usize::try_from(axiom.get_left_var_id()).ok()?)
        .and_then(|mapped| *mapped)?;
    let right = num_var_map
        .get(usize::try_from(axiom.get_right_var_id()).ok()?)
        .and_then(|mapped| *mapped)?;
    Some(ComparisonAxiom::new(
        affected as i32,
        left as i32,
        right as i32,
        axiom.get_operator().clone(),
    ))
}

fn project_assignment_axiom(
    axiom: &AssignmentAxiom,
    num_var_map: &[Option<usize>],
    is_auxiliary_num_var: &[bool],
) -> Option<AssignmentAxiom> {
    let affected = num_var_map
        .get(axiom.get_affected_var_id() as usize)
        .and_then(|mapped| *mapped)?;
    if is_auxiliary_num_var[affected] {
        return None;
    }
    let left = num_var_map
        .get(axiom.get_left_var_id() as usize)
        .and_then(|mapped| *mapped)?;
    let right = num_var_map
        .get(axiom.get_right_var_id() as usize)
        .and_then(|mapped| *mapped)?;
    Some(AssignmentAxiom::new(
        affected as u32,
        axiom.get_operator().clone(),
        left as u32,
        right as u32,
    ))
}

#[cfg(test)]
mod tests {
    use planners_sas::numeric::axioms::{ComparisonOperator, PropositionalAxiom};
    use planners_sas::numeric::numeric_task::{AssignmentOperation, NumericRootTask};

    use super::*;

    fn simple_var(name: &str, axiom_layer: i32) -> ExplicitVariable {
        ExplicitVariable::new(
            2,
            name.to_string(),
            vec![format!("{name}=0"), format!("{name}=1")],
            axiom_layer,
            1,
        )
    }

    fn sample_task() -> NumericRootTask {
        let variables = vec![
            simple_var("p", -1),
            ExplicitVariable::new(
                3,
                "cmp".to_string(),
                vec![
                    "cmp-true".to_string(),
                    "cmp-false".to_string(),
                    "cmp-unk".to_string(),
                ],
                0,
                2,
            ),
            simple_var("goal_marker", 1),
        ];
        let numeric_variables = vec![
            NumericVariable::new("const10".to_string(), NumericType::Constant, -1),
            NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            NumericVariable::new("sum".to_string(), NumericType::Derived, 0),
        ];
        let operators = vec![Operator::new(
            "inc-x".to_string(),
            vec![Fact::new(0, 0)],
            vec![],
            vec![AssignmentEffect::new(
                1,
                AssignmentOperation::Plus,
                0,
                false,
                vec![],
            )],
            1,
        )];
        let axioms = vec![PropositionalAxiom::new(vec![Fact::new(1, 0)], 2, 1, 0)];
        let comparison_axioms = vec![ComparisonAxiom::new(
            1,
            2,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )];
        let assignment_axioms = vec![AssignmentAxiom::new(2, CalOperator::Sum, 1, 0)];

        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            variables,
            numeric_variables,
            vec![Fact::new(2, 0)],
            vec![],
            vec![0, 2, 1],
            vec![10.0, 0.0, 10.0],
            operators,
            axioms,
            comparison_axioms,
            assignment_axioms,
            (0, 0),
        )
    }

    #[test]
    fn projected_task_closes_over_relevant_numeric_and_goal_axiom_vars() {
        let task = sample_task();
        let pattern = Pattern {
            regular: vec![0],
            numeric: vec![1],
        };

        let projected = ProjectedTask::new(&task, &pattern).unwrap();

        assert_eq!(projected.get_num_variables(), 3);
        assert_eq!(projected.numeric_variables().len(), 3);
        assert_eq!(projected.get_num_operators(), 1);
        assert_eq!(projected.get_num_cmp_axioms(), 1);
        assert_eq!(projected.get_num_axioms(), 1);
        assert_eq!(projected.get_num_goals(), 1);

        let init_num = projected.get_initial_numeric_state_values();
        assert_eq!(init_num.as_slice(), &[0.0, 10.0, 10.0]);
    }

    #[test]
    fn projected_task_rejects_subtraction_based_numeric_conditions() {
        let variables = vec![simple_var("p", -1), simple_var("cmp", 0)];
        let numeric_variables = vec![
            NumericVariable::new("const1".to_string(), NumericType::Constant, -1),
            NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            NumericVariable::new("diff".to_string(), NumericType::Derived, 0),
        ];
        let task = NumericRootTask::new(
            1,
            Metric::new(true, -1),
            variables,
            numeric_variables,
            vec![],
            vec![],
            vec![0, 1],
            vec![1.0, 2.0, 1.0],
            vec![],
            vec![],
            vec![ComparisonAxiom::new(
                1,
                2,
                0,
                ComparisonOperator::GreaterThanOrEqual,
            )],
            vec![AssignmentAxiom::new(2, CalOperator::Difference, 1, 0)],
            (0, 0),
        );

        let result = ProjectedTask::new(
            &task,
            &Pattern {
                regular: vec![0],
                numeric: vec![1],
            },
        );

        assert!(matches!(
            result,
            Err(ProjectedTaskBuildError::UnsupportedAssignmentOperator { .. })
        ));
    }

    #[test]
    fn projected_task_rejects_raw_derived_numeric_pattern_vars() {
        let task = sample_task();

        let result = ProjectedTask::new(
            &task,
            &Pattern {
                regular: vec![0],
                numeric: vec![2],
            },
        );

        assert!(matches!(
            result,
            Err(ProjectedTaskBuildError::UnsupportedPatternNumericVarType {
                numeric_var_id: 2,
                numeric_type: NumericType::Derived,
            })
        ));
    }

    #[test]
    fn projected_task_helper_pattern_var_gets_lifted_additive_effect() {
        let variables = vec![simple_var("goal", -1)];
        let numeric_variables = vec![
            NumericVariable::new("const5".to_string(), NumericType::Constant, -1),
            NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            NumericVariable::new("y".to_string(), NumericType::Regular, -1),
            NumericVariable::new("sum".to_string(), NumericType::Derived, 0),
        ];
        let operators = vec![Operator::new(
            "inc-x".to_string(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                1,
                AssignmentOperation::Plus,
                0,
                false,
                vec![],
            )],
            1,
        )];
        let task = NumericRootTask::new(
            1,
            Metric::new(true, -1),
            variables,
            numeric_variables,
            vec![],
            vec![],
            vec![0],
            vec![5.0, 2.0, 3.0, 5.0],
            operators,
            vec![],
            vec![],
            vec![AssignmentAxiom::new(3, CalOperator::Sum, 1, 2)],
            (0, 0),
        );

        let helper_pattern_numeric_id = task.numeric_variables().len();
        let projected = ProjectedTask::new(
            &task,
            &Pattern {
                regular: vec![],
                numeric: vec![helper_pattern_numeric_id],
            },
        )
        .unwrap();

        assert_eq!(projected.numeric_variables().len(), 2);
        assert_eq!(
            projected.numeric_variables()[0].get_type(),
            &NumericType::Regular
        );

        let initial_numeric_values = projected.get_initial_numeric_state_values();
        assert_eq!(initial_numeric_values.as_slice(), &[5.0, 5.0]);
        drop(initial_numeric_values);

        assert_eq!(projected.get_num_operators(), 1);
        let op = &projected.get_operators()[0];
        assert_eq!(op.assignment_effects().len(), 1);
        assert_eq!(op.assignment_effects()[0].affected_var_id(), 0);
        assert_eq!(
            op.assignment_effects()[0].operation(),
            &AssignmentOperation::Plus
        );
        assert_eq!(op.assignment_effects()[0].var_id(), 1);
    }
}
