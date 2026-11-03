//! Numeric conditions: the comparisons a task can test on its numeric state.
//!
//! A numeric condition is the semantic content of one [`ComparisonAxiom`]:
//! linear part and constant threshold for a linear comparison, or two
//! arithmetic operands for a nonlinear one. Definitions are expanded into an
//! arena DAG. A shared linear sum stays one node and canonicalizes once;
//! nonlinear arithmetic retains its binary operations.
//!
//! Conditions are a first-class part of a task: [`NumericConditions`] is
//! built once at task construction and reached through
//! [`crate::numeric_task::Task::numeric_conditions`]. Nothing
//! else in the codebase rebuilds the "propositional variable -> comparison
//! axiom" mapping.
//!
//! Evaluation is generic over the value domain via [`ConditionDomain`], so
//! concrete states ([`NumericValue`]) and abstract states ([`Interval`]) share one
//! implementation.

#[cfg(test)]
mod tests;

mod dag;
mod expressions;

pub use dag::assignment_axiom_lookup;
use dag::{DagBuilder, regular_leaf_dependencies, required_numeric_len};
pub use expressions::{ArithOp, CompOp, ConditionDomain, ConditionNode, ConditionValue};

use crate::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, NumericExpression,
};
use crate::numeric_task::value_types::{ExplicitFact, FactNamespace, NumericType, NumericVariable};
use crate::numeric_task::{AxiomIndex, NumericValue, VariableIndex};
use crate::utils::interval::{EMPTY_INTERVAL, Interval};

/// Index of a [`NumericCondition`] inside [`NumericConditions`].
///
/// Identical to the id of the [`ComparisonAxiom`] it was derived from:
/// conditions are stored in comparison-axiom order, one per axiom.
pub type NumericConditionId = usize;

/// Index into a [`NumericCondition`]'s node arena.
pub type NodeId = usize;

/// One comparison over the numeric state, with both sides expanded into an
/// expression DAG over the task's numeric variables.
///
/// Invariant, established by [`NumericConditions::build`]: a node's children
/// have strictly smaller [`NodeId`]s than the node itself, so the arena is
/// already in bottom-up order.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericCondition {
    id: NumericConditionId,
    prop_var_id: VariableIndex,
    op: CompOp,
    left_numeric_var_id: VariableIndex,
    right_numeric_var_id: VariableIndex,
    nodes: Vec<ConditionNode>,
    left_root: NodeId,
    right_root: NodeId,
    regular_numeric_var_dependencies: Vec<VariableIndex>,
    required_numeric_len: usize,
}

impl NumericCondition {
    /// Id of this condition, equal to the id of the comparison axiom it was
    /// derived from.
    #[inline]
    pub fn id(&self) -> NumericConditionId {
        self.id
    }

    /// Propositional variable holding this condition's truth value, encoded as
    /// a [`ConditionValue`].
    #[inline]
    pub fn prop_var_id(&self) -> VariableIndex {
        self.prop_var_id
    }

    #[inline]
    pub fn op(&self) -> CompOp {
        self.op
    }

    /// Numeric variable the comparison axiom names on its left/right side,
    /// before expansion into the DAG.
    #[inline]
    pub fn left_numeric_var_id(&self) -> VariableIndex {
        self.left_numeric_var_id
    }

    #[inline]
    pub fn right_numeric_var_id(&self) -> VariableIndex {
        self.right_numeric_var_id
    }

    #[inline]
    pub fn nodes(&self) -> &[ConditionNode] {
        &self.nodes
    }

    #[inline]
    pub fn left_root(&self) -> NodeId {
        self.left_root
    }

    #[inline]
    pub fn right_root(&self) -> NodeId {
        self.right_root
    }

    #[inline]
    pub fn node(&self, node_id: NodeId) -> &ConditionNode {
        &self.nodes[node_id]
    }

    /// Sorted ids of the `Regular` numeric variables this condition reads.
    /// Derived variables are excluded: they are recomputed from these.
    #[inline]
    pub fn regular_numeric_var_dependencies(&self) -> &[VariableIndex] {
        &self.regular_numeric_var_dependencies
    }

    /// Smallest numeric-variable table length this condition can be evaluated
    /// against.
    #[inline]
    pub fn required_numeric_len(&self) -> usize {
        self.required_numeric_len
    }

    /// Bottom-up evaluation: a post-order walk that computes both operand
    /// DAGs children-first, then compares them. This is the default order.
    #[inline]
    pub fn evaluate<T: ConditionDomain>(&self, inputs: &[T]) -> T::Verdict {
        let lhs = self.evaluate_node(self.left_root, inputs);
        let rhs = self.evaluate_node(self.right_root, inputs);
        T::compare(self.op, lhs, rhs)
    }

    /// Bottom-up evaluation on a concrete numeric state.
    #[inline]
    pub fn evaluate_point(&self, values: &[NumericValue]) -> bool {
        self.evaluate(values)
    }

    /// Bottom-up evaluation on an abstract numeric state: `None` when the
    /// intervals admit both outcomes.
    #[inline]
    pub fn evaluate_interval(&self, intervals: &[Interval]) -> Option<bool> {
        self.evaluate(intervals)
    }

    fn evaluate_node<T: ConditionDomain>(&self, node_id: NodeId, inputs: &[T]) -> T {
        match &self.nodes[node_id] {
            ConditionNode::Leaf { numeric_var_id } => inputs[numeric_var_id.index()],
            ConditionNode::Linear {
                result_numeric_var_id,
                terms,
            } => T::linear(terms, *result_numeric_var_id, inputs),
            ConditionNode::Arith {
                op,
                left,
                right,
                result_numeric_var_id,
                ..
            } => {
                let lhs = self.evaluate_node(*left, inputs);
                let rhs = self.evaluate_node(*right, inputs);
                T::combine(*op, lhs, rhs, *result_numeric_var_id, inputs)
            }
        }
    }

    /// Like [`Self::evaluate_interval`], but also writes each derived
    /// variable's refined interval back into `intervals`, so the caller
    /// learns the bounds of the intermediate expressions.
    pub fn evaluate_interval_and_fill(&self, intervals: &mut [Interval]) -> Option<bool> {
        let lhs = self.evaluate_node_and_fill(self.left_root, intervals);
        let rhs = self.evaluate_node_and_fill(self.right_root, intervals);
        self.op.apply_interval(lhs, rhs)
    }

    fn evaluate_node_and_fill(&self, node_id: NodeId, intervals: &mut [Interval]) -> Interval {
        match &self.nodes[node_id] {
            ConditionNode::Leaf { numeric_var_id } => intervals[numeric_var_id.index()],
            ConditionNode::Linear {
                result_numeric_var_id,
                terms,
            } => {
                let result = Interval::linear(terms, *result_numeric_var_id, intervals);
                intervals[result_numeric_var_id.index()] = result;
                result
            }
            ConditionNode::Arith {
                op,
                left,
                right,
                result_numeric_var_id,
                ..
            } => {
                let lhs = self.evaluate_node_and_fill(*left, intervals);
                let rhs = self.evaluate_node_and_fill(*right, intervals);
                let result = Interval::combine(*op, lhs, rhs, *result_numeric_var_id, intervals);
                intervals[result_numeric_var_id.index()] = result;
                result
            }
        }
    }

    /// Interval of `lhs - rhs`.
    ///
    /// `lhs op rhs` is equivalent to `f op 0` for `f = lhs - rhs`, and a
    /// linear operator shifts `f` by a constant, so abstract-operator-variant
    /// filtering can reduce a joint constraint on source and target values to
    /// a one-dimensional check on `f`.
    pub fn lhs_minus_rhs_interval(&self, intervals: &[Interval]) -> Interval {
        let lhs = self.evaluate_node(self.left_root, intervals);
        let rhs = self.evaluate_node(self.right_root, intervals);
        ArithOp::Sub.apply_interval(lhs, rhs)
    }

    /// `true` iff *some* concrete assignment inside `intervals` satisfies the
    /// comparison — the optimistic predicate abstract-operator construction
    /// needs, since concrete values are recomputed per state during heuristic
    /// evaluation.
    #[inline]
    pub fn admits_true(&self, intervals: &[Interval]) -> bool {
        self.evaluate_interval(intervals) != Some(false)
    }

    /// Companion of [`Self::admits_true`]: `true` iff some concrete
    /// assignment inside `intervals` falsifies the comparison.
    #[inline]
    pub fn admits_false(&self, intervals: &[Interval]) -> bool {
        self.evaluate_interval(intervals) != Some(true)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumericConditionError {
    UnknownPropositionalVar {
        comparison_axiom_id: AxiomIndex,
        provided: VariableIndex,
        num_propositional_vars: usize,
    },
    DuplicatePropositionalVar {
        prop_var_id: VariableIndex,
        first_comparison_axiom_id: AxiomIndex,
        second_comparison_axiom_id: AxiomIndex,
    },
    UnknownNumericVar {
        provided: VariableIndex,
        num_numeric_vars: usize,
    },
    InvalidAssignmentTarget {
        assignment_axiom_id: AxiomIndex,
        provided: VariableIndex,
        num_numeric_vars: usize,
    },
    DuplicateAssignmentTarget {
        numeric_var_id: VariableIndex,
        first_assignment_axiom_id: AxiomIndex,
        second_assignment_axiom_id: AxiomIndex,
    },
    CycleDetected {
        numeric_var_id: VariableIndex,
    },
}

impl std::fmt::Display for NumericConditionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownPropositionalVar {
                comparison_axiom_id,
                provided,
                num_propositional_vars,
            } => write!(
                f,
                "comparison axiom {comparison_axiom_id} writes propositional variable {provided}, \
                 but the task has only {num_propositional_vars}"
            ),
            Self::DuplicatePropositionalVar {
                prop_var_id,
                first_comparison_axiom_id,
                second_comparison_axiom_id,
            } => write!(
                f,
                "comparison axioms {first_comparison_axiom_id} and {second_comparison_axiom_id} \
                 both write propositional variable {prop_var_id}"
            ),
            Self::UnknownNumericVar {
                provided,
                num_numeric_vars,
            } => write!(
                f,
                "numeric variable {provided} referenced, but the task has only {num_numeric_vars}"
            ),
            Self::InvalidAssignmentTarget {
                assignment_axiom_id,
                provided,
                num_numeric_vars,
            } => write!(
                f,
                "assignment axiom {assignment_axiom_id} writes numeric variable {provided}, \
                 but the task has only {num_numeric_vars}"
            ),
            Self::DuplicateAssignmentTarget {
                numeric_var_id,
                first_assignment_axiom_id,
                second_assignment_axiom_id,
            } => write!(
                f,
                "assignment axioms {first_assignment_axiom_id} and {second_assignment_axiom_id} \
                 both write numeric variable {numeric_var_id}"
            ),
            Self::CycleDetected { numeric_var_id } => write!(
                f,
                "assignment axioms define numeric variable {numeric_var_id} in terms of itself"
            ),
        }
    }
}

impl std::error::Error for NumericConditionError {}

/// All numeric conditions of a task, indexed both by condition id and by the
/// propositional variable that carries a condition's truth value.
///
/// Built once per task; the single source of truth for "is this variable a
/// comparison result?".
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NumericConditions {
    conditions: Vec<NumericCondition>,
    by_prop_var: Vec<Option<NumericConditionId>>,
}

impl NumericConditions {
    pub fn build(
        num_propositional_vars: usize,
        numeric_variables: &[NumericVariable],
        comparison_axioms: &[ComparisonAxiom],
        assignment_axioms: &[AssignmentAxiom],
    ) -> Result<Self, NumericConditionError> {
        let num_numeric_vars = numeric_variables.len();
        let definitions = assignment_axiom_lookup(num_numeric_vars, assignment_axioms)?;

        let mut conditions = Vec::with_capacity(comparison_axioms.len());
        let mut by_prop_var = vec![None; num_propositional_vars];
        let mut builder = DagBuilder::new(num_numeric_vars);

        for (id, axiom) in comparison_axioms.iter().enumerate() {
            let prop_var_id = axiom.affected_var_id();
            let slot = by_prop_var.get_mut(prop_var_id.index()).ok_or(
                NumericConditionError::UnknownPropositionalVar {
                    comparison_axiom_id: AxiomIndex::new(id as u32),
                    provided: prop_var_id,
                    num_propositional_vars,
                },
            )?;
            if let Some(first_comparison_axiom_id) = slot.replace(id) {
                return Err(NumericConditionError::DuplicatePropositionalVar {
                    prop_var_id,
                    first_comparison_axiom_id: AxiomIndex::new(first_comparison_axiom_id as u32),
                    second_comparison_axiom_id: AxiomIndex::new(id as u32),
                });
            }

            let left_numeric_var_id = axiom.left_var_id();
            let right_numeric_var_id = axiom.right_var_id();
            builder.restart();
            let left_root = builder.expand(left_numeric_var_id, &definitions, assignment_axioms)?;
            let right_root =
                builder.expand(right_numeric_var_id, &definitions, assignment_axioms)?;
            let nodes = builder.take_nodes();

            conditions.push(NumericCondition {
                id,
                prop_var_id,
                op: CompOp::from(axiom.operator()),
                left_numeric_var_id,
                right_numeric_var_id,
                regular_numeric_var_dependencies: regular_leaf_dependencies(
                    &nodes,
                    numeric_variables,
                ),
                required_numeric_len: required_numeric_len(&nodes),
                nodes,
                left_root,
                right_root,
            });
        }

        Ok(Self {
            conditions,
            by_prop_var,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.conditions.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    #[inline]
    pub fn all(&self) -> &[NumericCondition] {
        &self.conditions
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, NumericCondition> {
        self.conditions.iter()
    }

    #[inline]
    pub fn get(&self, id: NumericConditionId) -> Option<&NumericCondition> {
        self.conditions.get(id)
    }

    /// Is `prop_var_id` the truth value of a numeric condition rather than an
    /// ordinary propositional variable?
    #[inline]
    pub fn is_condition_var(&self, prop_var_id: VariableIndex) -> bool {
        matches!(self.by_prop_var.get(prop_var_id.index()), Some(Some(_)))
    }

    /// Number of propositional variables the task has, condition-carrying ones
    /// included.
    #[inline]
    pub fn num_propositional_vars(&self) -> usize {
        self.by_prop_var.len()
    }

    /// The namespace variable `prop_var_id` belongs to.
    ///
    /// Only variables of *this* task have a namespace here; ids past the
    /// task's propositional variables belong to some other id space — a
    /// domain abstraction's [`FactNamespace::NumericVariable`] ids being the
    /// one that exists — and answering for them would be a guess.
    /// [`assert_fact_namespaces`](crate::numeric_task::assert_fact_namespaces)
    /// rejects them before they get here.
    #[inline]
    pub fn namespace_of(&self, prop_var_id: VariableIndex) -> FactNamespace {
        if self.is_condition_var(prop_var_id) {
            FactNamespace::Condition
        } else {
            FactNamespace::Propositional
        }
    }

    /// Build a correctly tagged fact. This is the authoritative constructor
    /// for callers that hold a variable id and cannot themselves know which
    /// namespace it names.
    #[inline]
    pub fn fact(&self, prop_var_id: usize, value: usize) -> ExplicitFact {
        ExplicitFact::in_namespace(
            self.namespace_of(VariableIndex::from_usize(prop_var_id)),
            prop_var_id,
            value,
        )
    }

    #[inline]
    pub fn id_for_var(&self, prop_var_id: VariableIndex) -> Option<NumericConditionId> {
        *self.by_prop_var.get(prop_var_id.index())?
    }

    #[inline]
    pub fn for_var(&self, prop_var_id: VariableIndex) -> Option<&NumericCondition> {
        self.conditions.get(self.id_for_var(prop_var_id)?)
    }

    /// Propositional variables carrying condition truth values, ascending.
    pub fn condition_var_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.by_prop_var
            .iter()
            .enumerate()
            .filter_map(|(prop_var_id, condition)| condition.map(|_| prop_var_id))
    }

    /// Is `precondition` unsatisfiable for every concrete numeric assignment
    /// inside `numeric_intervals`?
    ///
    /// Optimistic, matching the rest of abstract-operator construction: a
    /// `TRUE` precondition is contradicted only when no value in the
    /// intervals makes the comparison true, and symmetrically for `FALSE`.
    /// Facts on ordinary propositional variables are never contradicted here.
    pub fn precondition_is_contradicted(
        &self,
        precondition: &ExplicitFact,
        numeric_intervals: &[Interval],
    ) -> bool {
        let Some(condition) = self.for_var(precondition.var_index()) else {
            return false;
        };
        match ConditionValue::from_usize(precondition.value()) {
            Some(ConditionValue::True) => !condition.admits_true(numeric_intervals),
            Some(ConditionValue::False) => !condition.admits_false(numeric_intervals),
            // Abstracted values land outside the concrete domain: they name a
            // class of condition values rather than one, so no single numeric
            // assignment contradicts them.
            None => false,
        }
    }
}
