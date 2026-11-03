//! Building the arena DAG of a task's numeric conditions from its comparison
//! and assignment axioms.

use super::*;

/// Maps each numeric variable to the assignment axiom defining it, if any.
pub fn assignment_axiom_lookup(
    num_numeric_vars: usize,
    assignment_axioms: &[AssignmentAxiom],
) -> Result<Vec<Option<AxiomIndex>>, NumericConditionError> {
    let mut by_target = vec![None; num_numeric_vars];
    for (assignment_axiom_id, axiom) in assignment_axioms.iter().enumerate() {
        let assignment_axiom_id = AxiomIndex::new(assignment_axiom_id as u32);
        let target = axiom.affected_var_id();
        let slot = by_target.get_mut(target.index()).ok_or(
            NumericConditionError::InvalidAssignmentTarget {
                assignment_axiom_id,
                provided: target,
                num_numeric_vars,
            },
        )?;
        if let Some(first_assignment_axiom_id) = slot.replace(assignment_axiom_id) {
            return Err(NumericConditionError::DuplicateAssignmentTarget {
                numeric_var_id: target,
                first_assignment_axiom_id,
                second_assignment_axiom_id: assignment_axiom_id,
            });
        }
    }
    Ok(by_target)
}

/// Expands numeric variables into a node arena, memoising per variable so
/// shared sub-expressions become shared nodes, and rejecting cyclic
/// definitions.
pub(super) struct DagBuilder {
    nodes: Vec<ConditionNode>,
    memo: Vec<Option<NodeId>>,
    on_stack: Vec<bool>,
    memoised: Vec<VariableIndex>,
}

impl DagBuilder {
    pub(super) fn new(num_numeric_vars: usize) -> Self {
        Self {
            nodes: Vec::new(),
            memo: vec![None; num_numeric_vars],
            on_stack: vec![false; num_numeric_vars],
            memoised: Vec::new(),
        }
    }

    /// Start a fresh arena, clearing only the memo slots actually used.
    pub(super) fn restart(&mut self) {
        self.nodes.clear();
        for numeric_var_id in self.memoised.drain(..) {
            self.memo[numeric_var_id.index()] = None;
        }
    }

    pub(super) fn take_nodes(&mut self) -> Vec<ConditionNode> {
        std::mem::take(&mut self.nodes)
    }

    pub(super) fn expand(
        &mut self,
        numeric_var_id: VariableIndex,
        definitions: &[Option<AxiomIndex>],
        assignment_axioms: &[AssignmentAxiom],
    ) -> Result<NodeId, NumericConditionError> {
        let definition = *definitions.get(numeric_var_id.index()).ok_or(
            NumericConditionError::UnknownNumericVar {
                provided: numeric_var_id,
                num_numeric_vars: definitions.len(),
            },
        )?;

        if let Some(node_id) = self.memo[numeric_var_id.index()] {
            return Ok(node_id);
        }
        if self.on_stack[numeric_var_id.index()] {
            return Err(NumericConditionError::CycleDetected { numeric_var_id });
        }
        self.on_stack[numeric_var_id.index()] = true;

        let node = match definition {
            Some(assignment_axiom_id) => {
                let axiom = &assignment_axioms[assignment_axiom_id.index()];
                match &axiom.expression {
                    NumericExpression::Linear(terms) => {
                        for &(variable, _) in terms {
                            self.expand(variable, definitions, assignment_axioms)?;
                        }
                        ConditionNode::Linear {
                            result_numeric_var_id: numeric_var_id,
                            terms: terms.clone(),
                        }
                    }
                    NumericExpression::Binary {
                        operator,
                        left: left_numeric_var_id,
                        right: right_numeric_var_id,
                    } => {
                        let left =
                            self.expand(*left_numeric_var_id, definitions, assignment_axioms)?;
                        let right =
                            self.expand(*right_numeric_var_id, definitions, assignment_axioms)?;
                        ConditionNode::Arith {
                            result_numeric_var_id: numeric_var_id,
                            assignment_axiom_id,
                            op: ArithOp::from(operator),
                            left_numeric_var_id: *left_numeric_var_id,
                            right_numeric_var_id: *right_numeric_var_id,
                            left,
                            right,
                        }
                    }
                }
            }
            None => ConditionNode::Leaf { numeric_var_id },
        };

        // Children are pushed by the recursive calls above, so a node's
        // `NodeId` always exceeds its children's.
        let node_id = self.nodes.len();
        self.nodes.push(node);

        self.on_stack[numeric_var_id.index()] = false;
        self.memo[numeric_var_id.index()] = Some(node_id);
        self.memoised.push(numeric_var_id);
        Ok(node_id)
    }
}

pub(super) fn regular_leaf_dependencies(
    nodes: &[ConditionNode],
    numeric_variables: &[NumericVariable],
) -> Vec<VariableIndex> {
    let mut dependencies: Vec<VariableIndex> = nodes
        .iter()
        .filter_map(|node| match node {
            ConditionNode::Leaf { numeric_var_id } => numeric_variables
                .get(numeric_var_id.index())
                .filter(|variable| variable.numeric_type() == NumericType::Regular)
                .map(|_| *numeric_var_id),
            ConditionNode::Arith { .. } | ConditionNode::Linear { .. } => None,
        })
        .collect();
    dependencies.sort_unstable();
    dependencies.dedup();
    dependencies
}

/// Every numeric variable the arena touches is either a leaf's source or an
/// `Arith` node's result, so the largest of those ids bounds the table.
pub(super) fn required_numeric_len(nodes: &[ConditionNode]) -> usize {
    nodes
        .iter()
        .map(ConditionNode::result_numeric_var_id)
        .max()
        .map_or(0, |max_id| max_id.index() + 1)
}
