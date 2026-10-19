use super::*;

#[derive(Debug, Clone)]
pub(super) enum RefinementNode {
    Leaf {
        state_id: usize,
    },
    Propositional {
        var_id: VariableIndex,
        wanted: Vec<ExplicitValueIndex>,
        wanted_child: usize,
        other_child: usize,
    },
    Numeric {
        var_id: VariableIndex,
        boundary: NumericValue,
        lower_includes_boundary: bool,
        lower_child: usize,
        upper_child: usize,
    },
}

#[derive(Debug, Clone)]
pub struct CartesianRefinementHierarchy {
    pub(super) nodes: Vec<RefinementNode>,
}

impl CartesianRefinementHierarchy {
    pub(super) fn trivial() -> Self {
        Self {
            nodes: vec![RefinementNode::Leaf { state_id: 0 }],
        }
    }

    pub fn map_state(
        &self,
        propositional: &[ExplicitValueIndex],
        numeric: &[NumericValue],
    ) -> Result<usize> {
        let mut node_id = 0;
        loop {
            match self
                .nodes
                .get(node_id)
                .with_context(|| format!("missing Cartesian hierarchy node {node_id}"))?
            {
                RefinementNode::Leaf { state_id } => return Ok(*state_id),
                RefinementNode::Propositional {
                    var_id,
                    wanted,
                    wanted_child,
                    other_child,
                } => {
                    let value = *propositional.get(var_id.index()).with_context(|| {
                        format!("propositional state has no value for var {var_id}")
                    })? as ExplicitValueIndex;
                    node_id = if wanted.binary_search(&value).is_ok() {
                        *wanted_child
                    } else {
                        *other_child
                    };
                }
                RefinementNode::Numeric {
                    var_id,
                    boundary,
                    lower_includes_boundary,
                    lower_child,
                    upper_child,
                } => {
                    let value = *numeric
                        .get(var_id.index())
                        .with_context(|| format!("numeric state has no value for var {var_id}"))?;
                    ensure!(
                        value.is_finite(),
                        "numeric state var {var_id} is not finite: {value}"
                    );
                    let in_lower =
                        value < *boundary || (*lower_includes_boundary && value == *boundary);
                    node_id = if in_lower { *lower_child } else { *upper_child };
                }
            }
        }
    }

    pub(super) fn split_propositional(
        &mut self,
        leaf_node_id: usize,
        old_state_id: usize,
        new_state_id: usize,
        var_id: VariableIndex,
        mut wanted: Vec<ExplicitValueIndex>,
        old_state_is_wanted: bool,
    ) -> Result<()> {
        wanted.sort_unstable();
        wanted.dedup();
        let wanted_node_id = self.nodes.len();
        let other_node_id = wanted_node_id + 1;
        self.nodes.push(RefinementNode::Leaf {
            state_id: if old_state_is_wanted {
                old_state_id
            } else {
                new_state_id
            },
        });
        self.nodes.push(RefinementNode::Leaf {
            state_id: if old_state_is_wanted {
                new_state_id
            } else {
                old_state_id
            },
        });
        let node = self
            .nodes
            .get_mut(leaf_node_id)
            .with_context(|| format!("missing hierarchy leaf node {leaf_node_id}"))?;
        ensure!(
            matches!(node, RefinementNode::Leaf { state_id } if *state_id == old_state_id),
            "hierarchy node {leaf_node_id} is not leaf state {old_state_id}"
        );
        *node = RefinementNode::Propositional {
            var_id,
            wanted,
            wanted_child: wanted_node_id,
            other_child: other_node_id,
        };
        Ok(())
    }

    pub(super) fn split_numeric(
        &mut self,
        leaf_node_id: usize,
        old_state_id: usize,
        new_state_id: usize,
        split: NumericSplit,
    ) -> Result<()> {
        let NumericSplit {
            var_id,
            boundary,
            lower_includes_boundary,
            old_state_is_lower,
        } = split;
        ensure!(
            boundary.is_finite(),
            "Cartesian split boundary must be finite"
        );
        let lower_node_id = self.nodes.len();
        let upper_node_id = lower_node_id + 1;
        self.nodes.push(RefinementNode::Leaf {
            state_id: if old_state_is_lower {
                old_state_id
            } else {
                new_state_id
            },
        });
        self.nodes.push(RefinementNode::Leaf {
            state_id: if old_state_is_lower {
                new_state_id
            } else {
                old_state_id
            },
        });
        let node = self
            .nodes
            .get_mut(leaf_node_id)
            .with_context(|| format!("missing hierarchy leaf node {leaf_node_id}"))?;
        ensure!(
            matches!(node, RefinementNode::Leaf { state_id } if *state_id == old_state_id),
            "hierarchy node {leaf_node_id} is not leaf state {old_state_id}"
        );
        *node = RefinementNode::Numeric {
            var_id,
            boundary,
            lower_includes_boundary,
            lower_child: lower_node_id,
            upper_child: upper_node_id,
        };
        Ok(())
    }
}

/// Where a numeric refinement cuts one variable's range, and on which side of
/// the cut the state being split ends up.
#[derive(Debug, Clone, Copy)]
pub(super) struct NumericSplit {
    pub(super) var_id: VariableIndex,
    pub(super) boundary: NumericValue,
    /// Whether `boundary` itself belongs to the lower child.
    pub(super) lower_includes_boundary: bool,
    /// Whether the state that existed before the split becomes the lower child.
    pub(super) old_state_is_lower: bool,
}
