use super::*;

/// Equal for the same split with the same children; numeric boundaries
/// compare by their canonical bits.
#[derive(Debug, Clone)]
pub(crate) enum RefinementNode {
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

impl PartialEq for RefinementNode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Leaf { state_id },
                Self::Leaf {
                    state_id: other_state_id,
                },
            ) => state_id == other_state_id,
            (
                Self::Propositional {
                    var_id,
                    wanted,
                    wanted_child,
                    other_child,
                },
                Self::Propositional {
                    var_id: other_var_id,
                    wanted: other_wanted,
                    wanted_child: other_wanted_child,
                    other_child: other_other_child,
                },
            ) => {
                var_id == other_var_id
                    && wanted == other_wanted
                    && wanted_child == other_wanted_child
                    && other_child == other_other_child
            }
            (
                Self::Numeric {
                    var_id,
                    boundary,
                    lower_includes_boundary,
                    lower_child,
                    upper_child,
                },
                Self::Numeric {
                    var_id: other_var_id,
                    boundary: other_boundary,
                    lower_includes_boundary: other_lower_includes_boundary,
                    lower_child: other_lower_child,
                    upper_child: other_upper_child,
                },
            ) => {
                var_id == other_var_id
                    && float_tolerance::canonical_bits(*boundary)
                        == float_tolerance::canonical_bits(*other_boundary)
                    && lower_includes_boundary == other_lower_includes_boundary
                    && lower_child == other_lower_child
                    && upper_child == other_upper_child
            }
            _ => false,
        }
    }
}

impl Eq for RefinementNode {}

impl std::hash::Hash for RefinementNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Leaf { state_id } => state_id.hash(state),
            Self::Propositional {
                var_id,
                wanted,
                wanted_child,
                other_child,
            } => (var_id, wanted, wanted_child, other_child).hash(state),
            Self::Numeric {
                var_id,
                boundary,
                lower_includes_boundary,
                lower_child,
                upper_child,
            } => (
                var_id,
                float_tolerance::canonical_bits(*boundary),
                lower_includes_boundary,
                lower_child,
                upper_child,
            )
                .hash(state),
        }
    }
}

/// Equal for Cartesian abstractions that the same splits refined in the
/// same order.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CartesianRefinementHierarchy {
    pub(crate) nodes: Vec<RefinementNode>,
}

impl CartesianRefinementHierarchy {
    pub(crate) fn trivial() -> Self {
        Self {
            nodes: vec![RefinementNode::Leaf { state_id: 0 }],
        }
    }

    /// The abstract state of the concrete state with these values, reading
    /// numeric variables raw. The ICAPS26 abstraction maps states this way.
    pub fn map_state(
        &self,
        propositional: &[ExplicitValueIndex],
        numeric: &[NumericValue],
    ) -> Result<usize> {
        self.map_state_with(propositional, |var_id| numeric[var_id.index()])
    }

    /// The abstract state of the concrete state with propositional values
    /// `propositional` whose numeric coordinates have `coordinate_value`.
    pub(crate) fn map_state_with(
        &self,
        propositional: &[ExplicitValueIndex],
        coordinate_value: impl Fn(VariableIndex) -> NumericValue,
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
                    let value = coordinate_value(*var_id);
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

    /// Replaces leaf `leaf_node_id` of `old_state_id` by a node for `split`,
    /// whose wanted or lower child the old state keeps iff
    /// `old_state_is_wanted_or_lower`. Returns the leaves of the old and the
    /// new state.
    pub(crate) fn split(
        &mut self,
        leaf_node_id: usize,
        old_state_id: usize,
        new_state_id: usize,
        split: &Split,
        old_state_is_wanted_or_lower: bool,
    ) -> Result<(usize, usize)> {
        match split {
            Split::Propositional { var_id, wanted, .. } => self.split_propositional(
                leaf_node_id,
                old_state_id,
                new_state_id,
                *var_id,
                wanted.clone(),
                old_state_is_wanted_or_lower,
            )?,
            Split::Numeric {
                var_id,
                boundary,
                lower_includes_boundary,
                ..
            } => self.split_numeric(
                leaf_node_id,
                old_state_id,
                new_state_id,
                NumericSplit {
                    var_id: *var_id,
                    boundary: *boundary,
                    lower_includes_boundary: *lower_includes_boundary,
                    old_state_is_lower: old_state_is_wanted_or_lower,
                },
            )?,
        }
        // The first child node is the wanted or lower one.
        let first_child = self.nodes.len() - 2;
        Ok(if old_state_is_wanted_or_lower {
            (first_child, first_child + 1)
        } else {
            (first_child + 1, first_child)
        })
    }

    pub(crate) fn split_propositional(
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

    pub(crate) fn split_numeric(
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
pub(crate) struct NumericSplit {
    pub(crate) var_id: VariableIndex,
    pub(crate) boundary: NumericValue,
    /// Whether `boundary` itself belongs to the lower child.
    pub(crate) lower_includes_boundary: bool,
    /// Whether the state that existed before the split becomes the lower child.
    pub(crate) old_state_is_lower: bool,
}
