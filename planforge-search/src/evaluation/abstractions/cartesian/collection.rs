//! Collections of Cartesian abstractions: the generic collection generator
//! with Cartesian members, and the key standard collections drop duplicate
//! members by.

use super::*;
use crate::evaluation::abstractions::collections::generator::CollectionGenerator;

pub type CartesianAbstractionCollectionGenerator =
    CollectionGenerator<CartesianAbstractionGenerator>;

/// A refinement hierarchy as a value: equal for Cartesian abstractions that
/// the same splits refined in the same order.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum HierarchyKey {
    Leaf(usize),
    Propositional {
        var_id: VariableIndex,
        wanted: Vec<ExplicitValueIndex>,
        children: (usize, usize),
    },
    Numeric {
        var_id: VariableIndex,
        boundary_bits: u64,
        lower_includes_boundary: bool,
        children: (usize, usize),
    },
}

/// The key of `hierarchy`: equal for Cartesian abstractions that the same
/// splits refined in the same order.
pub(super) fn hierarchy_key(hierarchy: &CartesianRefinementHierarchy) -> Vec<HierarchyKey> {
    hierarchy
        .nodes
        .iter()
        .map(|node| match node {
            RefinementNode::Leaf { state_id } => HierarchyKey::Leaf(*state_id),
            RefinementNode::Propositional {
                var_id,
                wanted,
                wanted_child,
                other_child,
            } => HierarchyKey::Propositional {
                var_id: *var_id,
                wanted: wanted.clone(),
                children: (*wanted_child, *other_child),
            },
            RefinementNode::Numeric {
                var_id,
                boundary,
                lower_includes_boundary,
                lower_child,
                upper_child,
            } => HierarchyKey::Numeric {
                var_id: *var_id,
                boundary_bits: float_tolerance::canonical_bits(*boundary),
                lower_includes_boundary: *lower_includes_boundary,
                children: (*lower_child, *upper_child),
            },
        })
        .collect()
}
