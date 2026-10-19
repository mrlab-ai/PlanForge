use super::*;

#[derive(Debug, Clone)]
pub(super) enum Split {
    Propositional {
        state_id: usize,
        var_id: VariableIndex,
        wanted: Vec<ExplicitValueIndex>,
        witness_value: ExplicitValueIndex,
        description: String,
    },
    Numeric {
        state_id: usize,
        var_id: VariableIndex,
        boundary: NumericValue,
        lower_includes_boundary: bool,
        witness_value: NumericValue,
        desired_contains_witness: bool,
        integer_lattice: bool,
        description: String,
    },
}

impl Split {
    pub(super) fn state_id(&self) -> usize {
        match self {
            Self::Propositional { state_id, .. } | Self::Numeric { state_id, .. } => *state_id,
        }
    }

    pub(super) fn description(&self) -> &str {
        match self {
            Self::Propositional { description, .. } | Self::Numeric { description, .. } => {
                description
            }
        }
    }

    pub(super) fn dimension(&self) -> SplitDimension {
        match self {
            Self::Propositional { var_id, .. } => SplitDimension::Propositional(*var_id),
            Self::Numeric { var_id, .. } => SplitDimension::Numeric(*var_id),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum SplitIdentity {
    Propositional {
        state_id: usize,
        var_id: VariableIndex,
        wanted: Vec<ExplicitValueIndex>,
        witness_value: ExplicitValueIndex,
    },
    Numeric {
        state_id: usize,
        var_id: VariableIndex,
        boundary_bits: u64,
        lower_includes_boundary: bool,
        witness_bits: u64,
        integer_lattice: bool,
    },
}

impl From<&Split> for SplitIdentity {
    fn from(split: &Split) -> Self {
        match split {
            Split::Propositional {
                state_id,
                var_id,
                wanted,
                witness_value,
                ..
            } => Self::Propositional {
                state_id: *state_id,
                var_id: *var_id,
                wanted: wanted.clone(),
                witness_value: *witness_value,
            },
            Split::Numeric {
                state_id,
                var_id,
                boundary,
                lower_includes_boundary,
                witness_value,
                integer_lattice,
                ..
            } => Self::Numeric {
                state_id: *state_id,
                var_id: *var_id,
                boundary_bits: boundary.to_bits(),
                lower_includes_boundary: *lower_includes_boundary,
                witness_bits: witness_value.to_bits(),
                integer_lattice: *integer_lattice,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SplitDimension {
    Propositional(VariableIndex),
    Numeric(VariableIndex),
}
