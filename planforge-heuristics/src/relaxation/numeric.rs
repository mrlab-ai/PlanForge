//! The monotonic numeric relaxation of Metric-FF: each numeric variable
//! reaches an interval, which effects can only widen.

use super::*;

/// Monotonic-relaxation envelope for one numeric variable.
#[derive(Debug, Clone, Copy)]
pub(super) struct NumericRange {
    pub(super) max: NumericValue,
    pub(super) min: NumericValue,
}

impl NumericRange {
    pub(super) const fn singleton(v: NumericValue) -> Self {
        Self { max: v, min: v }
    }

    pub(super) fn affine(terms: &[(VariableIndex, NumericValue)], values: &[Self]) -> Self {
        let terms = terms
            .iter()
            .filter(|(_, coefficient)| *coefficient != ZERO_VALUE);
        let endpoint = |upper| {
            NumericValue::affine(
                terms.clone().map(|(variable, coefficient)| {
                    let range = values[variable.index()];
                    let value = if (*coefficient > ZERO_VALUE) == upper {
                        range.max
                    } else {
                        range.min
                    };
                    (coefficient.value(), value)
                }),
                0.0,
            )
        };
        Self {
            min: endpoint(false),
            max: endpoint(true),
        }
    }

    /// Returns `true` if `other` widens this range.
    pub(super) fn join(&mut self, other: NumericRange) -> bool {
        let new_max = if other.max > self.max {
            other.max
        } else {
            self.max
        };
        let new_min = if other.min < self.min {
            other.min
        } else {
            self.min
        };
        let changed = other.max > self.max || other.min < self.min;
        self.max = new_max;
        self.min = new_min;
        changed
    }
}

/// Monotonic direction in which an assignment effect can push the affected
/// variable's envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EffectDirection {
    /// Effect can grow `max[affected]` (and may also shrink `min[affected]`).
    GrowMax,
    /// Effect can shrink `min[affected]` (only).
    ShrinkMin,
    /// Effect could move either bound — typically because the RHS is a live
    /// numeric variable whose sign is not statically determined.
    Both,
}

impl EffectDirection {
    pub(super) fn includes_grow_max(self) -> bool {
        matches!(self, EffectDirection::GrowMax | EffectDirection::Both)
    }
    pub(super) fn includes_shrink_min(self) -> bool {
        matches!(self, EffectDirection::ShrinkMin | EffectDirection::Both)
    }
}

#[derive(Debug, Clone)]
pub(super) struct AssignmentEffectDesc {
    pub(super) affected_var: NumVarId,
    pub(super) operation: AssignmentOperation,
    pub(super) rhs_var: NumVarId,
    pub(super) direction: EffectDirection,
}

#[derive(Debug, Clone)]
pub(super) struct ComparisonAxiomDesc {
    /// FactId of the TRUE-value fact for the propositional variable backing
    /// this axiom.
    pub(super) true_fact: FactId,
    pub(super) left_var: NumVarId,
    pub(super) right_var: NumVarId,
    pub(super) op: ComparisonOperator,
}

/// Transitive numeric-var dependency closure. `depends_on[v]` lists every
/// numeric var `u` such that `v`'s value is computed (directly or
/// recursively) from `u` via `Sum`/`Difference` assignment axioms.
///
/// `v` always depends on itself. The closure converges in at most
/// `num_numeric` iterations because each iteration that adds *anything*
/// strictly grows at least one `depends_on[v]` set, and each set is
/// bounded by `num_numeric`.
pub(super) fn compute_numeric_dependency_closure(
    num_numeric: usize,
    assignment_axioms: &[AssignmentAxiom],
) -> Vec<Vec<NumVarId>> {
    let mut sets: Vec<HashSet<NumVarId>> = (0..num_numeric)
        .map(|v| {
            let mut s = HashSet::new();
            s.insert(VariableIndex::from_usize(v));
            s
        })
        .collect();
    loop {
        let mut changed = false;
        for ax in assignment_axioms {
            for operand in ax.expression.dependencies() {
                let sources = sets[operand.index()].clone();
                for variable in sources {
                    if sets[ax.affected_var_id.index()].insert(variable) {
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    sets.into_iter().map(|s| s.into_iter().collect()).collect()
}

/// Does the direction in which `affected_var`'s envelope can move
/// (under an effect with `direction`) advance the satisfaction of
/// `axiom`?
pub(super) fn axiom_needs_direction(
    affected_var: NumVarId,
    direction: EffectDirection,
    axiom: &ComparisonAxiomDesc,
) -> bool {
    let lhs = axiom.left_var == affected_var;
    let rhs = axiom.right_var == affected_var;
    if !lhs && !rhs {
        return false;
    }
    match axiom.op {
        ComparisonOperator::GreaterThan | ComparisonOperator::GreaterThanOrEqual => {
            // need max[L] big or min[R] small
            (lhs && direction.includes_grow_max()) || (rhs && direction.includes_shrink_min())
        }
        ComparisonOperator::LessThan | ComparisonOperator::LessThanOrEqual => {
            // need min[L] small or max[R] big
            (lhs && direction.includes_shrink_min()) || (rhs && direction.includes_grow_max())
        }
        ComparisonOperator::Equal => {
            // any envelope movement on either side can help meet equality
            true
        }
        ComparisonOperator::UnEqual => {
            // any movement breaks equality
            true
        }
    }
}

#[cfg(test)]
mod affine_range_tests {
    use super::*;

    #[test]
    fn shared_linear_ranges_round_each_endpoint_once() {
        let value = NumericRange::singleton(NumericValue::new(2e-12));
        let terms = [
            (VariableIndex::new(0), NumericValue::new(0.25)),
            (VariableIndex::new(1), NumericValue::new(0.25)),
            (VariableIndex::new(2), ZERO_VALUE),
        ];
        let unbounded = NumericRange {
            min: NEG_INF_VALUE,
            max: INF_VALUE,
        };
        let result = NumericRange::affine(&terms, &[value, value, unbounded]);
        assert_eq!(result.min, NumericValue::new(1e-12));
        assert_eq!(result.max, NumericValue::new(1e-12));
    }
}
