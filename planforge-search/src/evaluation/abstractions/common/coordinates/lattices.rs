//! Lattice inference must prove closure under floating execution. A nominal
//! decimal gcd alone is insufficient: canonical +0.1 can move by 0.100000000001.

use super::*;
use planforge_sas::utils::float_tolerance::{ABS_EPSILON, GRID_BOUND, GRID_SCALE};
use planforge_sas::utils::value_lattice::ValueLattice;
use std::collections::HashSet;

const WHOLE_UNITS: i64 = 1_000_000_000_000;

struct NumericInvariant {
    bounds: Vec<Interval>,
    integral: Vec<bool>,
    nonnegative: Vec<bool>,
    nonpositive: Vec<bool>,
    increments: Vec<f64>,
    changed: Vec<bool>,
    resources: Vec<bool>,
    resource_bound: Option<NumericValue>,
}

impl Coordinates<'_> {
    pub(crate) fn value_lattices(&self) -> Vec<ValueLattice> {
        let initial = self.task.get_initial_numeric_state_values();
        let mut generators = vec![Vec::new(); initial.len()];
        let mut invariant = NumericInvariant {
            bounds: initial.iter().copied().map(Interval::singleton).collect(),
            integral: initial
                .iter()
                .map(|value| {
                    value
                        .grid_units()
                        .is_some_and(|units| units % WHOLE_UNITS == 0)
                })
                .collect(),
            nonnegative: initial.iter().map(|value| *value >= ZERO_VALUE).collect(),
            nonpositive: initial.iter().map(|value| *value <= ZERO_VALUE).collect(),
            increments: vec![0.0; initial.len()],
            changed: vec![false; initial.len()],
            resources: initial
                .iter()
                .enumerate()
                .map(|(variable, value)| {
                    self.task.numeric_variables()[variable].get_type() == &NumericType::Regular
                        && *value >= ZERO_VALUE
                        && value
                            .grid_units()
                            .is_some_and(|units| units % WHOLE_UNITS == 0)
                })
                .collect(),
            resource_bound: None,
        };
        for operator in self.task.get_operators() {
            let mut targets = HashSet::new();
            for effect in operator.assignment_effects() {
                let target = effect.affected_var_id().index();
                let source = effect.var_id().index();
                let constant = (self.task.numeric_variables()[source].get_type()
                    == &NumericType::Constant)
                    .then_some(initial[source]);
                let delta = constant_effect_delta(self.task, effect);
                invariant.changed[target] = true;
                if let Some(delta) = delta {
                    invariant.increments[target] =
                        invariant.increments[target].max(delta.abs().value());
                }
                let generator = match effect.operation() {
                    AssignmentOperation::Plus | AssignmentOperation::Minus => {
                        delta.and_then(NumericValue::grid_units)
                    }
                    AssignmentOperation::Assign => constant
                        .and_then(NumericValue::grid_units)
                        .zip(initial[target].grid_units())
                        .map(|(value, offset)| value - offset),
                    AssignmentOperation::Times | AssignmentOperation::Divide => None,
                };
                generators[target].push(generator);
                invariant.integral[target] &= constant
                    .and_then(NumericValue::grid_units)
                    .is_some_and(|units| units % WHOLE_UNITS == 0)
                    && matches!(
                        effect.operation(),
                        AssignmentOperation::Assign
                            | AssignmentOperation::Plus
                            | AssignmentOperation::Minus
                    );
                let guard = self.effect_guard(operator, effect);
                let repeated = !targets.insert(target);
                let image = match (effect.operation(), constant, delta) {
                    (AssignmentOperation::Assign, Some(value), _) if value.is_finite() => {
                        Interval::singleton(value)
                    }
                    (_, _, Some(delta)) if !guard.is_empty() => guard + Interval::singleton(delta),
                    _ => Interval::unbounded(),
                };
                if repeated {
                    invariant.bounds[target] = Interval::unbounded();
                }
                invariant.bounds[target] = hull(invariant.bounds[target], image);
                invariant.nonnegative[target] &= !repeated
                    && match effect.operation() {
                        AssignmentOperation::Assign => {
                            constant.is_some_and(|value| value >= ZERO_VALUE)
                        }
                        _ => {
                            delta.is_some_and(|delta| delta >= ZERO_VALUE || guard.lower >= -delta)
                        }
                    };
                invariant.nonpositive[target] &= !repeated
                    && match effect.operation() {
                        AssignmentOperation::Assign => {
                            constant.is_some_and(|value| value <= ZERO_VALUE)
                        }
                        _ => {
                            delta.is_some_and(|delta| delta <= ZERO_VALUE || guard.upper <= -delta)
                        }
                    };
                invariant.resources[target] &= invariant.integral[target]
                    && !repeated
                    && !effect.is_conditional()
                    && effect.conditions().is_empty()
                    && delta.is_some_and(|delta| delta >= ZERO_VALUE || guard.lower >= -delta);
            }
        }
        for variable in 0..initial.len() {
            if invariant.nonnegative[variable] {
                invariant.bounds[variable] = invariant.bounds[variable]
                    .intersection(&Interval::new(ZERO_VALUE, INF_VALUE, true, false));
            }
            if invariant.nonpositive[variable] {
                invariant.bounds[variable] = invariant.bounds[variable]
                    .intersection(&Interval::new(NEG_INF_VALUE, ZERO_VALUE, false, true));
            }
        }
        invariant.resource_bound = self.resource_bound(&invariant.resources);
        if let Some(total) = invariant.resource_bound {
            for (variable, &resource) in invariant.resources.iter().enumerate() {
                if resource {
                    invariant.bounds[variable] = invariant.bounds[variable]
                        .intersection(&Interval::closed(ZERO_VALUE, total));
                }
            }
        }
        let mut lattices: Vec<_> = initial
            .iter()
            .zip(generators)
            .enumerate()
            .map(|(variable, (&value, generators))| {
                let nominal = ValueLattice::from_generators(value, generators);
                let bound = invariant.bounds[variable];
                // Both input representation error and addition error are covered.
                let magnitude = bound.lower.abs().max(bound.upper.abs()).value();
                let increments = invariant.increments[variable];
                let magnitude = ((magnitude + increments).next_up() + 1.0).next_up();
                if magnitude.is_finite() && (f64::EPSILON * magnitude).next_up() < ABS_EPSILON / 2.0
                {
                    nominal
                } else {
                    nominal.arithmetic_closure()
                }
            })
            .collect();
        for (variable, view) in self.views.iter().enumerate() {
            let Some(view) = view else {
                continue;
            };
            let nominal = ValueLattice::from_generators(
                view.evaluate(initial),
                (0..self.task.get_operators().len()).map(|operator| {
                    view.operator_delta(OperatorIndex::from_usize(operator))
                        .grid_units()
                }),
            );
            let NumericExpression::Linear(terms) = &view.expression else {
                unreachable!("additive views are explicit linear sums");
            };
            if self.exact_affine_grid(terms, &invariant) {
                lattices[variable] = nominal;
            } else {
                // Independent term quanta remain sound if correlations are
                // eventually lost at large floating magnitudes.
                let quanta = terms.iter().map(|(source, coefficient)| {
                    let source = source.index();
                    let quantum = if invariant.changed[source] {
                        if lattices[source] == ValueLattice::GRID {
                            return None;
                        }
                        coefficient.value() * lattices[source].step().value()
                    } else {
                        coefficient.value() * initial[source].value()
                    };
                    let value = NumericValue::new(quantum);
                    (value.value() == quantum
                        && value
                            .grid_units()
                            .is_some_and(|units| units.unsigned_abs().is_multiple_of(244_140_625)))
                    .then(|| value.grid_units().expect("checked binary grid value"))
                });
                lattices[variable] = ValueLattice::from_generators(view.evaluate(initial), quanta)
                    .arithmetic_closure();
            }
        }
        lattices
    }

    fn effect_guard(&self, operator: &Operator, effect: &AssignmentEffect) -> Interval {
        operator
            .preconditions()
            .iter()
            .chain(effect.conditions())
            .filter_map(|fact| self.requirement(fact))
            .filter(|requirement| requirement.coordinate == effect.affected_var_id())
            .fold(Interval::unbounded(), |guard, requirement| {
                guard.intersection(&requirement.values)
            })
    }

    /// An inductive nonnegative, nonincreasing sum of integer resources.
    fn resource_bound(&self, resources: &[bool]) -> Option<NumericValue> {
        for operator in self.task.get_operators() {
            let delta: i128 = operator
                .assignment_effects()
                .iter()
                .filter(|effect| resources[effect.affected_var_id().index()])
                .map(|effect| {
                    i128::from(
                        constant_effect_delta(self.task, effect)
                            .expect("resource effects are constant additive increments")
                            .grid_units()
                            .expect("resource increments have integer grid units"),
                    )
                })
                .sum();
            if delta > 0 {
                return None;
            }
        }
        let total: i128 = self
            .task
            .get_initial_numeric_state_values()
            .iter()
            .zip(resources)
            .filter(|(_, resource)| **resource)
            .map(|(value, _)| {
                i128::from(value.grid_units().expect("integer resource initial value"))
            })
            .sum();
        if total >= (1i128 << 53) {
            return None;
        }
        Some(NumericValue::new(total as f64 / GRID_SCALE))
    }

    fn exact_affine_grid(
        &self,
        terms: &[(VariableIndex, NumericValue)],
        invariant: &NumericInvariant,
    ) -> bool {
        if terms
            .iter()
            .any(|(source, _)| !invariant.integral[source.index()])
        {
            return false;
        }
        let mut magnitude = 0.0;
        let mut resource_coefficient: f64 = 0.0;
        for (source, coefficient) in terms {
            let source = source.index();
            if invariant.resources[source] && invariant.resource_bound.is_some() {
                resource_coefficient = resource_coefficient.max(coefficient.abs().value());
            } else {
                let bound = invariant.bounds[source];
                let term = (coefficient.abs().value()
                    * bound.lower.abs().max(bound.upper.abs()).value())
                .next_up();
                magnitude = (magnitude + term).next_up();
            }
        }
        if let Some(total) = invariant.resource_bound {
            magnitude = (magnitude + (resource_coefficient * total.value()).next_up()).next_up();
        }
        if !affine_grid_is_exact(magnitude, terms.len()) {
            return false;
        }
        self.task.get_operators().iter().all(|operator| {
            let magnitude: f64 = terms
                .iter()
                .map(|(source, coefficient)| {
                    coefficient.abs().value()
                        * regular_additive_delta_for_operator(self.task, *source, operator)
                            .expect("an additive view has constant deltas")
                            .abs()
                            .value()
                })
                .fold(0.0, |bound, term| (bound + term.next_up()).next_up());
            affine_grid_is_exact(magnitude, terms.len())
        })
    }
}

fn hull(left: Interval, right: Interval) -> Interval {
    if right.is_empty() {
        return left;
    }
    Interval::closed(left.lower.min(right.lower), left.upper.max(right.upper))
}

/// Integer inputs and canonical coefficients give exact nominal grid products.
/// Bound coefficient representation and multiplication error, then each sum's
/// rounding error. A strict half-grid margin certifies the final grid unit.
fn affine_grid_is_exact(magnitude: f64, terms: usize) -> bool {
    if !magnitude.is_finite() || magnitude >= GRID_BOUND {
        return false;
    }
    if terms == 0 {
        return true;
    }
    let padded = (magnitude.next_up() + ABS_EPSILON).next_up();
    let half_ulp = (padded.next_up() - padded) / 2.0;
    let relative_error = f64::EPSILON / (1.0 - f64::EPSILON);
    let products = (relative_error * magnitude.next_up()).next_up();
    let additions = ((terms - 1) as f64 * half_ulp).next_up();
    (products + additions).next_up() < ABS_EPSILON / 2.0
}
