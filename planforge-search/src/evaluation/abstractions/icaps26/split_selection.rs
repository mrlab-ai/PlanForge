use planforge_sas::numeric_task::{INF_VALUE, NEG_INF_VALUE, ONE_VALUE};

use super::*;

fn split_child_regions(
    working: &WorkingAbstraction,
    split: &Split,
) -> Result<(CartesianSet, CartesianSet)> {
    let parent = working
        .states
        .get(split.state_id())
        .with_context(|| format!("missing split state {}", split.state_id()))?;
    match split {
        Split::Propositional {
            var_id,
            wanted,
            witness_value,
            ..
        } => {
            let current = parent
                .propositions()
                .get(var_id.index())
                .with_context(|| format!("split references missing prop var {var_id}"))?;
            ensure!(
                wanted.windows(2).all(|values| values[0] < values[1]),
                "propositional Cartesian split values must be sorted and unique: {wanted:?}"
            );
            let wanted_values = current
                .iter()
                .copied()
                .filter(|value| wanted.binary_search(value).is_ok())
                .collect::<Vec<_>>();
            let other_values = current
                .iter()
                .copied()
                .filter(|value| wanted.binary_search(value).is_err())
                .collect::<Vec<_>>();
            ensure!(
                !wanted_values.is_empty() && !other_values.is_empty(),
                "non-strict propositional Cartesian split on var {var_id}: current={current:?}, wanted={wanted:?}"
            );
            let witness_is_wanted = wanted_values.binary_search(witness_value).is_ok();
            let mut wanted_region = parent.clone();
            wanted_region.narrow_prop(*var_id, wanted_values);
            let mut other_region = parent.clone();
            other_region.narrow_prop(*var_id, other_values);
            Ok(if witness_is_wanted {
                (wanted_region, other_region)
            } else {
                (other_region, wanted_region)
            })
        }
        Split::Numeric {
            var_id,
            boundary,
            lower_includes_boundary,
            witness_value,
            integer_lattice,
            ..
        } => {
            let current = *parent
                .numeric
                .get(var_id.index())
                .with_context(|| format!("split references missing numeric var {var_id}"))?;
            ensure!(
                current.can_split_at(*boundary, *lower_includes_boundary),
                "non-strict numeric Cartesian split on var {var_id} at {boundary}: parent={current:?}, include_lower={lower_includes_boundary}"
            );
            let (lower, upper) = numeric_split_intervals(
                current,
                *boundary,
                *lower_includes_boundary,
                *integer_lattice,
            )?;
            let witness_is_lower = lower.contains(*witness_value);
            ensure!(
                witness_is_lower ^ upper.contains(*witness_value),
                "numeric split does not place witness {witness_value} in exactly one child"
            );
            let mut lower_region = parent.clone();
            Arc::make_mut(&mut lower_region.numeric)[var_id.index()] = lower;
            let mut upper_region = parent.clone();
            Arc::make_mut(&mut upper_region.numeric)[var_id.index()] = upper;
            Ok(if witness_is_lower {
                (lower_region, upper_region)
            } else {
                (upper_region, lower_region)
            })
        }
    }
}

pub(super) fn numeric_split_intervals(
    parent: Interval,
    boundary: NumericValue,
    lower_includes_boundary: bool,
    integer_lattice: bool,
) -> Result<(Interval, Interval)> {
    let (lower_bound, upper_bound, lower_closed, upper_closed) = if integer_lattice {
        ensure!(
            boundary.is_finite() && approximately_equal(boundary.value(), boundary.value().round()),
            "integer Cartesian split has non-integer boundary {boundary}"
        );
        if lower_includes_boundary {
            (boundary, boundary + ONE_VALUE, true, true)
        } else {
            (boundary - ONE_VALUE, boundary, true, true)
        }
    } else {
        (
            boundary,
            boundary,
            lower_includes_boundary,
            !lower_includes_boundary,
        )
    };
    let lower = parent.intersection(&Interval::new(
        NEG_INF_VALUE,
        lower_bound,
        false,
        lower_closed,
    ));
    let upper = parent.intersection(&Interval::new(upper_bound, INF_VALUE, upper_closed, false));
    ensure!(
        !lower.is_empty() && !upper.is_empty(),
        "non-strict numeric Cartesian split at {boundary}: parent={parent:?}, include_lower={lower_includes_boundary}, integer_lattice={integer_lattice}"
    );
    Ok((lower, upper))
}

pub(super) fn artifact_unwanted_score(working: &WorkingAbstraction, split: &Split) -> Result<f64> {
    let parent = working
        .states
        .get(split.state_id())
        .with_context(|| format!("missing split state {}", split.state_id()))?;
    match split {
        Split::Propositional { var_id, wanted, .. } => {
            let current = parent
                .propositions()
                .get(var_id.index())
                .with_context(|| format!("split references missing prop var {var_id}"))?;
            let wanted_count = current
                .iter()
                .filter(|value| wanted.binary_search(value).is_ok())
                .count();
            ensure!(
                wanted_count > 0 && wanted_count < current.len(),
                "ICAPS 2026 selector received a non-strict propositional split"
            );
            Ok((current.len() - wanted_count) as f64)
        }
        Split::Numeric {
            var_id,
            desired_contains_witness,
            ..
        } => {
            let (witness_region, other_region) = split_child_regions(working, split)?;
            let desired_region = if *desired_contains_witness {
                witness_region
            } else {
                other_region
            };
            let desired = desired_region.numeric[var_id.index()];
            if !desired.lower.is_finite() || !desired.upper.is_finite() {
                return Ok(f64::INFINITY);
            }
            let current = parent.numeric[var_id.index()];
            if !current.lower.is_finite() || !current.upper.is_finite() {
                return Ok(f64::INFINITY);
            }
            let current_values = integer_interval_cardinality(current);
            let desired_values = integer_interval_cardinality(desired);
            let unwanted_values = current_values - desired_values;
            ensure!(
                unwanted_values >= 0.0,
                "ICAPS 2026 desired interval contains more integer values than its parent"
            );
            let unwanted_width =
                ((current.upper - current.lower) - (desired.upper - desired.lower)).value();
            ensure!(
                unwanted_width >= 0.0,
                "ICAPS 2026 desired interval is wider than its parent"
            );
            // Preserve the artifact's integer-task ordering whenever the
            // excluded child contains an integer value. Strict fractional
            // splits can exclude no integer while still having positive
            // width; only that previously unsupported case uses the width.
            Ok(if unwanted_values > 0.0 {
                unwanted_values
            } else {
                unwanted_width
            })
        }
    }
}

fn integer_interval_cardinality(interval: Interval) -> f64 {
    debug_assert!(interval.lower.is_finite() && interval.upper.is_finite());
    let first = if interval.lower_closed {
        interval.lower.value().ceil()
    } else {
        interval.lower.value().floor() + 1.0
    };
    let last = if interval.upper_closed {
        interval.upper.value().floor()
    } else {
        interval.upper.value().ceil() - 1.0
    };
    (last - first + 1.0).max(0.0)
}

pub(super) fn select_refinement_split(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    mut candidates: Vec<Split>,
    _tag: u64,
) -> Result<Split> {
    ensure!(
        !candidates.is_empty(),
        "cannot select a Cartesian refinement from an empty candidate set"
    );
    let policy = semantics.pick();
    if policy == Icaps26SplitSelection::Random {
        if candidates.len() == 1 {
            return Ok(candidates.pop().expect("checked nonempty split set"));
        }
        let index = semantics.choose_icaps_random_index(candidates.len());
        return Ok(candidates.swap_remove(index));
    }
    let mut selected = 0;
    let mut selected_score = artifact_unwanted_score(working, &candidates[0])?;
    for (index, candidate) in candidates.iter().enumerate().skip(1) {
        let score = artifact_unwanted_score(working, candidate)?;
        let better = match policy {
            Icaps26SplitSelection::MinUnwanted => score < selected_score,
            Icaps26SplitSelection::MaxUnwanted => score > selected_score,
            Icaps26SplitSelection::Random => unreachable!("handled above"),
        };
        if better {
            selected = index;
            selected_score = score;
        }
    }
    Ok(candidates.swap_remove(selected))
}
