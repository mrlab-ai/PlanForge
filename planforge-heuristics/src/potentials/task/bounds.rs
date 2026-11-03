//! Bounds on the numeric features: the bounds of the primitive variables
//! mapped to the features, and the bounds the conditions of the task imply.

use super::*;

pub(super) fn sparse_bounds(bounds: Vec<FeatureBounds>) -> Vec<(usize, FeatureBounds)> {
    bounds
        .into_iter()
        .enumerate()
        .filter(|(_, bounds)| bounds.lower.is_finite() || bounds.upper.is_finite())
        .collect()
}

pub(super) fn intersect_sparse_bounds(
    local: &[(usize, FeatureBounds)],
    global: &[FeatureBounds],
) -> Vec<(usize, FeatureBounds)> {
    let mut result = Vec::new();
    let mut local_index = 0;
    for (feature_id, global_bounds) in global.iter().copied().enumerate() {
        let local_bounds = if local
            .get(local_index)
            .is_some_and(|(id, _)| *id == feature_id)
        {
            let bounds = local[local_index].1;
            local_index += 1;
            bounds
        } else {
            FeatureBounds::default()
        };
        let bounds = FeatureBounds {
            lower: local_bounds.lower.max(global_bounds.lower),
            upper: local_bounds.upper.min(global_bounds.upper),
        };
        if bounds.lower.is_finite() || bounds.upper.is_finite() {
            result.push((feature_id, bounds));
        }
    }
    debug_assert_eq!(local_index, local.len());
    result
}

pub(super) fn map_primitive_bounds_to_features(
    features: &[NumericFeature],
    primitive_bounds: &[FeatureBounds],
    target: &mut [FeatureBounds],
) {
    for (feature_id, feature) in features.iter().enumerate() {
        if !feature.affine {
            continue;
        }
        let mut lower = feature.constant;
        let mut upper = feature.constant;
        for &(variable_id, coefficient) in &feature.coefficients {
            let bounds = primitive_bounds[variable_id.index()];
            if coefficient >= 0.0 {
                lower += coefficient * bounds.lower;
                upper += coefficient * bounds.upper;
            } else {
                lower += coefficient * bounds.upper;
                upper += coefficient * bounds.lower;
            }
        }
        target[feature_id].lower = target[feature_id].lower.max(lower);
        target[feature_id].upper = target[feature_id].upper.min(upper);
    }
}

pub(super) fn register_closed_condition_bound(
    condition: &LinearNumericCondition,
    features: &[NumericFeature],
    feature_keys: &BTreeMap<Vec<(VariableIndex, NumericValue)>, usize>,
    bounds: &mut [FeatureBounds],
) -> Result<(), String> {
    if condition
        .coefficients
        .iter()
        .all(|coefficient| *coefficient == 0.0)
    {
        return Ok(());
    }
    let (canonical, condition_scale) = canonical_dense_feature(&condition.coefficients)
        .expect("nonconstant condition must have a canonical feature");
    let key = feature_key(&canonical);
    let feature_id = *feature_keys.get(&key).ok_or_else(|| {
        format!(
            "numeric_potential internal error: no feature for action condition `{}`",
            condition.name
        )
    })?;
    let (_, feature_scale) = canonical_feature(&features[feature_id].coefficients)
        .expect("matched feature must be nonconstant");
    let ratio = condition_scale / feature_scale;
    let threshold = features[feature_id].constant - condition.constant / ratio;
    if ratio > 0.0 {
        bounds[feature_id].lower = bounds[feature_id].lower.max(threshold);
    } else {
        bounds[feature_id].upper = bounds[feature_id].upper.min(threshold);
    }
    Ok(())
}

pub(super) fn register_condition_bound(
    condition: &LinearNumericCondition,
    features: &[NumericFeature],
    feature_keys: &BTreeMap<Vec<(VariableIndex, NumericValue)>, usize>,
    bounds: &mut [FeatureBounds],
    lattices: &[ValueLattice],
    goal_unsatisfiable: &mut bool,
) -> Result<(), String> {
    if condition
        .coefficients
        .iter()
        .all(|coefficient| *coefficient == 0.0)
    {
        let constant = NumericValue::new(condition.constant);
        *goal_unsatisfiable |= if condition.is_strictly_greater {
            constant <= ZERO_VALUE
        } else {
            constant < ZERO_VALUE
        };
        return Ok(());
    }
    let (canonical, condition_scale) = canonical_dense_feature(&condition.coefficients)
        .expect("nonconstant condition must have a canonical feature");
    let key = feature_key(&canonical);
    let feature_id = *feature_keys.get(&key).ok_or_else(|| {
        format!(
            "numeric_potential internal error: no feature for goal condition `{}`",
            condition.name
        )
    })?;
    let (_, feature_scale) = canonical_feature(&features[feature_id].coefficients)
        .expect("matched feature must be nonconstant");
    let ratio = condition_scale / feature_scale;
    let mut threshold = features[feature_id].constant - condition.constant / ratio;
    if ratio > 0.0 {
        if condition.is_strictly_greater {
            threshold = lattices[features[feature_id].source_numeric_var.index()]
                .split_bounds(NumericValue::new(threshold), true)
                .ok_or_else(|| {
                    format!("non-finite strict numeric potential threshold: {threshold}")
                })?
                .1
                .value();
        }
        bounds[feature_id].lower = bounds[feature_id].lower.max(threshold);
    } else {
        if condition.is_strictly_greater {
            threshold = lattices[features[feature_id].source_numeric_var.index()]
                .split_bounds(NumericValue::new(threshold), false)
                .ok_or_else(|| {
                    format!("non-finite strict numeric potential threshold: {threshold}")
                })?
                .0
                .value();
        }
        bounds[feature_id].upper = bounds[feature_id].upper.min(threshold);
    }
    if bounds[feature_id].lower > bounds[feature_id].upper {
        *goal_unsatisfiable = true;
    }
    Ok(())
}
