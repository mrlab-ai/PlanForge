//! The numeric features of a potential task: the linear and auxiliary
//! expressions potentials weigh, as the C++ implementation builds them, and
//! their canonical, sparse representation.

use super::*;

#[derive(Clone)]
pub(super) struct ParsedNumericExpression {
    coefficients: Vec<(VariableIndex, f64)>,
    constant: f64,
    affine: bool,
    is_constant: bool,
}

pub(super) fn build_cpp_numeric_features(
    task: &Task,
    implied_comparisons: &[VariableIndex],
) -> Result<Vec<NumericFeature>, String> {
    let mut features = Vec::new();
    for (numeric_var_id, variable) in task.numeric_variables().iter().enumerate() {
        if variable.numeric_type() != NumericType::Regular {
            continue;
        }
        let numeric_var_id = VariableIndex::from_usize(numeric_var_id);
        features.push(NumericFeature {
            name: variable.name().to_string(),
            coefficients: vec![(numeric_var_id, 1.0)],
            constant: 0.0,
            affine: true,
            source_numeric_var: numeric_var_id,
        });
    }

    let comparison_by_var: BTreeMap<VariableIndex, usize> = task
        .comparison_axioms()
        .iter()
        .enumerate()
        .map(|(comparison_id, comparison)| (comparison.affected_var_id(), comparison_id))
        .collect();
    let mut used_comparison_vars = Vec::new();
    let mut seen_comparisons = BTreeSet::new();
    {
        let mut record_fact = |fact: &ExplicitFact| {
            let var = fact.var_index();
            if comparison_by_var.contains_key(&var) && seen_comparisons.insert(var) {
                used_comparison_vars.push(var);
            }
        };
        // The C++ numeric_pdb_helper construction order: operator
        // preconditions, then the axiom bodies, then the goal. The C++
        // translator's goal was a derived atom whose axiom body named the
        // goal comparisons; here the goal names them itself.
        //
        // Only the variable of a body fact matters here, and every variable a
        // rule refuting `v` names comes from a body of a rule proving `v` --
        // so refuting rules never contribute a variable, nor a position in
        // the order, that the proving rules have not already contributed.
        for operator in task.operators() {
            for precondition in operator.preconditions() {
                record_fact(precondition);
            }
        }
        for axiom in task.axioms() {
            for condition in axiom.conditions() {
                record_fact(condition);
            }
        }
        for goal in task.goals() {
            record_fact(goal);
        }
    }
    // The comparisons the goal implies enter the goal, so they need features
    // too.
    for &comparison_var in implied_comparisons {
        if seen_comparisons.insert(comparison_var) {
            used_comparison_vars.push(comparison_var);
        }
    }

    let mut auxiliary_by_name = BTreeMap::new();
    for comparison_var in used_comparison_vars {
        let comparison_id = comparison_by_var[&comparison_var];
        let comparison = &task.comparison_axioms()[comparison_id];
        let left = parse_cpp_numeric_expression(
            task,
            comparison.left_var_id(),
            &mut auxiliary_by_name,
            &mut features,
        )?;
        let right = parse_cpp_numeric_expression(
            task,
            comparison.right_var_id(),
            &mut auxiliary_by_name,
            &mut features,
        )?;
        if !left.is_constant && !right.is_constant {
            insert_cpp_auxiliary_feature(
                task,
                comparison.left_var_id(),
                left,
                &mut auxiliary_by_name,
                &mut features,
            );
            insert_cpp_auxiliary_feature(
                task,
                comparison.right_var_id(),
                right,
                &mut auxiliary_by_name,
                &mut features,
            );
        }
    }
    Ok(features)
}

pub(super) fn parse_cpp_numeric_expression(
    task: &Task,
    numeric_var_id: VariableIndex,
    auxiliary_by_name: &mut BTreeMap<String, usize>,
    features: &mut Vec<NumericFeature>,
) -> Result<ParsedNumericExpression, String> {
    let expression = match task.linearize_numeric_var(numeric_var_id) {
        Ok(expression) => ParsedNumericExpression {
            is_constant: expression.is_constant(),
            coefficients: expression
                .coefficients
                .into_iter()
                .enumerate()
                .filter(|(_, value)| *value != 0.0)
                .map(|(id, coefficient)| (VariableIndex::from_usize(id), coefficient))
                .collect(),
            constant: expression.constant,
            affine: true,
        },
        Err(
            planforge_sas::utils::linear_effects::LinearizationError::NonLinearAssignmentAxiom {
                ..
            },
        ) => ParsedNumericExpression {
            coefficients: Vec::new(),
            constant: 0.0,
            affine: false,
            is_constant: false,
        },
        Err(error) => {
            return Err(format!(
                "numeric_potential cannot expand variable {numeric_var_id}: {error}"
            ));
        }
    };
    if task.numeric_type(numeric_var_id) == NumericType::Derived {
        insert_cpp_auxiliary_feature(
            task,
            numeric_var_id,
            expression.clone(),
            auxiliary_by_name,
            features,
        );
    }
    Ok(expression)
}

pub(super) fn insert_cpp_auxiliary_feature(
    task: &Task,
    source_numeric_var: VariableIndex,
    expression: ParsedNumericExpression,
    auxiliary_by_name: &mut BTreeMap<String, usize>,
    features: &mut Vec<NumericFeature>,
) -> usize {
    let name = task.numeric_variables()[source_numeric_var.index()]
        .name()
        .to_string();
    if let Some(feature_id) = auxiliary_by_name.get(&name) {
        return *feature_id;
    }
    if !expression.affine {
        auxiliary_by_name.insert(name, usize::MAX);
        return usize::MAX;
    }
    let feature_id = features.len();
    features.push(NumericFeature {
        name: name.clone(),
        coefficients: expression.coefficients,
        constant: expression.constant,
        affine: expression.affine,
        source_numeric_var,
    });
    auxiliary_by_name.insert(name, feature_id);
    feature_id
}

pub(super) fn canonical_feature(
    coefficients: &[(VariableIndex, f64)],
) -> Option<(Vec<(VariableIndex, f64)>, f64)> {
    let scale = coefficients
        .iter()
        .map(|(_, coefficient)| *coefficient)
        .find(|coefficient| *coefficient != 0.0)?;
    let canonical = coefficients
        .iter()
        .filter_map(|&(variable_id, coefficient)| {
            let value = coefficient / scale;
            (value != 0.0).then_some((variable_id, value))
        })
        .collect();
    Some((canonical, scale))
}

pub(super) fn canonical_dense_feature(
    coefficients: &[f64],
) -> Option<(Vec<(VariableIndex, f64)>, f64)> {
    let sparse = coefficients
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, coefficient)| *coefficient != 0.0)
        .map(|(variable_id, coefficient)| (VariableIndex::from_usize(variable_id), coefficient))
        .collect::<Vec<_>>();
    canonical_feature(&sparse)
}

pub(super) fn feature_key(
    coefficients: &[(VariableIndex, f64)],
) -> Vec<(VariableIndex, NumericValue)> {
    coefficients
        .iter()
        .map(|&(variable_id, value)| (variable_id, NumericValue::new(value)))
        .collect()
}

pub(super) fn dot_sparse(left: &[(VariableIndex, f64)], right: &[(VariableIndex, f64)]) -> f64 {
    let mut result = 0.0;
    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left.len() && right_index < right.len() {
        let (left_id, left_value) = left[left_index];
        let (right_id, right_value) = right[right_index];
        match left_id.cmp(&right_id) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                result += left_value * right_value;
                left_index += 1;
                right_index += 1;
            }
        }
    }
    result
}

pub(super) fn sparse_value(sparse: &[(VariableIndex, f64)], index: VariableIndex) -> f64 {
    sparse
        .binary_search_by_key(&index, |(id, _)| *id)
        .map(|position| sparse[position].1)
        .unwrap_or(0.0)
}
