use super::*;

pub(super) fn splits_for_desired_facts(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    state_id: usize,
    facts: &[&ExplicitFact],
    prop_values: &[usize],
    numeric_values: &[f64],
    description: &str,
) -> Result<Vec<Split>> {
    let mut desired = semantics.trivial_region()?;
    for fact in facts {
        constrain_desired_region(semantics, &mut desired, fact)
            .with_context(|| format!("failed to construct desired region for {description}"))?;
    }
    splits_for_desired_region(
        working,
        semantics,
        state_id,
        &desired,
        prop_values,
        numeric_values,
        description,
    )
    .with_context(|| format!("failed to split desired region for {description}"))
}

fn constrain_desired_region(
    semantics: &CartesianSemantics<'_>,
    desired: &mut StateRegion,
    fact: &ExplicitFact,
) -> Result<()> {
    if let Some(comparison_axiom_id) = semantics.task().numeric_conditions().id_for_var(fact.var())
    {
        let (numeric_var_id, interval) =
            desired_comparison_interval(semantics, comparison_axiom_id, fact.value())?;
        let current = desired.numeric[numeric_var_id];
        let restricted = current.intersection(&interval);
        ensure!(
            !restricted.is_empty(),
            "desired comparison fact {fact:?} has an empty intersection on numeric variable {numeric_var_id}"
        );
        Arc::make_mut(&mut desired.numeric)[numeric_var_id] = restricted;
        return Ok(());
    }

    let supporting_axioms = semantics
        .propositional_axioms_by_prop_var
        .get(fact.var())
        .with_context(|| format!("missing propositional variable {}", fact.var()))?;
    if !supporting_axioms.is_empty() {
        let matching = supporting_axioms
            .iter()
            .filter(|&&axiom_id| semantics.task().axioms()[axiom_id].effect_value() == fact.value())
            .copied()
            .collect::<Vec<_>>();
        ensure!(
            matching.len() == 1,
            "desired derived fact {fact:?} requires exactly one supporting axiom, found {}",
            matching.len()
        );
        for condition in semantics.task().axioms()[matching[0]].conditions() {
            constrain_desired_region(semantics, desired, condition)?;
        }
        return Ok(());
    }

    let values = desired
        .propositions
        .get(fact.var())
        .with_context(|| format!("missing desired propositional variable {}", fact.var()))?;
    ensure!(
        values.binary_search(&(fact.value() as PropValueId)).is_ok(),
        "inconsistent desired fact {fact:?}"
    );
    Arc::make_mut(&mut desired.propositions)[fact.var()] = vec![fact.value() as PropValueId];
    Ok(())
}

pub(super) fn desired_comparison_interval(
    semantics: &CartesianSemantics<'_>,
    comparison_axiom_id: usize,
    desired_prop_value: usize,
) -> Result<(usize, Interval)> {
    let axiom = semantics
        .task
        .comparison_axioms()
        .get(comparison_axiom_id)
        .with_context(|| format!("missing comparison axiom {comparison_axiom_id}"))?;
    let left_id = axiom.get_left_var_id();
    let right_id = axiom.get_right_var_id();
    let left_type = semantics.task().numeric_variables()[left_id].get_type();
    let right_type = semantics.task().numeric_variables()[right_id].get_type();
    let left_is_coordinate = left_type == &NumericType::Regular
        || (left_type == &NumericType::Derived
            && semantics.additive_numeric_views()[left_id].is_some());
    let right_is_coordinate = right_type == &NumericType::Regular
        || (right_type == &NumericType::Derived
            && semantics.additive_numeric_views()[right_id].is_some());
    let initial = semantics.task().get_initial_numeric_state_values();
    let (numeric_var_id, threshold, mut operator) = match (
        left_is_coordinate,
        right_is_coordinate,
        left_type,
        right_type,
    ) {
        (true, false, _, NumericType::Constant) => {
            (left_id, initial[right_id], axiom.get_operator().clone())
        }
        (false, true, NumericType::Constant, _) => (
            right_id,
            initial[left_id],
            reverse_comparison_operator(axiom.get_operator()),
        ),
        _ => bail!(
            "desired-region Cartesian refinement requires each numeric condition to compare one exact additive coordinate with one constant; comparison axiom {comparison_axiom_id} has operand types {left_type:?} and {right_type:?}"
        ),
    };
    ensure!(
        threshold.is_finite(),
        "numeric comparison axiom {comparison_axiom_id} has non-finite threshold {threshold}"
    );
    if !comparison_truth(desired_prop_value)? {
        operator = negate_comparison_operator(&operator)?;
    }
    let threshold = float_tolerance::canonicalize(threshold);
    let interval = match operator {
        ComparisonOperator::LessThan => Interval::new(f64::NEG_INFINITY, threshold, false, false),
        ComparisonOperator::LessThanOrEqual => {
            Interval::new(f64::NEG_INFINITY, threshold, false, true)
        }
        ComparisonOperator::Equal => Interval::singleton(threshold),
        ComparisonOperator::GreaterThanOrEqual => {
            Interval::new(threshold, f64::INFINITY, true, false)
        }
        ComparisonOperator::GreaterThan => Interval::new(threshold, f64::INFINITY, false, false),
        ComparisonOperator::UnEqual => bail!(
            "desired-region refinement cannot represent a not-equal numeric condition as one interval (comparison axiom {comparison_axiom_id})"
        ),
    };
    Ok((numeric_var_id, interval))
}

fn reverse_comparison_operator(operator: &ComparisonOperator) -> ComparisonOperator {
    match operator {
        ComparisonOperator::LessThan => ComparisonOperator::GreaterThan,
        ComparisonOperator::LessThanOrEqual => ComparisonOperator::GreaterThanOrEqual,
        ComparisonOperator::Equal => ComparisonOperator::Equal,
        ComparisonOperator::GreaterThanOrEqual => ComparisonOperator::LessThanOrEqual,
        ComparisonOperator::GreaterThan => ComparisonOperator::LessThan,
        ComparisonOperator::UnEqual => ComparisonOperator::UnEqual,
    }
}

fn negate_comparison_operator(operator: &ComparisonOperator) -> Result<ComparisonOperator> {
    Ok(match operator {
        ComparisonOperator::LessThan => ComparisonOperator::GreaterThanOrEqual,
        ComparisonOperator::LessThanOrEqual => ComparisonOperator::GreaterThan,
        ComparisonOperator::Equal => ComparisonOperator::UnEqual,
        ComparisonOperator::GreaterThanOrEqual => ComparisonOperator::LessThan,
        ComparisonOperator::GreaterThan => ComparisonOperator::LessThanOrEqual,
        ComparisonOperator::UnEqual => ComparisonOperator::Equal,
    })
}

fn splits_for_desired_region(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    state_id: usize,
    desired: &StateRegion,
    prop_values: &[usize],
    numeric_values: &[f64],
    description: &str,
) -> Result<Vec<Split>> {
    let current = working
        .states
        .get(state_id)
        .with_context(|| format!("missing Cartesian state {state_id}"))?;
    let mut candidates = Vec::new();
    for (var_id, current_values) in current.propositions.iter().enumerate() {
        if semantics
            .task()
            .numeric_conditions()
            .is_condition_var(var_id)
            || !semantics.propositional_axioms_by_prop_var()[var_id].is_empty()
        {
            continue;
        }
        let witness = prop_values[var_id] as PropValueId;
        let desired_values = &desired.propositions[var_id];
        if desired_values.binary_search(&witness).is_ok() {
            continue;
        }
        ensure!(
            current_values.binary_search(&witness).is_ok(),
            "concrete witness {witness} is outside Cartesian prop var {var_id}"
        );
        let wanted = current_values
            .iter()
            .copied()
            .filter(|value| desired_values.binary_search(value).is_ok())
            .collect::<Vec<_>>();
        ensure!(
            !wanted.is_empty(),
            "desired region does not overlap Cartesian prop var {var_id}"
        );
        candidates.push(Split::Propositional {
            state_id,
            var_id,
            wanted,
            witness_value: witness,
            description: description.to_string(),
        });
    }

    for (var_id, (&parent, &target)) in current
        .numeric
        .iter()
        .zip(desired.numeric.iter())
        .enumerate()
    {
        let witness = numeric_values[var_id];
        if target.contains(witness) {
            continue;
        }
        ensure!(
            parent.contains(witness) && parent.intersects(&target),
            "desired numeric interval {target:?} does not overlap parent {parent:?} containing witness {witness} for var {var_id}"
        );
        let witness_is_below =
            witness < target.lower || (witness == target.lower && !target.lower_closed);
        let (boundary, lower_includes_boundary) = if witness_is_below {
            ensure!(
                target.lower.is_finite(),
                "desired lower split boundary is infinite"
            );
            (target.lower, !target.lower_closed)
        } else {
            ensure!(
                witness > target.upper || (witness == target.upper && !target.upper_closed),
                "numeric witness is neither below nor above desired interval"
            );
            ensure!(
                target.upper.is_finite(),
                "desired upper split boundary is infinite"
            );
            (target.upper, target.upper_closed)
        };
        let integer_lattice = semantics.numeric_integer_lattice()[var_id];
        numeric_split_intervals(parent, boundary, lower_includes_boundary, integer_lattice)
            .with_context(|| {
                format!(
                    "desired numeric split is not strict for var {var_id}: parent={parent:?}, target={target:?}, integer_lattice={integer_lattice}"
                )
            })?;
        candidates.push(Split::Numeric {
            state_id,
            var_id,
            boundary,
            lower_includes_boundary,
            witness_value: witness,
            desired_contains_witness: false,
            integer_lattice,
            description: description.to_string(),
        });
    }
    ensure!(
        !candidates.is_empty(),
        "flaw has no concrete value outside its desired region"
    );
    Ok(candidates)
}
