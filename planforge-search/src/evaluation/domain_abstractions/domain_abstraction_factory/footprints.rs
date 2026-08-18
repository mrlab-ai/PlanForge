use super::*;

pub(super) fn precise_operator_region_for_transition(
    transition: &AbstractTransition,
    concrete_op_id: usize,
    source_state_region: &StateRegion,
    abstract_operator_regions: &[AbstractOperatorRegions],
) -> Result<OperatorRegion> {
    let abstract_regions = abstract_operator_regions
        .get(transition.abstract_op_id)
        .with_context(|| {
            format!(
                "transition {} references missing abstract-operator region {}",
                transition.transition_id, transition.abstract_op_id
            )
        })?;
    let label_operator_region = abstract_regions
        .labels
        .iter()
        .find(|operator_region| operator_region.concrete_op_id == concrete_op_id)
        .with_context(|| {
            format!(
                "transition {} abstract operator {} has no operator region for concrete operator {concrete_op_id}",
                transition.transition_id, transition.abstract_op_id
            )
        })?;
    let source_region = state_region_intersection(
        source_state_region,
        &label_operator_region.source,
    )
    .with_context(|| {
        format!(
            "transition {} has an empty precise source operator region for concrete operator {concrete_op_id}",
            transition.transition_id
        )
    })?;
    Ok(OperatorRegion {
        concrete_op_id,
        source: Arc::new(source_region),
    })
}

struct DeterministicNumericEffectImage {
    image: Interval,
    inverse: DeterministicNumericEffectInverse,
}

#[derive(Debug, Clone, Copy)]
enum DeterministicNumericEffectInverse {
    Additive { delta: f64 },
    AssignmentConstant { value: f64 },
}

impl DeterministicNumericEffectImage {
    fn is_noop_for_source(&self, source_interval: Interval) -> bool {
        match self.inverse {
            DeterministicNumericEffectInverse::Additive { delta } => {
                delta.abs() <= float_tolerance::DIJKSTRA_EPSILON
            }
            DeterministicNumericEffectInverse::AssignmentConstant { value } => {
                interval_is_singleton(source_interval) && source_interval.contains(value)
            }
        }
    }

    fn inverse_source_for_target(&self, target_interval: Interval) -> Option<Interval> {
        match self.inverse {
            DeterministicNumericEffectInverse::Additive { delta } => {
                Some(shift_interval(target_interval, -delta))
            }
            DeterministicNumericEffectInverse::AssignmentConstant { value } => target_interval
                .contains(value)
                .then_some(Interval::unbounded()),
        }
    }
}

fn deterministic_numeric_effect_image(
    task: &dyn AbstractNumericTask,
    operator: &Operator,
    numeric_var_id: usize,
    source_interval: Interval,
) -> Option<DeterministicNumericEffectImage> {
    let initial_numeric = task.get_initial_numeric_state_values();
    let mut delta = 0.0;
    let mut assignment = None;
    let mut touched = false;
    for effect in operator
        .assignment_effects()
        .iter()
        .filter(|effect| effect.affected_var_id() == numeric_var_id)
    {
        if effect.is_conditional() || !effect.conditions().is_empty() {
            return None;
        }
        let rhs_value = match task.numeric_variables()[effect.var_id()].get_type() {
            NumericType::Constant | NumericType::Cost => {
                float_tolerance::canonicalize(*initial_numeric.get(effect.var_id())?)
            }
            _ => return None,
        };
        if !rhs_value.is_finite() {
            return None;
        }
        match effect.operation() {
            AssignmentOperation::Plus => {
                if assignment.is_some() {
                    return None;
                }
                delta = float_tolerance::canonicalize(delta + rhs_value);
                touched = true;
            }
            AssignmentOperation::Minus => {
                if assignment.is_some() {
                    return None;
                }
                delta = float_tolerance::canonicalize(delta - rhs_value);
                touched = true;
            }
            AssignmentOperation::Assign => {
                if touched || assignment.is_some() {
                    return None;
                }
                assignment = Some(rhs_value);
                touched = true;
            }
            AssignmentOperation::Times | AssignmentOperation::Divide => return None,
        }
    }
    if let Some(value) = assignment {
        Some(DeterministicNumericEffectImage {
            image: Interval::singleton(value),
            inverse: DeterministicNumericEffectInverse::AssignmentConstant { value },
        })
    } else if touched && delta.abs() > float_tolerance::DIJKSTRA_EPSILON {
        Some(DeterministicNumericEffectImage {
            image: shift_interval(source_interval, delta),
            inverse: DeterministicNumericEffectInverse::Additive { delta },
        })
    } else {
        None
    }
}

fn deterministic_affected_regular_numeric_vars(
    task: &dyn AbstractNumericTask,
    operator: &Operator,
) -> Vec<usize> {
    let mut deltas = vec![0.0; task.numeric_variables().len()];
    let mut assignments = Vec::new();
    for effect in operator.assignment_effects() {
        let affected_var_id = effect.affected_var_id();
        if task
            .numeric_variables()
            .get(affected_var_id)
            .is_none_or(|variable| variable.get_type() != &NumericType::Regular)
        {
            continue;
        }
        if effect.is_conditional() || !effect.conditions().is_empty() {
            continue;
        }
        if !matches!(
            effect.operation(),
            AssignmentOperation::Plus | AssignmentOperation::Minus | AssignmentOperation::Assign
        ) {
            continue;
        }
        if !matches!(
            task.numeric_variables()[effect.var_id()].get_type(),
            NumericType::Constant | NumericType::Cost
        ) {
            continue;
        }
        let Some(&rhs_value) = task.get_initial_numeric_state_values().get(effect.var_id()) else {
            continue;
        };
        let rhs_value = float_tolerance::canonicalize(rhs_value);
        if !rhs_value.is_finite() {
            continue;
        }
        match effect.operation() {
            AssignmentOperation::Plus => {
                deltas[affected_var_id] =
                    float_tolerance::canonicalize(deltas[affected_var_id] + rhs_value)
            }
            AssignmentOperation::Minus => {
                deltas[affected_var_id] =
                    float_tolerance::canonicalize(deltas[affected_var_id] - rhs_value)
            }
            AssignmentOperation::Assign => assignments.push(affected_var_id),
            AssignmentOperation::Times | AssignmentOperation::Divide => unreachable!(),
        }
    }
    let mut vars: Vec<usize> = deltas
        .iter()
        .enumerate()
        .filter_map(|(var_id, &delta)| {
            (delta.abs() > float_tolerance::DIJKSTRA_EPSILON).then_some(var_id)
        })
        .collect();
    vars.extend(assignments);
    vars.sort_unstable();
    vars.dedup();
    vars
}

fn shift_interval(interval: Interval, delta: f64) -> Interval {
    Interval::new(
        interval.lower + delta,
        interval.upper + delta,
        interval.lower_closed,
        interval.upper_closed,
    )
    .canonicalized()
}

fn interval_is_singleton(interval: Interval) -> bool {
    interval.lower == interval.upper && interval.lower_closed && interval.upper_closed
}

impl DomainAbstractionFactory {
    pub fn build_abstract_operator_regions(
        &self,
        task: &dyn AbstractNumericTask,
        operators: &[AbstractOperator],
    ) -> Result<Vec<AbstractOperatorRegions>> {
        operators
            .iter()
            .map(|operator| {
                let labels = operator
                    .concrete_op_ids
                    .iter()
                    .copied()
                    .map(|concrete_op_id| {
                        self.build_operator_region(task, operator, concrete_op_id)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(AbstractOperatorRegions { labels })
            })
            .collect()
    }

    pub(super) fn build_operator_region(
        &self,
        task: &dyn AbstractNumericTask,
        abstract_operator: &AbstractOperator,
        concrete_op_id: usize,
    ) -> Result<OperatorRegion> {
        let concrete_operator = task.get_operators().get(concrete_op_id).with_context(|| {
            format!("abstract operator references missing concrete operator {concrete_op_id}")
        })?;
        let abstract_source_region =
            self.state_region_from_facts(task, &abstract_operator.preconditions)?;
        // `state_region_from_facts` already initializes numeric intervals to
        // `Interval::unbounded()` for variables that have no partition fact in
        // the operator's preconditions, and to the partition's interval for
        // variables that do (which includes affected vars and variables pulled
        // in via comparison-axiom preconditions).
        //
        // The loop below tightens affected-variable intervals further by
        // intersecting with the inverse target image. For non-affected
        // variables that are pinned by a partition fact, the partition
        // interval is the tightest superset of the concrete preimage we can
        // recover at this layer, so we keep it. Wiping it back to unbounded
        // would still be admissible, but it would force the cost-partitioning
        // overlap check to treat distinct partitions as universally
        // overlapping on those axes, which is the over-conservativeness that
        // hides per-region cost claims.
        let mut source_region = abstract_source_region.clone();
        let target_region =
            self.state_region_from_facts(task, &abstract_operator.regression_preconditions)?;
        let mut affected_numeric_dimensions =
            deterministic_affected_regular_numeric_vars(task, concrete_operator);
        for (numeric_var_id, view) in self.additive_numeric_views.iter() {
            if self.numeric_domain_sizes[numeric_var_id] > 1
                && view.operator_delta(concrete_op_id)?.abs() >= float_tolerance::DIJKSTRA_EPSILON
            {
                affected_numeric_dimensions.push(numeric_var_id);
            }
        }
        affected_numeric_dimensions.sort_unstable();
        affected_numeric_dimensions.dedup();

        for numeric_var_id in affected_numeric_dimensions {
            ensure!(
                numeric_var_id < abstract_source_region.numeric.len(),
                "abstract operator references affected numeric var {numeric_var_id}, but operator region has {} numeric vars",
                abstract_source_region.numeric.len()
            );
            let source_interval = abstract_source_region.numeric[numeric_var_id];
            let effect_image = if let Some(view) = self.additive_numeric_views.get(numeric_var_id) {
                let delta = view.operator_delta(concrete_op_id)?;
                Some(DeterministicNumericEffectImage {
                    image: shift_interval(source_interval, delta),
                    inverse: DeterministicNumericEffectInverse::Additive { delta },
                })
            } else {
                deterministic_numeric_effect_image(
                    task,
                    concrete_operator,
                    numeric_var_id,
                    source_interval,
                )
            };
            let effect_image = effect_image.with_context(|| {
                format!(
                    "restricted SNP operator {concrete_op_id} has no exact deterministic effect image for numeric variable {numeric_var_id}"
                )
            })?;
            if effect_image.is_noop_for_source(source_interval) {
                continue;
            }
            ensure!(
                !effect_image.image.is_empty(),
                "restricted SNP operator {concrete_op_id} has an empty effect image for numeric variable {numeric_var_id}: source={source_interval:?}, image={:?}",
                effect_image.image,
            );
            let target_interval = target_region.numeric[numeric_var_id];
            let inverse_source = effect_image
                .inverse_source_for_target(target_interval)
                .with_context(|| {
                    format!(
                        "restricted SNP operator {concrete_op_id} cannot reach abstract target {target_interval:?} for numeric variable {numeric_var_id}"
                    )
                })?;
            let regressed_source = source_interval.intersection(&inverse_source);
            ensure!(
                !regressed_source.is_empty(),
                "restricted SNP operator {concrete_op_id} has an empty regressed source operator region for numeric variable {numeric_var_id}: source={source_interval:?}, target={target_interval:?}, image={:?}, inverse_source={inverse_source:?}",
                effect_image.image,
            );
            if regressed_source != source_interval {
                Arc::make_mut(&mut source_region.numeric)[numeric_var_id] = regressed_source;
            }
        }

        Ok(OperatorRegion {
            concrete_op_id,
            source: Arc::new(source_region),
        })
    }
}
