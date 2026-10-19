//! The task as a Cartesian abstraction sees it: operator costs, which
//! operators each split can affect, the coordinates, and the refinement
//! configuration. The child modules test regions and transitions against it.

use super::*;

mod regions;
mod split_choice;
mod transitions;

pub(super) struct CartesianSemantics<'task> {
    pub(super) task: &'task dyn AbstractNumericTask,
    pub(super) propositional_axioms_by_prop_var: Vec<Vec<usize>>,
    pub(super) operator_costs: Vec<NumericValue>,
    pub(super) prop_split_dependent_operators: Vec<OperatorBitSet>,
    pub(super) numeric_split_dependent_operators: Vec<OperatorBitSet>,
    pub(super) coordinates: Coordinates<'task>,
    pub(super) numeric_integer_lattice: Vec<bool>,
    pub(super) icaps_random: Option<RefCell<ArtifactMt19937>>,
    pub(super) pick: Icaps26SplitSelection,
    pub(super) target_split_boundaries: Vec<NumericValue>,
}

pub(super) fn mark_fact_split_dependencies(
    task: &dyn AbstractNumericTask,
    fact: &ExplicitFact,
    propositional_axioms_by_prop_var: &[Vec<usize>],
    visiting: &mut [bool],
    prop_dependencies: &mut [bool],
    numeric_dependencies: &mut [bool],
) -> Result<()> {
    let var_id = fact.var_index();
    if let Some(condition) = task.numeric_conditions().for_var(var_id) {
        let mut dimensions = condition.regular_numeric_var_dependencies().to_vec();
        dimensions.extend(comparison_refinement_dimensions(task, condition));
        dimensions.sort_unstable();
        dimensions.dedup();
        for numeric_var_id in dimensions {
            numeric_dependencies[numeric_var_id.index()] = true;
        }
        return Ok(());
    }
    if propositional_axioms_by_prop_var[var_id.index()].is_empty() {
        prop_dependencies[var_id.index()] = true;
        return Ok(());
    }
    ensure!(
        !visiting[var_id.index()],
        "cyclic propositional axiom dependency through variable {var_id}"
    );
    visiting[var_id.index()] = true;
    for &axiom_id in &propositional_axioms_by_prop_var[var_id.index()] {
        let axiom = task
            .axioms()
            .get(axiom_id)
            .with_context(|| format!("missing propositional axiom {axiom_id}"))?;
        for condition in axiom.conditions() {
            mark_fact_split_dependencies(
                task,
                condition,
                propositional_axioms_by_prop_var,
                visiting,
                prop_dependencies,
                numeric_dependencies,
            )?;
        }
    }
    visiting[var_id.index()] = false;
    Ok(())
}

impl<'task> CartesianSemantics<'task> {
    pub(super) fn task(&self) -> &dyn AbstractNumericTask {
        self.task
    }

    pub(super) fn propositional_axioms_by_prop_var(&self) -> &[Vec<usize>] {
        &self.propositional_axioms_by_prop_var
    }

    pub(super) fn operator_costs(&self) -> &[NumericValue] {
        &self.operator_costs
    }

    pub(super) fn additive_numeric_views(&self) -> &[Option<AdditiveNumericView>] {
        self.coordinates.views()
    }

    pub(super) fn numeric_integer_lattice(&self) -> &[bool] {
        &self.numeric_integer_lattice
    }

    pub(super) fn pick(&self) -> Icaps26SplitSelection {
        self.pick
    }

    pub(super) fn target_split_boundaries(&self) -> &[NumericValue] {
        &self.target_split_boundaries
    }

    pub(super) fn new(
        task: &'task dyn AbstractNumericTask,
        config: &Icaps26CartesianConfig,
    ) -> Result<Self> {
        validate_abstractable_goal(task).map_err(anyhow::Error::msg)?;
        let coordinates = Coordinates::new(task);
        for (op_id, op) in task.get_operators().iter().enumerate() {
            validate_abstraction_operator(task, op, OperatorIndex::from_usize(op_id))?;
        }

        let mut propositional_axioms_by_prop_var = vec![Vec::new(); task.get_num_variables()];
        for (axiom_id, axiom) in task.axioms().iter().enumerate() {
            let var_id = axiom.var_id();
            ensure!(
                var_id.index() < propositional_axioms_by_prop_var.len(),
                "propositional axiom {axiom_id} affects missing prop var {var_id}"
            );
            propositional_axioms_by_prop_var[var_id.index()].push(axiom_id);
        }
        let operator_costs = task
            .get_operators()
            .iter()
            .map(|op| metric_operator_cost_from_initial_values(task, op))
            .collect();
        let operator_count = task.get_operators().len();
        let additive_numeric_views = coordinates.views();
        let mut target_split_boundaries = task
            .numeric_variables()
            .iter()
            .enumerate()
            .filter(|(_, variable)| variable.get_type() == &NumericType::Constant)
            .filter_map(|(numeric_var_id, _)| {
                task.get_initial_numeric_state_values()
                    .get(numeric_var_id)
                    .copied()
                    .filter(|value| value.is_finite())
            })
            .map(float_tolerance::canonicalize)
            .collect::<Vec<_>>();
        target_split_boundaries.sort_by(NumericValue::total_cmp);
        target_split_boundaries.dedup_by(|left, right| left.to_bits() == right.to_bits());
        let mut prop_split_dependent_operators = (0..task.get_num_variables())
            .map(|_| OperatorBitSet::empty(operator_count))
            .collect::<Vec<_>>();
        let mut numeric_split_dependent_operators = (0..task.numeric_variables().len())
            .map(|_| OperatorBitSet::empty(operator_count))
            .collect::<Vec<_>>();
        for (op_id, op) in task.get_operators().iter().enumerate() {
            let mut prop_dependencies = vec![false; task.get_num_variables()];
            let mut numeric_dependencies = vec![false; task.numeric_variables().len()];
            let mut visiting = vec![false; task.get_num_variables()];
            for precondition in op.preconditions() {
                mark_fact_split_dependencies(
                    task,
                    precondition,
                    &propositional_axioms_by_prop_var,
                    &mut visiting,
                    &mut prop_dependencies,
                    &mut numeric_dependencies,
                )?;
            }
            for effect in op.effects() {
                let var_id = effect.var_id();
                if !task.numeric_conditions().is_condition_var(var_id)
                    && propositional_axioms_by_prop_var[var_id.index()].is_empty()
                {
                    prop_dependencies[var_id.index()] = true;
                }
            }
            for effect in op.assignment_effects() {
                let var_id = effect.affected_var_id();
                if task.numeric_variables()[var_id.index()].get_type() == &NumericType::Regular {
                    numeric_dependencies[var_id.index()] = true;
                }
            }
            for (numeric_var_id, view) in additive_numeric_views.iter().enumerate() {
                if view.as_ref().is_some_and(|view| {
                    view.operator_delta(OperatorIndex::from_usize(op_id))
                        .is_ok_and(|delta| delta.value().abs() > float_tolerance::SEARCH_EPSILON)
                }) {
                    numeric_dependencies[numeric_var_id] = true;
                }
            }
            debug_assert_eq!(
                prop_dependencies.len(),
                task.get_num_variables(),
                "operator {op_id} propositional dependency width changed"
            );
            for (var_id, depends) in prop_dependencies.into_iter().enumerate() {
                if depends {
                    prop_split_dependent_operators[var_id].insert(OperatorIndex::from_usize(op_id));
                }
            }
            for (var_id, depends) in numeric_dependencies.into_iter().enumerate() {
                if depends {
                    numeric_split_dependent_operators[var_id]
                        .insert(OperatorIndex::from_usize(op_id));
                }
            }
        }
        let icaps_random = (config.pick == Icaps26SplitSelection::Random)
            .then(|| RefCell::new(ArtifactMt19937::new(config.random_seed)));
        let initial_numeric = task.get_initial_numeric_state_values();
        let mut numeric_integer_lattice = initial_numeric
            .iter()
            .map(|&value| approximately_equal(value.value(), value.value().round()))
            .collect::<Vec<_>>();
        for op in task.get_operators() {
            for effect in op.assignment_effects() {
                let rhs = initial_numeric[effect.var_id().index()];
                let preserves_integers = match effect.operation() {
                    AssignmentOperation::Plus
                    | AssignmentOperation::Minus
                    | AssignmentOperation::Assign
                    | AssignmentOperation::Times => {
                        approximately_equal(rhs.value(), rhs.value().round())
                    }
                    AssignmentOperation::Divide => approximately_equal(rhs.value().abs(), 1.0),
                };
                numeric_integer_lattice[effect.affected_var_id().index()] &= preserves_integers;
            }
        }
        for (numeric_var_id, view) in additive_numeric_views.iter().enumerate() {
            let Some(view) = view else {
                continue;
            };
            let initial_value = view.evaluate(initial_numeric);
            numeric_integer_lattice[numeric_var_id] =
                approximately_equal(initial_value.value(), initial_value.value().round())
                    && (0..operator_count).all(|op_id| {
                        view.operator_delta(OperatorIndex::from_usize(op_id))
                            .is_ok_and(|delta| {
                                approximately_equal(delta.value(), delta.value().round())
                            })
                    });
        }
        Ok(Self {
            task,
            propositional_axioms_by_prop_var,
            operator_costs,
            prop_split_dependent_operators,
            numeric_split_dependent_operators,
            coordinates,
            numeric_integer_lattice,
            icaps_random,
            pick: config.pick,
            target_split_boundaries,
        })
    }
}
