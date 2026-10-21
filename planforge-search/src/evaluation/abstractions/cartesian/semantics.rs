//! The task as a Cartesian abstraction sees it: the shared CEGAR semantics,
//! and for every split dimension the operators whose transitions it can
//! change, which lets a split update only the transitions it affects.

use super::*;

mod transitions;

pub(super) struct CartesianSemantics<'task> {
    pub(super) cegar: CegarSemantics<'task>,
    pub(super) prop_split_dependent_operators: Vec<OperatorBitSet>,
    pub(super) numeric_split_dependent_operators: Vec<OperatorBitSet>,
}

pub(super) fn mark_fact_split_dependencies(
    task: &dyn AbstractNumericTask,
    fact: &ExplicitFact,
    regions: &RegionSemantics<'_>,
    visiting: &mut [bool],
    prop_dependencies: &mut [bool],
    numeric_dependencies: &mut [bool],
) -> Result<()> {
    let var_id = fact.var_index();
    if let Some(condition) = task.numeric_conditions().for_var(var_id) {
        let mut dimensions = condition.regular_numeric_var_dependencies().to_vec();
        dimensions.extend(regions.coordinates().refinement_dimensions(condition));
        dimensions.sort_unstable();
        dimensions.dedup();
        for numeric_var_id in dimensions {
            numeric_dependencies[numeric_var_id.index()] = true;
        }
        return Ok(());
    }
    if regions.axioms_for(var_id).is_empty() {
        prop_dependencies[var_id.index()] = true;
        return Ok(());
    }
    ensure!(
        !visiting[var_id.index()],
        "cyclic propositional axiom dependency through variable {var_id}"
    );
    visiting[var_id.index()] = true;
    for &axiom_id in regions.axioms_for(var_id) {
        let axiom = task
            .axioms()
            .get(axiom_id)
            .with_context(|| format!("missing propositional axiom {axiom_id}"))?;
        for condition in axiom.conditions() {
            mark_fact_split_dependencies(
                task,
                condition,
                regions,
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
    pub(super) fn cegar(&self) -> &CegarSemantics<'task> {
        &self.cegar
    }

    pub(super) fn task(&self) -> &'task dyn AbstractNumericTask {
        self.cegar.task()
    }

    pub(super) fn regions(&self) -> &RegionSemantics<'task> {
        self.cegar.regions()
    }

    pub(super) fn new(
        task: &'task dyn AbstractNumericTask,
        config: &CartesianAbstractionConfig,
    ) -> Result<Self> {
        let cegar = CegarSemantics::for_options(task, &config.cegar)?;
        let regions = cegar.regions();
        let additive_numeric_views = cegar.coordinates().views();
        let operator_count = task.get_operators().len();
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
                    regions,
                    &mut visiting,
                    &mut prop_dependencies,
                    &mut numeric_dependencies,
                )?;
            }
            for effect in op.effects() {
                let var_id = effect.var_id();
                if !task.numeric_conditions().is_condition_var(var_id)
                    && regions.axioms_for(var_id).is_empty()
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
        Ok(Self {
            cegar,
            prop_split_dependent_operators,
            numeric_split_dependent_operators,
        })
    }
}
