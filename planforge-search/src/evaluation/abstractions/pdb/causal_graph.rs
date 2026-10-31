#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};

use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, ExplicitFact, NumericType,
    VariableIndex, ZERO_VALUE,
};

use crate::task_restriction::validate_restricted_task;

/// A node of the causal graph: a propositional variable no axiom derives, or
/// a regular numeric variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CausalGraphVariable {
    Propositional(VariableIndex),
    Numeric(VariableIndex),
}

/// The causal graph of a restricted task, which the pattern generators
/// extend patterns along.
///
/// A precondition-to-effect arc runs from every variable an operator's
/// precondition, an effect's condition or a numeric effect's operand reads to
/// every variable it changes; a comparison in a precondition stands for its
/// non-constant numeric operand. Effect-to-effect edges join the variables one
/// operator changes together.
#[derive(Debug, Default, Clone)]
pub struct CausalGraph {
    /// The precondition-to-effect predecessors of each variable.
    eff_predecessors: BTreeMap<CausalGraphVariable, BTreeSet<CausalGraphVariable>>,
    /// The precondition-to-effect predecessors and effect-to-effect
    /// neighbours of each variable.
    predecessors: BTreeMap<CausalGraphVariable, BTreeSet<CausalGraphVariable>>,
    /// The non-constant numeric operand of each comparison axiom.
    comparison_numeric_vars: Vec<Option<VariableIndex>>,
}

impl CausalGraph {
    pub fn new(task: &dyn AbstractNumericTask) -> Result<Self, String> {
        validate_restricted_task(task)?;
        let comparison_numeric_vars = task
            .comparison_axioms()
            .iter()
            .map(|axiom| {
                let regular = |id: VariableIndex| {
                    task.numeric_variables()[id.index()].get_type() == &NumericType::Regular
                };
                let (left, right) = (axiom.get_left_var_id(), axiom.get_right_var_id());
                if regular(right) {
                    Some(right)
                } else if regular(left) {
                    Some(left)
                } else {
                    None
                }
            })
            .collect();
        let mut graph = Self {
            comparison_numeric_vars,
            ..Self::default()
        };
        for var_id in 0..task.variables().len() {
            let var_id = VariableIndex::from_usize(var_id);
            if is_regular_propositional_var(task, var_id) {
                graph.ensure_node(CausalGraphVariable::Propositional(var_id));
            }
        }
        for (var_id, variable) in task.numeric_variables().iter().enumerate() {
            if variable.get_type() == &NumericType::Regular {
                graph.ensure_node(CausalGraphVariable::Numeric(VariableIndex::from_usize(
                    var_id,
                )));
            }
        }

        for operator in task.operators() {
            let precondition_sources = graph.precondition_sources(task, operator.preconditions());
            let effect_targets: Vec<CausalGraphVariable> = operator
                .effects()
                .iter()
                .filter(|effect| is_regular_propositional_var(task, effect.var_id()))
                .map(|effect| CausalGraphVariable::Propositional(effect.var_id()))
                .chain(
                    operator
                        .assignment_effects()
                        .iter()
                        .filter_map(|effect| numeric_effect_target(task, effect)),
                )
                .collect();

            for &target in &effect_targets {
                for &source in &precondition_sources {
                    graph.add_pre_eff_arc(source, target);
                }
            }
            for effect in operator.effects() {
                if !is_regular_propositional_var(task, effect.var_id()) {
                    continue;
                }
                let target = CausalGraphVariable::Propositional(effect.var_id());
                for source in graph.precondition_sources(task, effect.conditions()) {
                    graph.add_pre_eff_arc(source, target);
                }
            }
            for effect in operator.assignment_effects() {
                let Some(target) = numeric_effect_target(task, effect) else {
                    continue;
                };
                for source in graph.precondition_sources(task, effect.conditions()) {
                    graph.add_pre_eff_arc(source, target);
                }
                if task.numeric_variables()[effect.var_id().index()].get_type()
                    == &NumericType::Regular
                {
                    graph.add_pre_eff_arc(CausalGraphVariable::Numeric(effect.var_id()), target);
                }
            }
            for (index, &first) in effect_targets.iter().enumerate() {
                for &second in &effect_targets[index + 1..] {
                    graph.add_eff_eff_edge(first, second);
                }
            }
        }
        Ok(graph)
    }

    /// The precondition-to-effect predecessors and effect-to-effect
    /// neighbours of `variable`.
    pub fn predecessors_of(
        &self,
        variable: CausalGraphVariable,
    ) -> impl Iterator<Item = CausalGraphVariable> + '_ {
        self.predecessors
            .get(&variable)
            .into_iter()
            .flat_map(|predecessors| predecessors.iter().copied())
    }

    /// The precondition-to-effect predecessors of `variable`.
    pub fn eff_pre_neighbors_of(
        &self,
        variable: CausalGraphVariable,
    ) -> impl Iterator<Item = CausalGraphVariable> + '_ {
        self.eff_predecessors
            .get(&variable)
            .into_iter()
            .flat_map(|predecessors| predecessors.iter().copied())
    }

    /// The non-constant numeric operand of comparison axiom
    /// `comparison_axiom_id`, if it has one.
    pub fn comparison_numeric_var(&self, comparison_axiom_id: usize) -> Option<VariableIndex> {
        self.comparison_numeric_vars[comparison_axiom_id]
    }

    fn precondition_sources(
        &self,
        task: &dyn AbstractNumericTask,
        preconditions: &[ExplicitFact],
    ) -> Vec<CausalGraphVariable> {
        let mut sources = BTreeSet::new();
        for fact in preconditions {
            if let Some(comparison_axiom_id) =
                task.numeric_conditions().id_for_var(fact.var_index())
            {
                if let Some(numeric_var_id) = self.comparison_numeric_var(comparison_axiom_id) {
                    sources.insert(CausalGraphVariable::Numeric(numeric_var_id));
                }
            } else if is_regular_propositional_var(task, fact.var_index()) {
                sources.insert(CausalGraphVariable::Propositional(fact.var_index()));
            }
        }
        sources.into_iter().collect()
    }

    fn ensure_node(&mut self, variable: CausalGraphVariable) {
        self.eff_predecessors.entry(variable).or_default();
        self.predecessors.entry(variable).or_default();
    }

    fn add_pre_eff_arc(&mut self, source: CausalGraphVariable, target: CausalGraphVariable) {
        self.ensure_node(source);
        self.ensure_node(target);
        if source == target {
            return;
        }
        self.eff_predecessors
            .entry(target)
            .or_default()
            .insert(source);
        self.predecessors.entry(target).or_default().insert(source);
    }

    fn add_eff_eff_edge(&mut self, lhs: CausalGraphVariable, rhs: CausalGraphVariable) {
        self.ensure_node(lhs);
        self.ensure_node(rhs);
        if lhs == rhs {
            return;
        }
        self.predecessors.entry(lhs).or_default().insert(rhs);
        self.predecessors.entry(rhs).or_default().insert(lhs);
    }
}

fn numeric_effect_target(
    task: &dyn AbstractNumericTask,
    effect: &AssignmentEffect,
) -> Option<CausalGraphVariable> {
    if task
        .numeric_variables()
        .get(effect.affected_var_id().index())
        .is_none_or(|variable| variable.get_type() != &NumericType::Regular)
    {
        return None;
    }
    if matches!(
        effect.operation(),
        AssignmentOperation::Plus | AssignmentOperation::Minus
    ) && task
        .numeric_variables()
        .get(effect.var_id().index())
        .is_some_and(|variable| variable.get_type() == &NumericType::Constant)
        && task
            .initial_numeric_values()
            .get(effect.var_id().index())
            .copied()
            == Some(ZERO_VALUE)
    {
        return None;
    }
    Some(CausalGraphVariable::Numeric(effect.affected_var_id()))
}

fn is_regular_propositional_var(task: &dyn AbstractNumericTask, var_id: VariableIndex) -> bool {
    task.variable(var_id).axiom_layer().is_none()
        && !task.numeric_conditions().is_condition_var(var_id)
}
