#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use planforge_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, NumericType,
};

use super::numeric_support::NumericSupportContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CausalGraphVariable {
    Regular(usize),
    Numeric(usize),
}

#[derive(Debug, Default, Clone)]
pub struct MixedCausalGraph {
    eff_predecessors: BTreeMap<CausalGraphVariable, BTreeSet<CausalGraphVariable>>,
    predecessors: BTreeMap<CausalGraphVariable, BTreeSet<CausalGraphVariable>>,
    successors: BTreeMap<CausalGraphVariable, BTreeSet<CausalGraphVariable>>,
    goal_distances: BTreeMap<CausalGraphVariable, usize>,
    causal_levels: BTreeMap<CausalGraphVariable, usize>,
}

impl MixedCausalGraph {
    pub fn new(task: &dyn AbstractNumericTask) -> Self {
        let mut graph = Self::default();
        let numeric_support = NumericSupportContext::new(task);

        for var_id in 0..task.variables().len() {
            if is_cpp_regular_propositional_var(task, var_id) {
                graph.ensure_node(CausalGraphVariable::Regular(var_id));
            }
        }
        for numeric_var_id in 0..numeric_support.helper_space_len(task) {
            if is_cpp_regular_numeric_var(task, numeric_var_id, &numeric_support) {
                graph.ensure_node(CausalGraphVariable::Numeric(numeric_var_id));
            }
        }

        for operator in task.get_operators() {
            let precondition_sources =
                cpp_operator_precondition_sources(task, &numeric_support, operator.preconditions());
            let propositional_effect_targets: Vec<_> = operator
                .effects()
                .iter()
                .filter_map(|effect| {
                    is_cpp_regular_propositional_var(task, effect.var_id())
                        .then_some(CausalGraphVariable::Regular(effect.var_id()))
                })
                .collect();
            let numeric_effect_targets: Vec<_> = operator
                .assignment_effects()
                .iter()
                .filter_map(|effect| cpp_numeric_effect_target(task, &numeric_support, effect))
                .collect();

            for target in propositional_effect_targets
                .iter()
                .chain(numeric_effect_targets.iter())
                .copied()
            {
                for source in precondition_sources.iter().copied() {
                    graph.add_pre_eff_arc(source, target);
                }
            }

            let effect_targets: Vec<_> = propositional_effect_targets
                .into_iter()
                .chain(numeric_effect_targets.into_iter())
                .collect();

            for effect_index in 0..effect_targets.len() {
                for other_index in (effect_index + 1)..effect_targets.len() {
                    graph.add_eff_eff_edge(
                        effect_targets[effect_index],
                        effect_targets[other_index],
                    );
                }
            }
        }

        graph.compute_goal_distances(task);
        graph.compute_causal_levels();
        graph
    }

    pub fn predecessors_of(
        &self,
        variable: CausalGraphVariable,
    ) -> impl Iterator<Item = CausalGraphVariable> + '_ {
        self.predecessors
            .get(&variable)
            .into_iter()
            .flat_map(|predecessors| predecessors.iter().copied())
    }

    pub fn successors_of(
        &self,
        variable: CausalGraphVariable,
    ) -> impl Iterator<Item = CausalGraphVariable> + '_ {
        self.successors
            .get(&variable)
            .into_iter()
            .flat_map(|successors| successors.iter().copied())
    }

    pub fn eff_pre_neighbors_of(
        &self,
        variable: CausalGraphVariable,
    ) -> impl Iterator<Item = CausalGraphVariable> + '_ {
        self.eff_predecessors
            .get(&variable)
            .into_iter()
            .flat_map(|predecessors| predecessors.iter().copied())
    }

    pub fn goal_distance(&self, variable: CausalGraphVariable) -> Option<usize> {
        self.goal_distances.get(&variable).copied()
    }

    pub fn causal_level(&self, variable: CausalGraphVariable) -> Option<usize> {
        self.causal_levels.get(&variable).copied()
    }

    pub fn predecessor_count(&self, variable: CausalGraphVariable) -> usize {
        self.predecessors
            .get(&variable)
            .map(|predecessors| predecessors.len())
            .unwrap_or(0)
    }

    fn ensure_node(&mut self, variable: CausalGraphVariable) {
        self.eff_predecessors.entry(variable).or_default();
        self.predecessors.entry(variable).or_default();
        self.successors.entry(variable).or_default();
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
        self.successors.entry(source).or_default().insert(target);
        self.predecessors.entry(target).or_default().insert(source);
    }

    fn add_eff_eff_edge(&mut self, lhs: CausalGraphVariable, rhs: CausalGraphVariable) {
        self.ensure_node(lhs);
        self.ensure_node(rhs);
        if lhs == rhs {
            return;
        }
        self.successors.entry(lhs).or_default().insert(rhs);
        self.successors.entry(rhs).or_default().insert(lhs);
        self.predecessors.entry(lhs).or_default().insert(rhs);
        self.predecessors.entry(rhs).or_default().insert(lhs);
    }

    fn compute_goal_distances(&mut self, task: &dyn AbstractNumericTask) {
        let mut queue = VecDeque::new();
        for goal_index in 0..task.get_num_goals() {
            let goal_var = CausalGraphVariable::Regular(task.get_goal_fact(goal_index).var());
            if self.goal_distances.insert(goal_var, 0).is_none() {
                queue.push_back(goal_var);
            }
        }

        while let Some(variable) = queue.pop_front() {
            let distance = self.goal_distances[&variable];
            let predecessors: Vec<_> = self.predecessors_of(variable).collect();
            for predecessor in predecessors {
                if self.goal_distances.contains_key(&predecessor) {
                    continue;
                }
                self.goal_distances.insert(predecessor, distance + 1);
                queue.push_back(predecessor);
            }
        }
    }

    fn compute_causal_levels(&mut self) {
        let mut queue = VecDeque::new();
        for (&variable, predecessors) in &self.predecessors {
            if predecessors.is_empty() {
                self.causal_levels.insert(variable, 0);
                queue.push_back(variable);
            }
        }

        while let Some(variable) = queue.pop_front() {
            let level = self.causal_levels[&variable];
            let successors: Vec<_> = self
                .successors
                .get(&variable)
                .into_iter()
                .flat_map(|successors| successors.iter().copied())
                .collect();

            for successor in successors {
                let next_level = level + 1;
                let should_enqueue = match self.causal_levels.get(&successor).copied() {
                    Some(existing_level) if existing_level <= next_level => false,
                    Some(_) => {
                        self.causal_levels.insert(successor, next_level);
                        true
                    }
                    None => {
                        self.causal_levels.insert(successor, next_level);
                        true
                    }
                };
                if should_enqueue {
                    queue.push_back(successor);
                }
            }
        }

        for &variable in self.predecessors.keys() {
            self.causal_levels.entry(variable).or_insert(usize::MAX / 2);
        }
    }
}

fn cpp_operator_precondition_sources(
    task: &dyn AbstractNumericTask,
    numeric_support: &NumericSupportContext,
    preconditions: &[planforge_sas::numeric::numeric_task::ExplicitFact],
) -> Vec<CausalGraphVariable> {
    let mut sources = BTreeSet::new();
    for fact in preconditions {
        if let Some(comparison_axiom_id) = comparison_axiom_id_for_var(task, fact.var()) {
            if let Some(numeric_var_id) =
                cpp_regular_numeric_condition_var_id(task, numeric_support, comparison_axiom_id)
            {
                sources.insert(CausalGraphVariable::Numeric(numeric_var_id));
            }
        } else if is_cpp_regular_propositional_var(task, fact.var()) {
            sources.insert(CausalGraphVariable::Regular(fact.var()));
        }
    }
    sources.into_iter().collect()
}

fn is_cpp_regular_propositional_var(task: &dyn AbstractNumericTask, var_id: usize) -> bool {
    if task
        .get_variable_axiom_layer(var_id)
        .unwrap_or(None)
        .is_some()
    {
        return false;
    }
    comparison_axiom_id_for_var(task, var_id).is_none()
}

fn is_cpp_regular_numeric_var(
    task: &dyn AbstractNumericTask,
    numeric_var_id: usize,
    numeric_support: &NumericSupportContext,
) -> bool {
    task.numeric_variables()
        .get(numeric_var_id)
        .map(|numeric_var| numeric_var.get_type() == &NumericType::Regular)
        .unwrap_or_else(|| numeric_support.is_helper_var_id(task, numeric_var_id))
}

fn cpp_numeric_effect_target(
    task: &dyn AbstractNumericTask,
    numeric_support: &NumericSupportContext,
    effect: &AssignmentEffect,
) -> Option<CausalGraphVariable> {
    if !is_cpp_regular_numeric_var(task, effect.affected_var_id(), numeric_support) {
        return None;
    }

    if matches!(
        effect.operation(),
        AssignmentOperation::Plus | AssignmentOperation::Minus
    ) && task
        .numeric_variables()
        .get(effect.var_id())
        .is_some_and(|var| var.get_type() == &NumericType::Constant)
        && task
            .get_initial_numeric_state_values()
            .get(effect.var_id())
            .copied()
            == Some(0.0)
    {
        return None;
    }

    Some(CausalGraphVariable::Numeric(effect.affected_var_id()))
}

fn comparison_axiom_id_for_var(task: &dyn AbstractNumericTask, var_id: usize) -> Option<usize> {
    task.comparison_axioms()
        .iter()
        .position(|axiom| axiom.get_affected_var_id() == var_id)
}

pub(crate) fn cpp_regular_numeric_condition_var_id(
    task: &dyn AbstractNumericTask,
    numeric_support: &NumericSupportContext,
    comparison_axiom_id: usize,
) -> Option<usize> {
    let comparison_axiom = task.comparison_axioms().get(comparison_axiom_id)?;
    let left_id = comparison_axiom.get_left_var_id();
    let right_id = comparison_axiom.get_right_var_id();
    let left_nonconstant = !numeric_expr_is_constant(task, numeric_support, left_id);
    let right_nonconstant = !numeric_expr_is_constant(task, numeric_support, right_id);

    if left_nonconstant && right_nonconstant {
        return numeric_condition_side_var_id(task, numeric_support, right_id);
    }
    if right_nonconstant {
        return numeric_condition_side_var_id(task, numeric_support, right_id);
    }
    if left_nonconstant {
        return numeric_condition_side_var_id(task, numeric_support, left_id);
    }
    None
}

fn numeric_condition_side_var_id(
    task: &dyn AbstractNumericTask,
    numeric_support: &NumericSupportContext,
    numeric_var_id: usize,
) -> Option<usize> {
    if let Some(helper_id) = numeric_support.helper_id_for_source(numeric_var_id) {
        return Some(helper_id);
    }
    if task
        .numeric_variables()
        .get(numeric_var_id)
        .is_some_and(|var| var.get_type() == &NumericType::Regular)
    {
        return Some(numeric_var_id);
    }
    let support_ids = numeric_support.numeric_var_support_ids(task, numeric_var_id);
    (support_ids.len() == 1).then_some(support_ids[0])
}

fn numeric_expr_is_constant(
    task: &dyn AbstractNumericTask,
    numeric_support: &NumericSupportContext,
    numeric_var_id: usize,
) -> bool {
    match task
        .numeric_variables()
        .get(numeric_var_id)
        .map(|var| var.get_type())
    {
        Some(NumericType::Constant | NumericType::Cost) => true,
        Some(NumericType::Regular) => false,
        Some(NumericType::Derived) => {
            if numeric_support
                .helper_id_for_source(numeric_var_id)
                .is_some()
            {
                return false;
            }
            let Some(axiom_id) = numeric_support.assignment_axiom_id_for(numeric_var_id) else {
                return false;
            };
            let Some(axiom) = task.assignment_axioms().get(axiom_id) else {
                return false;
            };
            numeric_expr_is_constant(task, numeric_support, axiom.get_left_var_id())
                && numeric_expr_is_constant(task, numeric_support, axiom.get_right_var_id())
        }
        None => false,
    }
}
