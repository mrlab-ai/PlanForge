#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use planners_sas::numeric::axioms::PropositionalAxiom;
use planners_sas::numeric::numeric_task::AbstractNumericTask;

use super::numeric_support::NumericSupportContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CausalGraphVariable {
    Regular(usize),
    Numeric(usize),
}

#[derive(Debug, Default, Clone)]
pub struct MixedCausalGraph {
    predecessors: BTreeMap<CausalGraphVariable, BTreeSet<CausalGraphVariable>>,
    successors: BTreeMap<CausalGraphVariable, BTreeSet<CausalGraphVariable>>,
    goal_distances: BTreeMap<CausalGraphVariable, usize>,
    causal_levels: BTreeMap<CausalGraphVariable, usize>,
}

impl MixedCausalGraph {
    pub fn new(task: &dyn AbstractNumericTask) -> Self {
        let mut graph = Self::default();
        let numeric_support = NumericSupportContext::new(task);
        let comparison_axiom_by_affected_var = build_comparison_axiom_lookup(task);
        let propositional_axioms_by_affected_var = build_propositional_axiom_lookup(task);

        for var_id in 0..task.variables().len() {
            graph.ensure_node(CausalGraphVariable::Regular(var_id));
        }
        for numeric_var_id in 0..numeric_support.helper_space_len(task) {
            graph.ensure_node(CausalGraphVariable::Numeric(numeric_var_id));
        }

        for operator in task.get_operators() {
            let precondition_sources: Vec<CausalGraphVariable> = operator
                .preconditions()
                .iter()
                .flat_map(|fact| {
                    fact_support_sources(
                        task,
                        fact.var,
                        &comparison_axiom_by_affected_var,
                        &propositional_axioms_by_affected_var,
                        &numeric_support,
                    )
                })
                .collect();

            for effect in operator.effects() {
                let target = CausalGraphVariable::Regular(effect.var_id());
                for source in
                    precondition_sources
                        .iter()
                        .copied()
                        .chain(effect.conditions().iter().flat_map(|fact| {
                            fact_support_sources(
                                task,
                                fact.var,
                                &comparison_axiom_by_affected_var,
                                &propositional_axioms_by_affected_var,
                                &numeric_support,
                            )
                        }))
                {
                    graph.add_edge(source, target);
                }
            }

            for effect in operator.assignment_effects() {
                let target = CausalGraphVariable::Numeric(effect.affected_var_id());
                for source in precondition_sources
                    .iter()
                    .copied()
                    .chain(effect.conditions().iter().flat_map(|fact| {
                        fact_support_sources(
                            task,
                            fact.var,
                            &comparison_axiom_by_affected_var,
                            &propositional_axioms_by_affected_var,
                            &numeric_support,
                        )
                    }))
                    .chain(
                        numeric_support
                            .numeric_var_support_ids(task, effect.var_id())
                            .into_iter()
                            .map(CausalGraphVariable::Numeric),
                    )
                {
                    graph.add_edge(source, target);
                }
            }
        }

        for axiom in task.axioms() {
            let target = CausalGraphVariable::Regular(axiom.var_id());
            for source in axiom.conditions().iter().flat_map(|fact| {
                fact_support_sources(
                    task,
                    fact.var,
                    &comparison_axiom_by_affected_var,
                    &propositional_axioms_by_affected_var,
                    &numeric_support,
                )
            }) {
                graph.add_edge(source, target);
            }
        }

        for (comparison_axiom_id, axiom) in task.comparison_axioms().iter().enumerate() {
            let target_var_id = axiom.get_affected_var_id();
            let target = CausalGraphVariable::Regular(target_var_id);
            for source_var_id in numeric_support.comparison_support_ids(task, comparison_axiom_id) {
                graph.add_edge(CausalGraphVariable::Numeric(source_var_id), target);
            }
        }

        for auxiliary_numeric_var in numeric_support.auxiliary_numeric_vars() {
            let Some(axiom_id) = numeric_support
                .assignment_axiom_id_for(auxiliary_numeric_var.source_numeric_var_id)
            else {
                continue;
            };
            let Some(axiom) = task.assignment_axioms().get(axiom_id) else {
                continue;
            };
            let target = CausalGraphVariable::Numeric(auxiliary_numeric_var.helper_id);
            for source_var_id in numeric_support
                .numeric_var_support_ids(task, axiom.get_left_var_id())
                .into_iter()
                .chain(numeric_support.numeric_var_support_ids(task, axiom.get_right_var_id()))
            {
                graph.add_edge(CausalGraphVariable::Numeric(source_var_id), target);
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
        self.predecessors.entry(variable).or_default();
        self.successors.entry(variable).or_default();
    }

    fn add_edge(&mut self, source: CausalGraphVariable, target: CausalGraphVariable) {
        self.ensure_node(source);
        self.ensure_node(target);
        if source == target {
            return;
        }
        self.successors.entry(source).or_default().insert(target);
        self.predecessors.entry(target).or_default().insert(source);
    }

    fn compute_goal_distances(&mut self, task: &dyn AbstractNumericTask) {
        let mut queue = VecDeque::new();
        for goal_index in 0..task.get_num_goals() {
            let goal_var = CausalGraphVariable::Regular(task.get_goal_fact(goal_index).var);
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

fn build_comparison_axiom_lookup(task: &dyn AbstractNumericTask) -> BTreeMap<usize, usize> {
    task.comparison_axioms()
        .iter()
        .enumerate()
        .map(|(comparison_axiom_id, comparison_axiom)| {
            (comparison_axiom.get_affected_var_id(), comparison_axiom_id)
        })
        .collect()
}

fn build_propositional_axiom_lookup(
    task: &dyn AbstractNumericTask,
) -> BTreeMap<usize, Vec<PropositionalAxiom>> {
    let mut axioms_by_var: BTreeMap<usize, Vec<PropositionalAxiom>> = BTreeMap::new();
    for axiom in task.axioms() {
        axioms_by_var
            .entry(axiom.var_id())
            .or_default()
            .push(axiom.clone());
    }
    axioms_by_var
}

fn fact_support_sources(
    task: &dyn AbstractNumericTask,
    var_id: usize,
    comparison_axiom_by_affected_var: &BTreeMap<usize, usize>,
    propositional_axioms_by_affected_var: &BTreeMap<usize, Vec<PropositionalAxiom>>,
    numeric_support: &NumericSupportContext,
) -> Vec<CausalGraphVariable> {
    let mut sources = BTreeSet::new();
    collect_fact_support_sources(
        task,
        var_id,
        comparison_axiom_by_affected_var,
        propositional_axioms_by_affected_var,
        numeric_support,
        &mut BTreeSet::new(),
        &mut sources,
    );
    sources.into_iter().collect()
}

fn collect_fact_support_sources(
    task: &dyn AbstractNumericTask,
    var_id: usize,
    comparison_axiom_by_affected_var: &BTreeMap<usize, usize>,
    propositional_axioms_by_affected_var: &BTreeMap<usize, Vec<PropositionalAxiom>>,
    numeric_support: &NumericSupportContext,
    visiting_props: &mut BTreeSet<usize>,
    sources: &mut BTreeSet<CausalGraphVariable>,
) {
    if let Some(&comparison_axiom_id) = comparison_axiom_by_affected_var.get(&var_id) {
        for numeric_var_id in numeric_support.comparison_support_ids(task, comparison_axiom_id) {
            sources.insert(CausalGraphVariable::Numeric(numeric_var_id));
        }
        return;
    }

    if task
        .get_variable_axiom_layer(var_id)
        .unwrap_or(None)
        .is_none()
    {
        sources.insert(CausalGraphVariable::Regular(var_id));
        return;
    }

    if !visiting_props.insert(var_id) {
        return;
    }

    if let Some(axioms) = propositional_axioms_by_affected_var.get(&var_id) {
        for axiom in axioms {
            for condition in axiom.conditions() {
                collect_fact_support_sources(
                    task,
                    condition.var,
                    comparison_axiom_by_affected_var,
                    propositional_axioms_by_affected_var,
                    numeric_support,
                    visiting_props,
                    sources,
                );
            }
        }
    }

    visiting_props.remove(&var_id);
}
