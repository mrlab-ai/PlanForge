use std::collections::{BTreeMap, BTreeSet, VecDeque};

use planners_sas::numeric::numeric_task::AbstractNumericTask;

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

        for var_id in 0..task.variables().len() {
            graph.ensure_node(CausalGraphVariable::Regular(var_id));
        }
        for numeric_var_id in 0..task.numeric_variables().len() {
            graph.ensure_node(CausalGraphVariable::Numeric(numeric_var_id));
        }

        for operator in task.get_operators() {
            let precondition_sources: Vec<CausalGraphVariable> = operator
                .preconditions()
                .iter()
                .map(|fact| CausalGraphVariable::Regular(fact.var() as usize))
                .collect();

            for effect in operator.effects() {
                let target = CausalGraphVariable::Regular(effect.var_id() as usize);
                for source in precondition_sources.iter().copied().chain(
                    effect
                        .conditions()
                        .iter()
                        .map(|fact| CausalGraphVariable::Regular(fact.var() as usize)),
                ) {
                    graph.add_edge(source, target);
                }
            }

            for effect in operator.assignment_effects() {
                let target = CausalGraphVariable::Numeric(effect.affected_var_id() as usize);
                for source in precondition_sources
                    .iter()
                    .copied()
                    .chain(
                        effect
                            .conditions()
                            .iter()
                            .map(|fact| CausalGraphVariable::Regular(fact.var() as usize)),
                    )
                    .chain(std::iter::once(CausalGraphVariable::Numeric(
                        effect.var_id() as usize,
                    )))
                {
                    graph.add_edge(source, target);
                }
            }
        }

        for axiom in task.axioms() {
            let target = CausalGraphVariable::Regular(axiom.var_id() as usize);
            for source in axiom
                .conditions()
                .iter()
                .map(|fact| CausalGraphVariable::Regular(fact.var() as usize))
            {
                graph.add_edge(source, target);
            }
        }

        for axiom in task.comparison_axioms() {
            let Ok(target_var_id) = usize::try_from(axiom.get_affected_var_id()) else {
                continue;
            };
            let target = CausalGraphVariable::Regular(target_var_id);
            for source_var_id in [axiom.get_left_var_id(), axiom.get_right_var_id()] {
                let Ok(source_var_id) = usize::try_from(source_var_id) else {
                    continue;
                };
                graph.add_edge(CausalGraphVariable::Numeric(source_var_id), target);
            }
        }

        for axiom in task.assignment_axioms() {
            let target = CausalGraphVariable::Numeric(axiom.get_affected_var_id() as usize);
            graph.add_edge(
                CausalGraphVariable::Numeric(axiom.get_left_var_id() as usize),
                target,
            );
            graph.add_edge(
                CausalGraphVariable::Numeric(axiom.get_right_var_id() as usize),
                target,
            );
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
        for goal_index in 0..usize::try_from(task.get_num_goals().max(0)).unwrap_or(0) {
            let goal_var =
                CausalGraphVariable::Regular(task.get_goal_fact(goal_index as i32).var() as usize);
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

#[cfg(test)]
mod tests {
    use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator, PropositionalAxiom};
    use planners_sas::numeric::numeric_task::{
        AssignmentEffect, AssignmentOperation, ExplicitVariable, Fact, Metric, NumericRootTask,
        NumericType, NumericVariable, Operator,
    };

    use super::*;

    fn simple_var(name: &str, axiom_layer: i32) -> ExplicitVariable {
        ExplicitVariable::new(
            2,
            name.to_string(),
            vec![format!("{name}=0"), format!("{name}=1")],
            axiom_layer,
            1,
        )
    }

    #[test]
    fn causal_graph_collects_operator_and_axiom_dependencies() {
        let task = NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![
                simple_var("pre", -1),
                simple_var("goal", -1),
                ExplicitVariable::new(
                    3,
                    "cmp".to_string(),
                    vec!["t".to_string(), "f".to_string(), "u".to_string()],
                    0,
                    2,
                ),
            ],
            vec![
                NumericVariable::new("c".to_string(), NumericType::Constant, -1),
                NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            ],
            vec![Fact::new(1, 1)],
            vec![],
            vec![0, 0, 2],
            vec![1.0, 0.0],
            vec![Operator::new(
                "advance".to_string(),
                vec![Fact::new(0, 1), Fact::new(2, 0)],
                vec![planners_sas::numeric::numeric_task::Effect::new(
                    vec![],
                    1,
                    0,
                    1,
                )],
                vec![AssignmentEffect::new(
                    1,
                    AssignmentOperation::Plus,
                    0,
                    false,
                    vec![],
                )],
                1,
            )],
            vec![PropositionalAxiom::new(vec![Fact::new(0, 1)], 1, 0, 1)],
            vec![ComparisonAxiom::new(
                2,
                1,
                0,
                ComparisonOperator::GreaterThanOrEqual,
            )],
            vec![],
            (0, 0),
        );

        let graph = MixedCausalGraph::new(&task);

        assert!(
            graph
                .predecessors_of(CausalGraphVariable::Regular(1))
                .collect::<Vec<_>>()
                .contains(&CausalGraphVariable::Regular(0))
        );
        assert!(
            graph
                .predecessors_of(CausalGraphVariable::Regular(2))
                .collect::<Vec<_>>()
                .contains(&CausalGraphVariable::Numeric(1))
        );
        assert_eq!(
            graph.goal_distance(CausalGraphVariable::Regular(1)),
            Some(0)
        );
        assert_eq!(
            graph.goal_distance(CausalGraphVariable::Regular(0)),
            Some(1)
        );
    }
}
