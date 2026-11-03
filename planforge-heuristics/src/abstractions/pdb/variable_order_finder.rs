//! The order in which the greedy pattern generator adds variables, as Fast
//! Downward's variable order finder.

#[cfg(test)]
mod tests;

use planforge_sas::numeric_task::Task;
use planforge_sas::numeric_task::VariableKind;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use planforge_sas::numeric_task::{NumericType, VariableIndex};

use super::causal_graph::{CausalGraph, CausalGraphVariable};
use planforge_search::config::option_enum;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GreedyVariableOrderType {
    CgGoalLevel,
    CgGoalRandom,
    #[default]
    GoalCgLevel,
}

option_enum!("variable_order_type", GreedyVariableOrderType {
    CgGoalLevel => "cg_goal_level",
    CgGoalRandom => "cg_goal_random",
    GoalCgLevel => "goal_cg_level",
});

#[derive(Debug)]
pub struct VariableOrderFinder {
    remaining_vars: Vec<(VariableIndex, bool)>,
    is_goal_variable: Vec<bool>,
    is_numeric_goal_variable: Vec<bool>,
    is_causal_predecessor: Vec<bool>,
    num_propositional_variables: usize,
    variable_order_type: GreedyVariableOrderType,
    causal_graph: CausalGraph,
}

impl VariableOrderFinder {
    pub(crate) fn new(
        task: &Task,
        variable_order_type: GreedyVariableOrderType,
        numeric_variables_first: bool,
        random_seed: u64,
    ) -> Self {
        let causal_graph =
            CausalGraph::new(task).expect("PDB variable ordering requires a restricted task");
        let mut remaining_vars = Vec::new();

        if numeric_variables_first {
            add_numeric_vars(task, &mut remaining_vars);
        }
        for var_id in 0..task.variables().len() {
            if task.variable_kind(VariableIndex::from_usize(var_id)) == VariableKind::Regular {
                remaining_vars.push((VariableIndex::from_usize(var_id), false));
            }
        }
        if !numeric_variables_first {
            add_numeric_vars(task, &mut remaining_vars);
        }

        if variable_order_type == GreedyVariableOrderType::CgGoalRandom {
            let mut rng = SmallRng::seed_from_u64(random_seed);
            remaining_vars.shuffle(&mut rng);
        }

        let mut is_goal_variable = vec![false; task.variables().len()];
        for goal in task.goals() {
            is_goal_variable[goal.var()] = true;
        }

        let num_numeric_variables = task.numeric_variables().len();
        let mut is_numeric_goal_variable = vec![false; num_numeric_variables];
        // A numeric variable counts as a goal variable when the goal names a
        // comparison over it; `validate_abstractable_goal` guarantees that the
        // goal names the comparison variable itself, so `is_goal_variable`
        // already holds it.
        for (comparison_axiom_id, comparison_axiom) in task.comparison_axioms().iter().enumerate() {
            if !is_goal_variable[comparison_axiom.affected_var_id().index()] {
                continue;
            }
            if let Some(numeric_var_id) = causal_graph.comparison_numeric_var(comparison_axiom_id)
                && numeric_var_id.index() < is_numeric_goal_variable.len()
            {
                is_numeric_goal_variable[numeric_var_id.index()] = true;
            }
        }

        Self {
            remaining_vars,
            is_goal_variable,
            is_numeric_goal_variable,
            is_causal_predecessor: vec![false; task.variables().len() + num_numeric_variables],
            num_propositional_variables: task.variables().len(),
            variable_order_type,
            causal_graph,
        }
    }

    pub fn done(&self) -> bool {
        self.remaining_vars.is_empty()
    }

    /// The next variable in the order, or `None` when no remaining variable
    /// qualifies under the configured order type. Not `Iterator::next`: it is
    /// a hard error to call this once `done()` reports exhaustion.
    pub fn next_variable(&mut self) -> Option<(VariableIndex, bool)> {
        assert!(
            !self.done(),
            "VariableOrderFinder::next_variable called with no remaining variables"
        );

        match self.variable_order_type {
            GreedyVariableOrderType::CgGoalLevel | GreedyVariableOrderType::CgGoalRandom => {
                if let Some(position) = self.find_causal_predecessor() {
                    return Some(self.select_next(position));
                }
                if let Some(position) = self.find_goal_variable() {
                    return Some(self.select_next(position));
                }
            }
            GreedyVariableOrderType::GoalCgLevel => {
                if let Some(position) = self.find_goal_variable() {
                    return Some(self.select_next(position));
                }
                if let Some(position) = self.find_causal_predecessor() {
                    return Some(self.select_next(position));
                }
            }
        }

        None
    }

    fn select_next(&mut self, position: usize) -> (VariableIndex, bool) {
        let (var_id, is_numeric) = self.remaining_vars.remove(position);
        if is_numeric {
            let predecessors: Vec<_> = self
                .causal_graph
                .eff_pre_neighbors_of(CausalGraphVariable::Numeric(var_id))
                .collect();
            for predecessor in predecessors {
                self.mark_causal_predecessor(predecessor);
            }
        } else {
            let predecessors: Vec<_> = self
                .causal_graph
                .eff_pre_neighbors_of(CausalGraphVariable::Propositional(var_id))
                .collect();
            for predecessor in predecessors {
                self.mark_causal_predecessor(predecessor);
            }
        }
        (var_id, is_numeric)
    }

    fn mark_causal_predecessor(&mut self, variable: CausalGraphVariable) {
        let index = match variable {
            CausalGraphVariable::Propositional(var_id) => var_id,
            CausalGraphVariable::Numeric(var_id) => {
                VariableIndex::from_usize(self.num_propositional_variables + var_id.index())
            }
        };
        if index.index() < self.is_causal_predecessor.len() {
            self.is_causal_predecessor[index.index()] = true;
        }
    }

    fn find_causal_predecessor(&self) -> Option<usize> {
        self.remaining_vars
            .iter()
            .position(|&(var_id, is_numeric)| {
                let index = if is_numeric {
                    VariableIndex::from_usize(self.num_propositional_variables + var_id.index())
                } else {
                    var_id
                };
                self.is_causal_predecessor
                    .get(index.index())
                    .copied()
                    .unwrap_or(false)
            })
    }

    fn find_goal_variable(&self) -> Option<usize> {
        self.remaining_vars
            .iter()
            .position(|&(var_id, is_numeric)| {
                if is_numeric {
                    self.is_numeric_goal_variable
                        .get(var_id.index())
                        .copied()
                        .unwrap_or(false)
                } else {
                    self.is_goal_variable
                        .get(var_id.index())
                        .copied()
                        .unwrap_or(false)
                }
            })
    }
}

fn add_numeric_vars(task: &Task, remaining_vars: &mut Vec<(VariableIndex, bool)>) {
    for numeric_var_id in 0..task.numeric_variables().len() {
        let is_regular = task
            .numeric_variables()
            .get(numeric_var_id)
            .map(|numeric_var| numeric_var.numeric_type() == NumericType::Regular)
            .unwrap_or(false);
        if is_regular {
            remaining_vars.push((VariableIndex::from_usize(numeric_var_id), true));
        }
    }
}
