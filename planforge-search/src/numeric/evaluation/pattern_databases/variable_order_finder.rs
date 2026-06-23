#[cfg(test)]
mod tests;

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::fmt;

use planforge_sas::numeric::numeric_task::{AbstractNumericTask, NumericType};

use super::causal_graph::{CausalGraphVariable, MixedCausalGraph};
use super::numeric_support::NumericSupportContext;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum GreedyVariableOrderType {
    CgGoalLevel,
    CgGoalRandom,
    #[default]
    GoalCgLevel,
}

impl fmt::Display for GreedyVariableOrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::CgGoalLevel => "cg_goal_level",
            Self::CgGoalRandom => "cg_goal_random",
            Self::GoalCgLevel => "goal_cg_level",
        };
        write!(f, "{name}")
    }
}

impl crate::config::sealed::Sealed for GreedyVariableOrderType {}

impl crate::config::FromOptionValue for GreedyVariableOrderType {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "cg_goal_level" => Ok(Self::CgGoalLevel),
            "cg_goal_random" => Ok(Self::CgGoalRandom),
            "goal_cg_level" => Ok(Self::GoalCgLevel),
            other => Err(format!("invalid GreedyVariableOrderType `{other}`")),
        }
    }
}

pub struct VariableOrderFinder {
    remaining_vars: Vec<(usize, bool)>,
    is_goal_variable: Vec<bool>,
    is_numeric_goal_variable: Vec<bool>,
    is_causal_predecessor: Vec<bool>,
    num_propositional_variables: usize,
    variable_order_type: GreedyVariableOrderType,
    causal_graph: MixedCausalGraph,
}

impl VariableOrderFinder {
    pub(crate) fn new(
        task: &dyn AbstractNumericTask,
        numeric_support: &NumericSupportContext,
        variable_order_type: GreedyVariableOrderType,
        numeric_variables_first: bool,
        random_seed: u64,
    ) -> Self {
        let causal_graph = MixedCausalGraph::new(task);
        let mut remaining_vars = Vec::new();

        if numeric_variables_first {
            add_numeric_vars(task, numeric_support, &mut remaining_vars);
        }
        for var_id in 0..task.variables().len() {
            if task
                .get_variable_axiom_layer(var_id)
                .unwrap_or(None)
                .is_none()
                && !task
                    .comparison_axioms()
                    .iter()
                    .any(|axiom| axiom.get_affected_var_id() == var_id)
            {
                remaining_vars.push((var_id, false));
            }
        }
        if !numeric_variables_first {
            add_numeric_vars(task, numeric_support, &mut remaining_vars);
        }

        if variable_order_type == GreedyVariableOrderType::CgGoalRandom {
            let mut rng = SmallRng::seed_from_u64(random_seed as i64 as u64);
            remaining_vars.shuffle(&mut rng);
        }

        let mut is_goal_variable = vec![false; task.variables().len()];
        for goal_index in 0..task.get_num_goals() {
            let goal = task.get_goal_fact(goal_index);
            is_goal_variable[goal.var()] = true;
        }

        let helper_space_len = numeric_support.helper_space_len(task);
        let mut is_numeric_goal_variable = vec![false; helper_space_len];
        let goal_related_propositional_vars = collect_goal_related_propositional_closure(task);
        for (comparison_axiom_id, comparison_axiom) in task.comparison_axioms().iter().enumerate() {
            let affected_var_id = comparison_axiom.get_affected_var_id();
            if !goal_related_propositional_vars.contains(&affected_var_id) {
                continue;
            }
            for numeric_var_id in numeric_support.comparison_support_ids(task, comparison_axiom_id)
            {
                if numeric_var_id < is_numeric_goal_variable.len() {
                    is_numeric_goal_variable[numeric_var_id] = true;
                }
            }
        }

        Self {
            remaining_vars,
            is_goal_variable,
            is_numeric_goal_variable,
            is_causal_predecessor: vec![false; task.variables().len() + helper_space_len],
            num_propositional_variables: task.variables().len(),
            variable_order_type,
            causal_graph,
        }
    }

    pub fn done(&self) -> bool {
        self.remaining_vars.is_empty()
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(usize, bool)> {
        assert!(
            !self.done(),
            "VariableOrderFinder::next called with no remaining variables"
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

    fn select_next(&mut self, position: usize) -> (usize, bool) {
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
                .eff_pre_neighbors_of(CausalGraphVariable::Regular(var_id))
                .collect();
            for predecessor in predecessors {
                self.mark_causal_predecessor(predecessor);
            }
        }
        (var_id, is_numeric)
    }

    fn mark_causal_predecessor(&mut self, variable: CausalGraphVariable) {
        let index = match variable {
            CausalGraphVariable::Regular(var_id) => var_id,
            CausalGraphVariable::Numeric(var_id) => self.num_propositional_variables + var_id,
        };
        if index < self.is_causal_predecessor.len() {
            self.is_causal_predecessor[index] = true;
        }
    }

    fn find_causal_predecessor(&self) -> Option<usize> {
        self.remaining_vars
            .iter()
            .position(|&(var_id, is_numeric)| {
                let index = if is_numeric {
                    self.num_propositional_variables + var_id
                } else {
                    var_id
                };
                self.is_causal_predecessor
                    .get(index)
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
                        .get(var_id)
                        .copied()
                        .unwrap_or(false)
                } else {
                    self.is_goal_variable.get(var_id).copied().unwrap_or(false)
                }
            })
    }
}

fn collect_goal_related_propositional_closure(task: &dyn AbstractNumericTask) -> Vec<usize> {
    let mut goal_related: Vec<usize> = (0..task.get_num_goals())
        .map(|goal_id| task.get_goal_fact(goal_id).var())
        .collect();
    goal_related.sort_unstable();
    goal_related.dedup();

    loop {
        let mut changed = false;
        for axiom in task.axioms() {
            let affected_var_id = axiom.var_id();
            if goal_related.binary_search(&affected_var_id).is_ok() {
                for condition in axiom.conditions() {
                    if goal_related.binary_search(&condition.var()).is_err() {
                        goal_related.push(condition.var());
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
        goal_related.sort_unstable();
        goal_related.dedup();
    }

    goal_related
}

fn add_numeric_vars(
    task: &dyn AbstractNumericTask,
    numeric_support: &NumericSupportContext,
    remaining_vars: &mut Vec<(usize, bool)>,
) {
    for numeric_var_id in 0..numeric_support.helper_space_len(task) {
        let is_regular = task
            .numeric_variables()
            .get(numeric_var_id)
            .map(|numeric_var| numeric_var.get_type() == &NumericType::Regular)
            .unwrap_or_else(|| numeric_support.is_helper_var_id(task, numeric_var_id));
        if is_regular {
            remaining_vars.push((numeric_var_id, true));
        }
    }
}
