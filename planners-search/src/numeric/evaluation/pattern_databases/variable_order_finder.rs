#[cfg(test)]
mod tests;

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::fmt;

use super::causal_graph::{CausalGraphVariable, MixedCausalGraph};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum GreedyVariableOrderType {
    #[default]
    CgGoalLevel,
    Random,
    Level,
    ReverseLevel,
}

impl fmt::Display for GreedyVariableOrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::CgGoalLevel => "cg_goal_level",
            Self::Random => "random",
            Self::Level => "level",
            Self::ReverseLevel => "reverse_level",
        };
        write!(f, "{name}")
    }
}

pub fn order_variable_ids(
    variable_ids: &mut [usize],
    order_type: GreedyVariableOrderType,
    random_seed: u64,
) {
    match order_type {
        GreedyVariableOrderType::Random => {
            let mut rng = SmallRng::seed_from_u64(random_seed);
            variable_ids.shuffle(&mut rng);
        }
        GreedyVariableOrderType::ReverseLevel => {
            variable_ids.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
        }
        GreedyVariableOrderType::CgGoalLevel | GreedyVariableOrderType::Level => {
            variable_ids.sort_unstable();
        }
    }
}

pub fn order_causal_graph_variables(
    variable_ids: &mut [CausalGraphVariable],
    graph: &MixedCausalGraph,
    order_type: GreedyVariableOrderType,
    random_seed: u64,
) {
    match order_type {
        GreedyVariableOrderType::Random => {
            let mut rng = SmallRng::seed_from_u64(random_seed);
            variable_ids.shuffle(&mut rng);
        }
        GreedyVariableOrderType::CgGoalLevel => {
            variable_ids.sort_unstable_by_key(|&variable| {
                (
                    graph.goal_distance(variable).unwrap_or(usize::MAX),
                    graph.predecessor_count(variable),
                    variable,
                )
            });
        }
        GreedyVariableOrderType::Level => {
            variable_ids.sort_unstable_by_key(|&variable| {
                (graph.causal_level(variable).unwrap_or(usize::MAX), variable)
            });
        }
        GreedyVariableOrderType::ReverseLevel => {
            variable_ids.sort_unstable_by_key(|&variable| {
                (
                    std::cmp::Reverse(graph.causal_level(variable).unwrap_or(0)),
                    variable,
                )
            });
        }
    }
}
