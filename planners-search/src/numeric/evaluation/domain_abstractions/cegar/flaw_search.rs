pub mod flaw_selection;
pub mod progression;
pub mod state;
#[cfg(test)]
mod tests;

use anyhow::{Result, ensure};
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::utils::int_packer::IntDoublePacker;
use std::collections::BTreeSet;
use std::fmt;

use planners_sas::numeric::numeric_task::{AbstractNumericTask, ExplicitFact, Operator};

use serde::{Deserialize, Serialize};

use super::{determine_include_in_lower, make_prop_state_packer};
use crate::numeric::evaluation::domain_abstractions::comparison_expression::Interval;
use crate::numeric::evaluation::domain_abstractions::cegar::flaw_search::progression::get_progression_flaws;
use crate::numeric::evaluation::domain_abstractions::domain_abstraction::{
    ComparisonAxiomIndex, NumericPartitions,
};
use crate::numeric::evaluation::domain_abstractions::domain_abstraction_factory::WildcardPlanResult;
use crate::numeric::evaluation::domain_abstractions::utils::{fact_is_hold, get_initial_state};
use state::progress;

/// Mirrors numeric-FD's `NumericFlaw = tuple<int, ap_float, bool>`.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericFlaw {
    pub numeric_var_id: usize,
    pub value: f64,
    pub include_in_lower: bool,
}

/// Mirrors numeric-FD's `PropFlaw = pair<Fact, vector<NumericFlaw>>`.
#[derive(Debug, Clone, PartialEq)]
pub struct PropFlaw {
    pub fact: ExplicitFact,
    pub dependent_numeric_flaws: Vec<NumericFlaw>,
}

/// Mirrors numeric-FD's `Flaw = variant<PropFlaw, NumericFlaw>`.
#[derive(Debug, Clone, PartialEq)]
pub enum Flaw {
    Propositional(PropFlaw),
    Numeric(NumericFlaw),
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecEntirePlanMode {
    StopAtFirstFlaw,
    ExecuteEntirePlan,
}

impl fmt::Display for ExecEntirePlanMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StopAtFirstFlaw => write!(f, "stop_at_first_flaw"),
            Self::ExecuteEntirePlan => write!(f, "execute_entire_plan"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependentNumericRefinement {
    None,
    One,
    All,
}

pub fn get_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    wildcard_plan: &WildcardPlanResult,
    execute_entire_plan: bool,
) -> Result<Vec<Flaw>> {
    get_progression_flaws(task, partitions, wildcard_plan, execute_entire_plan)
}

#[allow(unused)]
fn score_flaw(
    flaw: &Flaw,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    _abstraction_size: usize,
) -> usize {
    match flaw {
        Flaw::Numeric(nf) => numeric_domain_sizes
            .get(nf.numeric_var_id)
            .copied()
            .unwrap_or(0),
        Flaw::Propositional(pf) => {
            let var_id = pf.fact.var;
            let base = domain_sizes.get(var_id).copied().unwrap_or(0);
            let max_dep = pf
                .dependent_numeric_flaws
                .iter()
                .filter_map(|nf| numeric_domain_sizes.get(nf.numeric_var_id).copied())
                .max()
                .unwrap_or(0);
            base + max_dep
        }
    }
}

fn dependent_numeric_flaws_for_comparison_prop_var(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    prop_var_id: usize,
    numeric_state: &[f64],
) -> Vec<NumericFlaw> {
    let Some(tree) = comparison_index.comparison_tree(prop_var_id) else {
        return vec![];
    };

    let mut out: Vec<NumericFlaw> = Vec::new();
    for dep_var_id in tree.regular_numeric_var_dependencies(task) {
        let Some(&concrete_value) = numeric_state.get(dep_var_id) else {
            continue;
        };
        let include_in_lower =
            determine_include_in_lower(tree, dep_var_id, concrete_value, numeric_state);

        if can_split_numeric_var(partitions, dep_var_id, concrete_value, include_in_lower) {
            out.push(NumericFlaw {
                numeric_var_id: dep_var_id,
                value: concrete_value,
                include_in_lower,
            });
        } else if can_split_numeric_var(partitions, dep_var_id, concrete_value, !include_in_lower) {
            out.push(NumericFlaw {
                numeric_var_id: dep_var_id,
                value: concrete_value,
                include_in_lower: !include_in_lower,
            });
        }
    }
    out
}

fn can_split_numeric_var(
    partitions: &NumericPartitions,
    numeric_var_id: usize,
    value: f64,
    include_in_lower: bool,
) -> bool {
    let Some(parts) = partitions.partitions(numeric_var_id) else {
        return false;
    };
    let Some(part_id) = parts.iter().position(|iv| iv.contains(value)) else {
        return false;
    };
    parts[part_id].can_split_at(value, include_in_lower)
}
