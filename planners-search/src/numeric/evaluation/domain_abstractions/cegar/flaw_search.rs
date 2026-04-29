pub mod flaw_selection;
pub mod progression;
pub mod regression;
pub mod sequence;
pub mod state;
#[cfg(test)]
mod tests;

use anyhow::Result;
use std::fmt;

use planners_sas::numeric::numeric_task::{AbstractNumericTask, ExplicitFact};

use serde::{Deserialize, Serialize};

use super::determine_include_in_lower;
use crate::numeric::evaluation::domain_abstractions::abstract_operator_generator::DomainMapping;
use crate::numeric::evaluation::domain_abstractions::cegar::determine_include_in_lower_for_flaw_search_state;
use crate::numeric::evaluation::domain_abstractions::cegar::flaw_search::progression::get_progression_flaws;
use crate::numeric::evaluation::domain_abstractions::cegar::flaw_search::regression::get_regression_flaws;
use crate::numeric::evaluation::domain_abstractions::cegar::flaw_search::sequence::{
    SequenceDirection, get_sequence_flaws,
};
use crate::numeric::evaluation::domain_abstractions::cegar::flaw_search::state::FlawSearchState;
use crate::numeric::evaluation::domain_abstractions::domain_abstraction::{
    ComparisonAxiomIndex, NumericPartitions,
};
use crate::numeric::evaluation::domain_abstractions::domain_abstraction_factory::WildcardPlanResult;
use crate::numeric::evaluation::domain_abstractions::utils::partition_for_value;

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
pub enum FlawKind {
    Progression,
    Regression,
    SequenceProgression,
    SequenceRegression,
    SequenceBidirectional,
}

impl fmt::Display for FlawKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Progression => write!(f, "progression"),
            Self::Regression => write!(f, "regression"),
            Self::SequenceProgression => write!(f, "sequence_progression"),
            Self::SequenceRegression => write!(f, "sequence_regression"),
            Self::SequenceBidirectional => write!(f, "sequence_bidirectional"),
        }
    }
}

impl FlawKind {
    pub fn get_flaws(
        &self,
        task: &dyn AbstractNumericTask,
        partitions: &NumericPartitions,
        domain_mapping: &DomainMapping,
        wildcard_plan: &WildcardPlanResult,
    ) -> Result<Vec<Flaw>> {
        match self {
            Self::Progression => get_progression_flaws(task, partitions, wildcard_plan),
            Self::Regression => {
                let mut flaws = get_regression_flaws(task, domain_mapping, wildcard_plan);
                // Progression flaw fallback if no regression flaw is found
                // (numeric deviation flaws not detected).
                if let Ok(ref flaws_ok) = flaws
                    && flaws_ok.is_empty()
                {
                    flaws = get_progression_flaws(task, partitions, wildcard_plan);
                }

                flaws
            }
            Self::SequenceProgression => get_sequence_flaws(
                task,
                partitions,
                domain_mapping,
                wildcard_plan,
                SequenceDirection::Progression,
            ),
            // Sequence progression flaw fallback already searched here.
            Self::SequenceRegression => get_sequence_flaws(
                task,
                partitions,
                domain_mapping,
                wildcard_plan,
                SequenceDirection::Regression,
            ),
            Self::SequenceBidirectional => get_sequence_flaws(
                task,
                partitions,
                domain_mapping,
                wildcard_plan,
                SequenceDirection::Bidirectional,
            ),
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
    domain_mapping: &DomainMapping,
    wildcard_plan: &WildcardPlanResult,
    flaw_kind: FlawKind,
) -> Result<Vec<Flaw>> {
    flaw_kind.get_flaws(task, partitions, domain_mapping, wildcard_plan)
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

fn dependent_numeric_flaws_in_interval_for_comparison_prop_var(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    prop_var_id: usize,
    state: &FlawSearchState,
) -> Vec<NumericFlaw> {
    let Some(tree) = comparison_index.comparison_tree(prop_var_id) else {
        return vec![];
    };

    let mut out: Vec<NumericFlaw> = Vec::new();
    for dep_var_id in tree.regular_numeric_var_dependencies(task) {
        let include_in_lower = determine_include_in_lower_for_flaw_search_state(tree, state);

        if can_split_numeric_var(
            partitions,
            dep_var_id,
            state.numeric[dep_var_id].upper,
            include_in_lower,
        ) {
            out.push(NumericFlaw {
                numeric_var_id: dep_var_id,
                value: state.numeric[dep_var_id].upper,
                include_in_lower,
            });
        } else if can_split_numeric_var(
            partitions,
            dep_var_id,
            state.numeric[dep_var_id].lower,
            !include_in_lower,
        ) {
            out.push(NumericFlaw {
                numeric_var_id: dep_var_id,
                value: state.numeric[dep_var_id].lower,
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
    let Some(part_id) = partition_for_value(parts, value) else {
        return false;
    };
    parts[part_id].can_split_at(value, include_in_lower)
}
