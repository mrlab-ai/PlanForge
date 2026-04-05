use std::collections::BTreeSet;

use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericType};

use super::projected_task::Pattern;

pub const DEFAULT_MAX_PDB_STATES: usize = 100_000;
const FIXED_NUMERIC_DOMAIN_SIZE: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GreedyPatternGeneratorConfig {
    pub max_pdb_states: usize,
}

impl Default for GreedyPatternGeneratorConfig {
    fn default() -> Self {
        Self {
            max_pdb_states: DEFAULT_MAX_PDB_STATES,
        }
    }
}

pub fn generate_greedy_pattern(
    task: &dyn AbstractNumericTask,
    config: GreedyPatternGeneratorConfig,
) -> Pattern {
    let (goal_regular, goal_numeric, true_goal_regular) = collect_goal_variables(task);

    let mut pattern = Pattern {
        regular: Vec::new(),
        numeric: Vec::new(),
    };
    let mut size = 1usize;

    for var_id in 0..task.variables().len() {
        if !goal_regular.contains(&var_id) {
            continue;
        }
        let domain_size = task.get_variable_domain_size(var_id).unwrap_or(1).max(1) as usize;
        if size.saturating_mul(domain_size) > config.max_pdb_states {
            return pattern;
        }
        pattern.regular.push(var_id);
        size *= domain_size;
    }

    if pattern.regular.is_empty() {
        for &var_id in &true_goal_regular {
            let domain_size = task.get_variable_domain_size(var_id).unwrap_or(1).max(1) as usize;
            if size.saturating_mul(domain_size) > config.max_pdb_states {
                break;
            }
            pattern.regular.push(var_id);
            size *= domain_size;
            break;
        }
    }

    for numeric_var_id in 0..task.numeric_variables().len() {
        if !goal_numeric.contains(&numeric_var_id) {
            continue;
        }
        if task.numeric_variables()[numeric_var_id].get_type() != &NumericType::Regular {
            continue;
        }
        if size.saturating_mul(FIXED_NUMERIC_DOMAIN_SIZE) > config.max_pdb_states {
            return pattern;
        }
        pattern.numeric.push(numeric_var_id);
        size *= FIXED_NUMERIC_DOMAIN_SIZE;
    }

    for var_id in 0..task.variables().len() {
        if goal_regular.contains(&var_id)
            || task
                .get_variable_axiom_layer(var_id)
                .is_ok_and(|x| x.is_some())
        {
            continue;
        }
        let domain_size = task.get_variable_domain_size(var_id).unwrap_or(1).max(1);
        if size.saturating_mul(domain_size) > config.max_pdb_states {
            break;
        }
        pattern.regular.push(var_id);
        size *= domain_size;
    }

    for numeric_var_id in 0..task.numeric_variables().len() {
        if goal_numeric.contains(&numeric_var_id)
            || task.numeric_variables()[numeric_var_id].get_type() != &NumericType::Regular
        {
            continue;
        }
        if size.saturating_mul(FIXED_NUMERIC_DOMAIN_SIZE) > config.max_pdb_states {
            break;
        }
        pattern.numeric.push(numeric_var_id);
        size *= FIXED_NUMERIC_DOMAIN_SIZE;
    }

    pattern
}

fn collect_goal_variables(
    task: &dyn AbstractNumericTask,
) -> (BTreeSet<usize>, BTreeSet<usize>, BTreeSet<usize>) {
    let mut regular = BTreeSet::new();
    let mut numeric = BTreeSet::new();
    let mut true_goal_regular = collect_goal_related_propositional_vars(task);

    for goal_index in 0..task.get_num_goals() {
        let goal = task.get_goal_fact(goal_index);
        let goal_var_id = goal.var;
        if task
            .get_variable_axiom_layer(goal_var_id)
            .unwrap_or(None)
            .is_none()
        {
            regular.insert(goal_var_id);
            true_goal_regular.insert(goal_var_id);
        }

        for comparison_axiom in task.comparison_axioms() {
            if usize::try_from(comparison_axiom.get_affected_var_id()).ok() != Some(goal_var_id) {
                continue;
            }

            for numeric_var_id in [
                comparison_axiom.get_left_var_id(),
                comparison_axiom.get_right_var_id(),
            ] {
                if task
                    .numeric_variables()
                    .get(numeric_var_id)
                    .map(|numeric_var| numeric_var.get_type() == &NumericType::Regular)
                    .unwrap_or(false)
                {
                    numeric.insert(numeric_var_id);
                }
            }
        }
    }

    (regular, numeric, true_goal_regular)
}

fn collect_goal_related_propositional_vars(task: &dyn AbstractNumericTask) -> BTreeSet<usize> {
    let mut goal_related: BTreeSet<usize> = (0..task.get_num_goals())
        .filter_map(|goal_id| usize::try_from(task.get_goal_fact(goal_id).var).ok())
        .collect();

    loop {
        let mut changed = false;

        for axiom in task.axioms() {
            let affected_var_id = axiom.var_id() as usize;
            if goal_related.contains(&affected_var_id) {
                for condition in axiom.conditions() {
                    changed |= goal_related.insert(condition.var);
                }
            }
        }

        if !changed {
            break;
        }
    }

    goal_related.retain(|&var_id| {
        task.get_variable_axiom_layer(var_id)
            .unwrap_or(None)
            .is_none()
    });
    goal_related
}

#[cfg(test)]
mod tests {
    use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator, PropositionalAxiom};
    use planners_sas::numeric::numeric_task::{
        AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitVariable, Metric,
        NumericRootTask, NumericType, NumericVariable, Operator,
    };

    use super::*;

    fn simple_var(name: &str, axiom_layer: Option<usize>) -> ExplicitVariable {
        ExplicitVariable::new(
            2,
            name.to_string(),
            vec![format!("{name}=0"), format!("{name}=1")],
            axiom_layer,
            1,
        )
    }

    fn sample_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, None),
            vec![
                simple_var("p", None),
                ExplicitVariable::new(
                    3,
                    "cmp".to_string(),
                    vec!["t".to_string(), "f".to_string(), "u".to_string()],
                    Some(0),
                    2,
                ),
            ],
            vec![
                NumericVariable::new("c".to_string(), NumericType::Constant, None),
                NumericVariable::new("x".to_string(), NumericType::Regular, None),
            ],
            vec![ExplicitFact::new(1, 0)],
            vec![],
            vec![0, 2],
            vec![1.0, 0.0],
            vec![Operator::new(
                "inc".to_string(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    1,
                    AssignmentOperation::Plus,
                    0,
                    false,
                    vec![],
                )],
                1,
            )],
            vec![PropositionalAxiom::new(
                vec![ExplicitFact::new(1, 0)],
                0,
                1,
                0,
            )],
            vec![ComparisonAxiom::new(
                1,
                1,
                0,
                ComparisonOperator::GreaterThanOrEqual,
            )],
            vec![],
            ExplicitFact::new(0, 0),
        )
    }

    #[test]
    fn greedy_pattern_prefers_goal_variables() {
        let task = sample_task();
        let pattern = generate_greedy_pattern(&task, GreedyPatternGeneratorConfig::default());

        assert!(pattern.numeric.contains(&1));
    }

    #[test]
    fn greedy_pattern_includes_true_goal_support_var() {
        let task = NumericRootTask::new(
            1,
            Metric::new(true, None),
            vec![
                simple_var("support", None),
                ExplicitVariable::new(
                    2,
                    "goal".to_string(),
                    vec!["off".to_string(), "on".to_string()],
                    Some(1),
                    0,
                ),
            ],
            vec![],
            vec![ExplicitFact::new(1, 1)],
            vec![],
            vec![0, 0],
            vec![],
            vec![],
            vec![PropositionalAxiom::new(
                vec![ExplicitFact::new(0, 1)],
                1,
                0,
                1,
            )],
            vec![],
            vec![],
            ExplicitFact::new(0, 0),
        );

        let pattern = generate_greedy_pattern(&task, GreedyPatternGeneratorConfig::default());

        assert!(pattern.regular.contains(&0));
    }
}
