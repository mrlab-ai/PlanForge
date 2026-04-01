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
    let (goal_regular, goal_numeric) = collect_goal_variables(task);

    let mut pattern = Pattern {
        regular: Vec::new(),
        numeric: Vec::new(),
    };
    let mut size = 1usize;

    for var_id in 0..task.variables().len() {
        if !goal_regular.contains(&var_id) {
            continue;
        }
        let domain_size = task
            .get_variable_domain_size(var_id as i32)
            .unwrap_or(1)
            .max(1) as usize;
        if size.saturating_mul(domain_size) > config.max_pdb_states {
            return pattern;
        }
        pattern.regular.push(var_id);
        size *= domain_size;
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
            || task.get_variable_axiom_layer(var_id as i32).unwrap_or(-1) != -1
        {
            continue;
        }
        let domain_size = task
            .get_variable_domain_size(var_id as i32)
            .unwrap_or(1)
            .max(1) as usize;
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

fn collect_goal_variables(task: &dyn AbstractNumericTask) -> (BTreeSet<usize>, BTreeSet<usize>) {
    let mut regular = BTreeSet::new();
    let mut numeric = BTreeSet::new();

    for goal_index in 0..usize::try_from(task.get_num_goals().max(0)).unwrap_or(0) {
        let goal = task.get_goal_fact(goal_index as i32);
        let goal_var_id = goal.var() as usize;
        if task
            .get_variable_axiom_layer(goal_var_id as i32)
            .unwrap_or(-1)
            == -1
        {
            regular.insert(goal_var_id);
        }

        for comparison_axiom in task.comparison_axioms() {
            if usize::try_from(comparison_axiom.get_affected_var_id()).ok() != Some(goal_var_id) {
                continue;
            }

            for numeric_var_id in [
                comparison_axiom.get_left_var_id(),
                comparison_axiom.get_right_var_id(),
            ] {
                let Ok(numeric_var_id) = usize::try_from(numeric_var_id) else {
                    continue;
                };
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

    (regular, numeric)
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

    fn sample_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![
                simple_var("p", -1),
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
            vec![Fact::new(1, 0)],
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
            vec![PropositionalAxiom::new(vec![Fact::new(1, 0)], 0, 1, 0)],
            vec![ComparisonAxiom::new(
                1,
                1,
                0,
                ComparisonOperator::GreaterThanOrEqual,
            )],
            vec![],
            (0, 0),
        )
    }

    #[test]
    fn greedy_pattern_prefers_goal_variables() {
        let task = sample_task();
        let pattern = generate_greedy_pattern(&task, GreedyPatternGeneratorConfig::default());

        assert!(pattern.numeric.contains(&1));
    }
}
