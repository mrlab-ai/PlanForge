use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, NumericRootTask,
    NumericRootTaskParts, Operator, OperatorCost, VariableIndex,
};

use super::*;
use crate::evaluation::abstractions::pdb::pattern_database::PdbHeuristicConfig;

fn binary_var(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        None,
        ExplicitValueIndex::new(1),
    )
}

/// `open` sets the goal `door` and needs the `key`, which `fetch` gets.
fn key_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![binary_var("key"), binary_var("door")],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(1, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![
            Operator::new(
                "fetch".to_string(),
                vec![ExplicitFact::propositional(0, 0)],
                vec![Effect::new(
                    vec![],
                    VariableIndex::new(0),
                    Some(ExplicitValueIndex::new(0)),
                    ExplicitValueIndex::new(1),
                )],
                vec![],
                OperatorCost::new(1),
            ),
            Operator::new(
                "open".to_string(),
                vec![ExplicitFact::propositional(0, 1)],
                vec![Effect::new(
                    vec![],
                    VariableIndex::new(1),
                    Some(ExplicitValueIndex::new(0)),
                    ExplicitValueIndex::new(1),
                )],
                vec![],
                OperatorCost::new(1),
            ),
        ],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

fn initial_h(task: &NumericRootTask, pdbs: &[PatternDatabase<'_>]) -> f64 {
    let initial = [ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)];
    assert_eq!(task.variables().len(), initial.len());
    pdbs.iter().map(|pdb| pdb.value(&initial, &[])).sum()
}

#[test]
fn hill_climbing_adds_the_precondition_variable_of_the_goal() {
    let task = key_task();
    let config = HillClimbingConfig {
        num_samples: 200,
        min_improvement: 1,
        ..HillClimbingConfig::default()
    };
    let pdbs =
        generate_hill_climbing_pdbs(&task, config, PdbHeuristicConfig::default(), false, None)
            .unwrap();

    assert_eq!(pdbs.len(), 2);
    // {door} alone gives 1; {door, key} gives the optimal 2 on its three
    // reachable states, and the two share `open`, so the canonical heuristic
    // takes their maximum.
    assert_eq!(pdbs[1].num_states(), 3);
    assert_eq!(pdbs[1].value(&[ExplicitValueIndex::new(0); 2], &[]), 2.0);
    assert_eq!(initial_h(&task, &pdbs[..1]), 1.0);
}

#[test]
fn hill_climbing_rejects_candidates_beyond_max_pdb_size() {
    let task = key_task();
    let config = HillClimbingConfig {
        num_samples: 200,
        min_improvement: 1,
        max_pdb_size: 2,
        ..HillClimbingConfig::default()
    };
    let pdbs =
        generate_hill_climbing_pdbs(&task, config, PdbHeuristicConfig::default(), false, None)
            .unwrap();

    assert_eq!(pdbs.len(), 1);
}

#[test]
fn candidate_raises_the_heuristic_through_an_additive_subset() {
    let values = [2.0, 3.0];
    let subsets = vec![vec![0], vec![1]];

    assert!(raises_canonical_value(2.0, &values, 4.0, &subsets));
    assert!(!raises_canonical_value(1.0, &values, 4.0, &subsets));
    assert!(raises_canonical_value(
        f64::INFINITY,
        &values,
        4.0,
        &subsets
    ));
    assert!(!raises_canonical_value(
        9.0,
        &values,
        f64::INFINITY,
        &subsets
    ));
}

#[test]
fn min_improvement_beyond_num_samples_is_rejected() {
    let config = HillClimbingConfig {
        num_samples: 5,
        min_improvement: 6,
        ..HillClimbingConfig::default()
    };

    assert!(config.validate().is_err());
}
