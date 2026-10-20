use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use planforge_sas::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitValueIndex,
    ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts, NumericType, NumericValue,
    NumericVariable, ONE_VALUE, Operator, OperatorCost, OperatorIndex, VariableIndex,
};

use super::{
    CartesianAbstractionCollectionConfig, CartesianAbstractionCollectionGenerator,
    CartesianAbstractionConfig, CartesianAbstractionGenerator, CartesianSemantics, CartesianSet,
    CartesianView, FlawKind, FlawScope, OperatorBitSet, RefinementLimits, RefinementNode,
    ShortestPaths, Split, SplitDimension, TransitionKey, WorkingAbstraction,
};
use crate::evaluation::abstractions::collections::strategy::CollectionStrategy;
use crate::evaluation::abstractions::common::cegar::CegarStopReason;
use crate::evaluation::abstractions::common::cegar::selection::{
    push_unique_split, retain_min_growth_splits, select_refinement_split,
};
use crate::evaluation::abstractions::common::cegar::semantics::numeric_split_choice_key;
use crate::evaluation::abstractions::common::cegar::{SplitKind, SplitSelection};
use planforge_sas::utils::interval::Interval;

mod collection;
mod generator;
mod regression;
mod semantics;
mod split_selection;
mod working;

fn comparison_variable(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        ConditionValue::DOMAIN_SIZE,
        name.into(),
        vec!["true".into(), "false".into()],
        Some(0),
        ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
    )
}

fn numeric_goal_task(
    variables: Vec<ExplicitVariable>,
    numeric_variables: Vec<NumericVariable>,
    initial_numeric: Vec<NumericValue>,
    operators: Vec<Operator>,
    comparison_axioms: Vec<ComparisonAxiom>,
    assignment_axioms: Vec<AssignmentAxiom>,
) -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(2)],
        numeric_state: initial_numeric,
        operators,
        axioms: vec![],
        comparison_axioms,
        assignment_axioms,
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

fn assert_solved_with_h(task: &NumericRootTask, expected_h: f64) {
    for flaw_scope in [FlawScope::First, FlawScope::WholePlan] {
        let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
            cegar: CegarOptions {
                max_states: 64,
                flaw_kind: FlawKind::Progression,
                flaw_scope,
                ..CartesianAbstractionConfig::default().cegar
            },
            ..Default::default()
        })
        .unwrap()
        .generate(task)
        .unwrap();
        assert_eq!(
            abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash],
            expected_h,
            "unexpected initial h for {flaw_scope}"
        );
        assert!(
            abstraction.metadata.solved_by_self(),
            "{flaw_scope} progression failed to produce a concrete plan"
        );
        assert_eq!(
            abstraction.metadata.stop_reason,
            CegarStopReason::ConcretePlan
        );
        assert_eq!(
            abstraction.abstract_source_sets.len(),
            abstraction.transition_system.transitions.len()
        );
    }
}

/// Shortest paths that make every state a goal: for tests that read only the
/// regions and growth of a view.
fn all_goal_paths(states: usize) -> ShortestPaths {
    ShortestPaths::for_test(
        vec![0.0; states],
        vec![None; states],
        vec![Vec::new(); states],
        vec![None; states],
        vec![true; states],
    )
}
