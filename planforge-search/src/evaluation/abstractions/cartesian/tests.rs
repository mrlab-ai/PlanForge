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
    CartesianAbstractionConfig, CartesianAbstractionGenerator, CartesianRefinementDirection,
    CartesianSemantics, CartesianSet, CartesianSplitSelection, CartesianStopReason, FlawKind,
    OperatorBitSet, RefinementNode, ShortestPaths, Split, SplitDimension, TransitionKey,
    WorkingAbstraction, numeric_split_choice_key, push_unique_split, retain_min_growth_splits,
    select_refinement_split,
};
use crate::evaluation::abstractions::collections::portfolio::CollectionStrategy;
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
    for flaw_kind in [FlawKind::Progression, FlawKind::ExecuteEntirePlan] {
        let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
            max_states: 64,
            flaw_kind,
            ..Default::default()
        })
        .unwrap()
        .generate(task)
        .unwrap();
        assert_eq!(
            abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash],
            expected_h,
            "unexpected initial h for {flaw_kind}"
        );
        assert!(
            abstraction.metadata.solved_by_self,
            "{flaw_kind} failed to produce a concrete plan"
        );
        assert_eq!(
            abstraction.metadata.stop_reason,
            CartesianStopReason::ConcretePlan
        );
        assert_eq!(
            abstraction.abstract_source_sets.len(),
            abstraction.transition_system.transitions.len()
        );
    }
}
