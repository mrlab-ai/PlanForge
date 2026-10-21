use std::sync::Arc;

use super::*;
use crate::numeric_task::TaskRef;
use crate::state_registry::StateRegistry;
use crate::tests::get_root_task;

/// The registry's values of the state its handle names.
fn registry_values(
    registry: &StateRegistry<'_>,
    state: &crate::state_registry::ConcreteState,
) -> ExplicitState {
    let mut values = ExplicitState::default();
    registry
        .view(state)
        .fill_explicit(&mut values)
        .expect("a registered state has numeric values");
    values
}

#[test]
fn initial_state_and_successor_agree_with_the_registry() {
    let task: TaskRef = Arc::new(get_root_task());
    let space = ExplicitStateSpace::new(&*task);
    let mut registry = StateRegistry::for_task(task.clone());
    let registered_initial = registry.get_initial_state();
    let initial = space.initial_state().unwrap();
    assert_eq!(initial, registry_values(&registry, &registered_initial));

    let operator = &task.get_operators()[0];
    assert!(initial.is_applicable(operator));
    let registered_successor = registry
        .get_successor_state(&registered_initial, operator)
        .unwrap();
    assert_eq!(
        space.successor(&initial, operator).unwrap(),
        registry_values(&registry, &registered_successor)
    );
    assert_eq!(
        space.replay(&initial, &[OperatorIndex::new(0)]).unwrap(),
        registry_values(&registry, &registered_successor)
    );
}

#[test]
#[should_panic(expected = "wrong number of propositional values")]
fn a_state_of_another_layout_is_rejected() {
    let task = get_root_task();
    let space = ExplicitStateSpace::new(&task);
    let mut state = space.initial_state().unwrap();
    state.propositional.pop();
    let _ = space.successor(&state, &task.get_operators()[0]);
}

#[test]
fn unsatisfied_lists_the_failed_facts_in_order() {
    let state = ExplicitState {
        propositional: vec![ExplicitValueIndex::new(1), ExplicitValueIndex::new(0)],
        numeric: Vec::new(),
    };
    let facts = [
        ExplicitFact::propositional(0, 0),
        ExplicitFact::propositional(1, 0),
        ExplicitFact::propositional(1, 1),
    ];
    assert!(facts[1].holds_in(&state.propositional));
    assert!(!state.holds_all(&facts));
    assert!(state.holds_all(&facts[1..2]));
    assert_eq!(
        state.unsatisfied(&facts).collect::<Vec<_>>(),
        [&facts[0], &facts[2]]
    );
}
