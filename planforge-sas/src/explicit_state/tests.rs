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
    ExplicitState {
        propositional: state.get_state(registry),
        numeric: registry
            .get_numeric_vars(state)
            .expect("a registered state has numeric values"),
    }
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
