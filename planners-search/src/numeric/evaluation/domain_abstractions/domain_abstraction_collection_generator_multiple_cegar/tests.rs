use super::*;
use crate::numeric::evaluation::domain_abstractions::max_domain_abstraction_heuristic::MaxDomainAbstractionHeuristic;
use crate::numeric::evaluation::evaluator::EvaluationState;
use crate::numeric::evaluation::heuristic::Heuristic;
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_task::NumericRootTask;
use planners_sas::numeric::state_registry::StateRegistry;
use planners_sas::numeric::utils::int_packer::IntDoublePacker;

#[test]
fn single_init_split_selection_uses_round_robin_iteration_order() {
    let candidates = [0usize, 1, 2, 3, 4];
    let selected = (1..=8)
        .map(|iteration| select_single_init_split_var(&candidates, iteration).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(selected, vec![1, 2, 3, 4, 0, 1, 2, 3]);
}

#[test]
fn single_init_split_selection_handles_empty_candidates() {
    assert_eq!(select_single_init_split_var(&[], 1), None);
}

#[test]
#[ignore = "requires the local pfile2.sas regression input"]
fn pfile2_multi_domain_abstractions_initial_heuristic_is_finite() {
    let task_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../pfile2.sas");
    assert!(
        task_path.exists(),
        "expected local regression input at {}",
        task_path.display()
    );
    let task = NumericRootTask::from_file(&task_path);

    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        max_abstraction_size: 100_000,
        max_collection_size: 1_000_000,
        total_max_time: 150.0,
        debug: true,
        random_seed: 0,
        ..Default::default()
    };
    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(config);
    let abstractions = generator.generate_collection(&task).unwrap();

    let heuristic = MaxDomainAbstractionHeuristic::new(None, abstractions);
    let packer = IntDoublePacker::from_task(&task);
    let axiom_evaluator = AxiomEvaluator::new(&task, &packer);
    let mut registry = StateRegistry::new(&task, &packer, &axiom_evaluator);
    let initial_state = registry.get_initial_state();
    let eval_state =
        EvaluationState::new_with_registry(&initial_state, 0.0, false, &task, &registry);
    let initial_h = heuristic.compute_heuristic(&eval_state).unwrap();

    assert!(
        initial_h.is_finite(),
        "multi_domain_abstractions initial heuristic should be finite for pfile2"
    );
}

#[test]
#[ignore = "requires the local pfile2.sas regression input"]
fn pfile2_collection_inf_abstraction_reduced_case() {
    let task_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../pfile2.sas");
    assert!(
        task_path.exists(),
        "expected local regression input at {}",
        task_path.display()
    );
    let task = NumericRootTask::from_file(&task_path);
    let sas = std::fs::read_to_string(&task_path).unwrap();
    assert!(sas.contains(
        "begin_variable\nvar9\n-1\n2\nAtom clear(pallet2)\nNegatedAtom clear(pallet2)\nend_variable"
    ));
    assert_eq!(task.get_variable_name(13).unwrap(), "var9");
    assert_eq!(task.get_variable_name(0).unwrap(), "var20");

    let goal = ExplicitFact::new(25, 10);
    let single_goal_task = SingleGoalTask::new(&task, goal.clone());
    let config = CegarConfig {
        max_abstraction_size: 100_000,
        debug: true,
        random_seed: Some(11_890_779_981_456_599_205),
        init_split_method: InitSplitMethod::InitValue,
        init_split_var_ids: Some(HashSet::from([13])),
        ..Default::default()
    };

    let outcome =
        crate::numeric::evaluation::domain_abstractions::cegar::Cegar::new(config.clone())
            .unwrap()
            .build_abstraction(&single_goal_task)
            .unwrap();
    assert!(
        outcome.last_step.wildcard_plan.is_some(),
        "the reduced pfile2 abstraction should have an abstract plan"
    );
    let distance_table = outcome
        .final_state
        .factory
        .build_abstract_distance_table(&single_goal_task, config.combine_labels, false)
        .unwrap();
    let initial_h = distance_table.distances[distance_table.initial_state_hash];
    assert!(
        initial_h.is_finite(),
        "reduced pfile2 collection abstraction should not make the initial abstract state a dead end"
    );
}
