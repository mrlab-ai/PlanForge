use planforge_sas::axioms::PropositionalAxiom;
use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, NumericRootTask,
    NumericRootTaskParts, Operator, OperatorCost, VariableIndex,
};

use super::Transcription;
use crate::residuals::{Assignment, evaluate};

fn duplicate_precondition_task(duplicate: bool) -> NumericRootTask {
    let variables = vec![
        ExplicitVariable::new(
            2,
            "global".to_string(),
            vec!["holds".to_string(), "default".to_string()],
            Some(0),
            ExplicitValueIndex::new(1),
        ),
        ExplicitVariable::new(
            2,
            "left".to_string(),
            vec!["left-0".to_string(), "left-1".to_string()],
            None,
            ExplicitValueIndex::new(0),
        ),
        ExplicitVariable::new(
            2,
            "right".to_string(),
            vec!["right-0".to_string(), "right-1".to_string()],
            None,
            ExplicitValueIndex::new(0),
        ),
    ];
    let left_zero = ExplicitFact::propositional(1, 0);
    let right_zero = ExplicitFact::propositional(2, 0);
    let preconditions = if duplicate {
        vec![left_zero, right_zero, left_zero]
    } else {
        vec![left_zero, right_zero]
    };
    let operator = Operator::new(
        "set-left".to_string(),
        preconditions,
        vec![Effect::new(
            Vec::new(),
            VariableIndex::from_usize(1),
            Some(ExplicitValueIndex::new(0)),
            ExplicitValueIndex::new(1),
        )],
        Vec::new(),
        OperatorCost::new(1),
    );
    NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables: Vec::new(),
        goals: vec![ExplicitFact::propositional(1, 1)],
        mutexes: Vec::new(),
        state: vec![
            ExplicitValueIndex::new(1),
            ExplicitValueIndex::new(0),
            ExplicitValueIndex::new(0),
        ],
        numeric_state: Vec::new(),
        operators: vec![operator],
        axioms: vec![PropositionalAxiom::new(
            Vec::new(),
            VariableIndex::from_usize(0),
            ExplicitValueIndex::new(1),
            ExplicitValueIndex::new(0),
        )],
        comparison_axioms: Vec::new(),
        assignment_axioms: Vec::new(),
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

fn integral_residuals(
    transcription: &Transcription,
    current: &[usize],
    next: &[usize],
) -> crate::residuals::Residuals {
    let mut assignment = Assignment::zeros(transcription, 1);
    assignment.set_action_one_hot(0, 0);
    assignment.set_state_one_hot(transcription, 0, current);
    assignment.set_state_one_hot(transcription, 1, next);
    evaluate(transcription, &assignment)
}

#[test]
fn duplicate_preconditions_are_canonicalized_without_changing_core_semantics() {
    let duplicate =
        Transcription::build(&duplicate_precondition_task(true)).expect("transcription");
    let canonical =
        Transcription::build(&duplicate_precondition_task(false)).expect("transcription");
    let left_zero = duplicate.fact(0, 0);
    let right_zero = duplicate.fact(1, 0);

    assert_eq!(duplicate.pre_action(), &[0, 0]);
    assert_eq!(duplicate.pre_fact(), &[left_zero, right_zero]);
    assert_eq!(duplicate.pre_action(), canonical.pre_action());
    assert_eq!(duplicate.pre_fact(), canonical.pre_fact());

    let valid_duplicate = integral_residuals(&duplicate, &[0, 0], &[1, 0]);
    let valid_canonical = integral_residuals(&canonical, &[0, 0], &[1, 0]);
    assert_eq!(valid_duplicate, valid_canonical);
    assert!(valid_duplicate.is_zero(1e-12));

    let invalid_duplicate = integral_residuals(&duplicate, &[1, 0], &[1, 0]);
    let invalid_canonical = integral_residuals(&canonical, &[1, 0], &[1, 0]);
    assert_eq!(invalid_duplicate, invalid_canonical);
    assert_eq!(invalid_duplicate.precondition, vec![1.0, 0.0]);
}

#[cfg(feature = "candle")]
#[test]
fn duplicate_precondition_does_not_create_more_than_unit_causal_demand() {
    use candle_core::{Device, Tensor};

    use crate::tensor::{DTYPE, TensorPlan};

    let transcription =
        Transcription::build(&duplicate_precondition_task(true)).expect("transcription");
    let device = Device::Cpu;
    let plan = TensorPlan::new(&transcription, 1, 1, device.clone()).expect("tensor plan");
    let action_logits =
        Tensor::from_vec(vec![30.0, -30.0], (1, 1, 2), &device).expect("action logits");
    let mut state_values = vec![-30.0; transcription.num_facts()];
    state_values[transcription.fact(0, 1) as usize] = 30.0;
    state_values[transcription.fact(1, 0) as usize] = 30.0;
    let state_logits = Tensor::from_vec(state_values, (1, 1, transcription.num_facts()), &device)
        .expect("state logits");
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
    let forward = plan
        .forward(&action_logits, &state_logits, &temperature, &temperature)
        .expect("forward");
    let link_shape = plan.causal_link_shape();
    let link_logits = Tensor::zeros(&link_shape, DTYPE, &device).expect("link logits");
    let link_temperature = Tensor::ones((1, 1, 1, 1), DTYPE, &device).expect("link temperature");
    let links = plan
        .causal_link_forward(&forward, &link_logits, &link_temperature)
        .expect("causal links");

    let action_probability = forward
        .action
        .to_vec3::<f64>()
        .expect("action distribution")[0][0][0];
    let demand = links.demand.to_vec3::<f64>().expect("causal demand");
    for fact in [transcription.fact(0, 0), transcription.fact(1, 0)] {
        let fact_demand = demand[0][0][fact as usize];
        assert!(
            (fact_demand - action_probability).abs() < 1e-12,
            "fact {fact} has demand {fact_demand}, expected one action mass {action_probability}"
        );
        assert!(fact_demand <= 1.0);
    }
}
