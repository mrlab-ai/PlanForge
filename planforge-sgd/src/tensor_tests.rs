//! The tensor layer, checked against the `f64` reference and against finite
//! differences.
//!
//! Two independent things are verified here, and both matter:
//!
//! * the forward pass agrees with [`crate::residuals`], which is the semantics
//!   the exactness proof and the exhaustive integrality test are about;
//! * [`crate::tensor::SegSum`]'s hand-written backward pass agrees with finite
//!   differences, since an autograd bug there would silently corrupt every
//!   gradient without changing any forward value.

#![cfg(all(test, feature = "candle"))]

use candle_core::{DType, Device, Tensor};
use planforge_sas::axioms::PropositionalAxiom;
use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, Operator,
};

use crate::residuals::{Assignment, evaluate};
use crate::tensor::{
    CausalLinkInput, DTYPE, Forward, SegProd, SegSum, TensorPlan, bottleneck_norm_per_particle,
};
use crate::testing::{Rng, random_task};
use crate::transcription::Transcription;

fn cpu() -> Device {
    Device::Cpu
}

/// Read a `[M, H, K]` tensor into a nested vector.
fn to_vec3(tensor: &Tensor) -> Vec<Vec<Vec<f64>>> {
    tensor
        .to_dtype(DType::F64)
        .expect("dtype conversion")
        .to_vec3::<f64>()
        .expect("tensor is rank 3")
}

/// Read a `[M, H + 1, F, H + 1]` tensor into a nested vector.
fn to_vec4(tensor: &Tensor) -> Vec<Vec<Vec<Vec<f64>>>> {
    let dims = tensor.dims();
    assert_eq!(dims.len(), 4, "tensor is rank 4");
    let flat = tensor
        .to_dtype(DType::F64)
        .expect("dtype conversion")
        .flatten_all()
        .expect("flatten rank-4 tensor")
        .to_vec1::<f64>()
        .expect("flattened values");
    let mut values = vec![vec![vec![vec![0.0; dims[3]]; dims[2]]; dims[1]]; dims[0]];
    let mut index = 0;
    for particle in &mut values {
        for consumer in particle {
            for fact in consumer {
                fact.copy_from_slice(&flat[index..index + dims[3]]);
                index += dims[3];
            }
        }
    }
    values
}

#[test]
fn seg_sum_forward_matches_a_direct_loop() {
    let device = cpu();
    let segments = vec![0u32, 2, 0, 1, 2, 2];
    let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 0.5, 0.25, 8.0, 2.0, 1.0];
    let input = Tensor::from_vec(values.clone(), (2, 1, 6), &device).expect("input");

    let output = input
        .apply_op1(SegSum::new(3, segments.clone()))
        .expect("seg-sum");
    assert_eq!(output.dims(), &[2, 1, 3]);

    let got = to_vec3(&output);
    for row in 0..2 {
        let mut expected = vec![0f64; 3];
        for (index, &segment) in segments.iter().enumerate() {
            expected[segment as usize] += values[row * 6 + index];
        }
        for segment in 0..3 {
            assert!(
                (got[row][0][segment] - expected[segment]).abs() < 1e-12,
                "row {row} segment {segment}: {} vs {}",
                got[row][0][segment],
                expected[segment]
            );
        }
    }
}

/// The backward pass is hand-written, so it gets a finite-difference check.
/// A wrong gradient here would not change any forward value and would be
/// invisible to every other test.
#[test]
fn seg_sum_backward_matches_finite_differences() {
    let device = cpu();
    let segments = vec![0u32, 1, 1, 2, 0];
    let base: Vec<f64> = vec![0.3, -1.2, 0.75, 2.0, 0.1];
    // A non-symmetric downstream function, so every segment gets a distinct
    // gradient and a sign error cannot cancel out.
    let weights = Tensor::from_vec(vec![1.0f64, -2.0, 3.5], (1, 1, 3), &device).expect("weights");

    let loss_of = |values: &[f64]| -> f64 {
        let input = Tensor::from_vec(values.to_vec(), (1, 1, 5), &device).expect("input");
        let summed = input
            .apply_op1(SegSum::new(3, segments.clone()))
            .expect("seg-sum");
        (summed * &weights)
            .expect("weighting")
            .sum_all()
            .expect("sum")
            .to_scalar::<f64>()
            .expect("scalar")
    };

    let variable = candle_core::Var::from_vec(base.clone(), (1, 1, 5), &device).expect("var");
    let summed = variable
        .as_tensor()
        .apply_op1(SegSum::new(3, segments.clone()))
        .expect("seg-sum");
    let loss = (summed * &weights)
        .expect("weighting")
        .sum_all()
        .expect("sum");
    let grads = loss.backward().expect("backward");
    let analytic = grads
        .get(&variable)
        .expect("gradient for the input")
        .to_vec3::<f64>()
        .expect("rank 3");

    let epsilon = 1e-6;
    for index in 0..base.len() {
        let mut plus = base.clone();
        let mut minus = base.clone();
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let numeric = (loss_of(&plus) - loss_of(&minus)) / (2.0 * epsilon);
        assert!(
            (analytic[0][0][index] - numeric).abs() < 1e-6,
            "gradient mismatch at {index}: analytic {} vs numeric {numeric}",
            analytic[0][0][index]
        );
    }
}

#[test]
fn seg_prod_preserves_empty_and_zero_products_and_their_derivatives() {
    let device = cpu();
    // Segment 0 has exactly one zero, segment 3 has two, and segment 4 is
    // empty. These are precisely the cases that product/output division gets
    // wrong.
    let segments = vec![0u32, 1, 0, 2, 1, 3, 3];
    let values = vec![0.0f64, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0];
    let variable = candle_core::Var::from_vec(values, (1, 1, 7), &device).expect("product input");
    let product = variable
        .as_tensor()
        .apply_op1(SegProd::new(5, segments))
        .expect("segmented product");
    assert_eq!(to_vec3(&product)[0][0], vec![0.0, 10.0, 4.0, 0.0, 1.0]);

    let weights = Tensor::from_vec(vec![7.0f64, 11.0, 13.0, 17.0, 19.0], (1, 1, 5), &device)
        .expect("weights");
    let loss = (product * weights)
        .expect("weighted products")
        .sum_all()
        .expect("loss");
    let gradients = loss.backward().expect("seg-prod backward");
    let got = to_vec3(
        gradients
            .get(&variable)
            .expect("gradient for segmented-product input"),
    );
    assert_eq!(got[0][0], vec![21.0, 55.0, 0.0, 13.0, 22.0, 0.0, 0.0]);
}

/// The forward pass must reproduce the `f64` reference on random fractional
/// assignments, not just at integrality — a relaxation bug that only shows up
/// between the corners is exactly what would derail optimization.
#[test]
fn tensor_forward_matches_the_f64_reference() {
    let device = cpu();
    let mut rng = Rng::new(0xA11CE_5EED);
    let mut compared = 0usize;

    let mut multi_write_compared = 0usize;
    let mut conditional_multi_write_compared = 0usize;
    for _ in 0..600 {
        let task = random_task(&mut rng);
        let Ok(transcription) = Transcription::build(&task) else {
            continue;
        };

        let horizon = 1 + rng.below(3);
        let particles = 1 + rng.below(2);
        let plan = TensorPlan::new(&transcription, horizon, particles, device.clone())
            .expect("valid transcription is supported");

        // Random logits, shared between the two implementations.
        let action_logits: Vec<f64> = (0..particles * horizon * transcription.num_actions())
            .map(|_| rng.below(2000) as f64 / 1000.0 - 1.0)
            .collect();
        let state_logits: Vec<f64> = (0..particles * horizon * transcription.num_facts())
            .map(|_| rng.below(2000) as f64 / 1000.0 - 1.0)
            .collect();

        let action_temperature =
            Tensor::from_vec(vec![0.7f64; particles], (particles, 1, 1), &device)
                .expect("action temperature");
        let state_temperature =
            Tensor::from_vec(vec![1.3f64; particles], (particles, 1, 1), &device)
                .expect("state temperature");

        let z = Tensor::from_vec(
            action_logits.clone(),
            (particles, horizon, transcription.num_actions()),
            &device,
        )
        .expect("action logits");
        let u = Tensor::from_vec(
            state_logits.clone(),
            (particles, horizon, transcription.num_facts()),
            &device,
        )
        .expect("state logits");

        let forward = plan
            .forward(&z, &u, &action_temperature, &state_temperature)
            .expect("forward pass");

        let action = to_vec3(&forward.action);
        let state = to_vec3(&forward.state);

        for particle in 0..particles {
            // Rebuild the same assignment for the reference implementation by
            // copying the tensor distributions, so any disagreement is in the
            // residuals rather than in the softmaxes.
            let mut assignment = Assignment::zeros(&transcription, horizon);
            for t in 0..horizon {
                assignment
                    .action_row_mut(t)
                    .copy_from_slice(&action[particle][t]);
            }
            for t in 0..=horizon {
                assignment
                    .state_row_mut(t)
                    .copy_from_slice(&state[particle][t]);
            }

            let reference = evaluate(&transcription, &assignment);
            let precondition = to_vec3(&forward.precondition);
            let goal = to_vec3(&forward.goal);
            let transition: Vec<Vec<Vec<Vec<f64>>>> =
                forward.transition.iter().map(to_vec3).collect();

            let num_pre = transcription.pre_action().len();
            for t in 0..horizon {
                for index in 0..num_pre {
                    let expected = reference.precondition[t * num_pre + index];
                    let got = precondition[particle][t][index];
                    assert!(
                        (got - expected).abs() < 1e-10,
                        "precondition residual mismatch at t={t} index={index}: {got} vs {expected}"
                    );
                }
                for fact in 0..transcription.num_facts() {
                    for family in 0..4 {
                        let expected =
                            reference.transition[family][t * transcription.num_facts() + fact];
                        let got = transition[family][particle][t][fact];
                        assert!(
                            (got - expected).abs() < 1e-10,
                            "transition[{family}] mismatch at t={t} fact={fact}: {got} vs {expected}"
                        );
                    }
                }
            }
            for (index, &expected) in reference.goal.iter().enumerate() {
                let got = goal[particle][0][index];
                assert!(
                    (got - expected).abs() < 1e-10,
                    "goal residual mismatch at {index}: {got} vs {expected}"
                );
            }
            compared += 1;
            multi_write_compared += usize::from(transcription.max_group_size() > 1);
            conditional_multi_write_compared += usize::from(
                transcription.max_group_size() > 1 && !transcription.cond_effect().is_empty(),
            );
        }
    }

    assert!(
        compared > 400,
        "too few forward passes compared: {compared}"
    );
    assert!(
        multi_write_compared > 100,
        "too few multi-write forward passes compared: {multi_write_compared}"
    );
    assert!(
        conditional_multi_write_compared > 100,
        "too few conditional multi-write passes compared: {conditional_multi_write_compared}"
    );
}

/// State rows must be proper per-variable distributions: that structural
/// property is what rules out a relaxed state placing one object in two
/// mutually exclusive situations.
#[test]
fn state_rows_are_per_variable_distributions() {
    let device = cpu();
    let mut rng = Rng::new(0xD15C_0FFEE);
    let horizon = 3;
    let particles = 2;

    // Find a task the tensor backend accepts. Fail loudly rather than silently
    // testing nothing if the generator never produces one.
    let (transcription, plan) = (0..200)
        .find_map(|_| {
            let task = random_task(&mut rng);
            let transcription = Transcription::build(&task).ok()?;
            let plan = TensorPlan::new(&transcription, horizon, particles, device.clone()).ok()?;
            Some((transcription, plan))
        })
        .expect("the generator produced no tensor-representable task");

    let u = Tensor::rand(
        -1.0f64,
        1.0f64,
        (particles, horizon, transcription.num_facts()),
        &device,
    )
    .expect("random logits")
    .to_dtype(DTYPE)
    .expect("dtype");

    let temperature =
        Tensor::from_vec(vec![0.8f64; particles], (particles, 1, 1), &device).expect("temperature");
    let state = plan
        .state_distribution(&u, &temperature)
        .expect("state rows");
    let values = to_vec3(&state);

    for particle in 0..particles {
        for t in 0..=horizon {
            for var in 0..transcription.num_variables() {
                let offset = transcription.var_offset()[var] as usize;
                let size = transcription.var_domain()[var] as usize;
                let total: f64 = values[particle][t][offset..offset + size].iter().sum();
                assert!(
                    (total - 1.0).abs() < 1e-10,
                    "variable {var} at t={t} sums to {total}, not 1"
                );
            }
        }
    }

    // Row 0 is the fixed initial state, so it must be exactly integral.
    for particle in 0..particles {
        for (fact, &value) in values[particle][0].iter().enumerate() {
            let expected = if transcription.initial_fact().contains(&(fact as u32)) {
                1.0
            } else {
                0.0
            };
            assert!(
                (value - expected).abs() < 1e-12,
                "initial row fact {fact} is {value}, expected {expected}"
            );
        }
    }
}

fn joint_effect_task(initial_right: usize) -> NumericRootTask {
    let variables = vec![
        ExplicitVariable::new(
            2,
            "global".to_string(),
            vec!["holds".to_string(), "default".to_string()],
            Some(0),
            1,
        ),
        ExplicitVariable::new(
            2,
            "left".to_string(),
            vec!["left-0".to_string(), "left-1".to_string()],
            None,
            0,
        ),
        ExplicitVariable::new(
            2,
            "right".to_string(),
            vec!["right-0".to_string(), "right-1".to_string()],
            None,
            0,
        ),
    ];
    let action = Operator::new(
        "set-both".to_string(),
        vec![
            ExplicitFact::propositional(1, 0),
            ExplicitFact::propositional(2, 0),
        ],
        vec![
            Effect::new(Vec::new(), 1, None, 1),
            Effect::new(Vec::new(), 2, None, 1),
        ],
        Vec::new(),
        1,
    );
    NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        Vec::new(),
        vec![
            ExplicitFact::propositional(1, 1),
            ExplicitFact::propositional(2, 1),
        ],
        Vec::new(),
        vec![1, 0, initial_right],
        Vec::new(),
        vec![action],
        vec![PropositionalAxiom::new(Vec::new(), 0, 1, 0)],
        Vec::new(),
        Vec::new(),
        ExplicitFact::propositional(0, 0),
    )
}

/// An action is one stochastic joint event. Giving it mass `p` gives every one
/// of its effects mass `p`; the effects are not sampled independently.
#[test]
fn stochastic_action_mass_is_shared_by_all_joint_effects() {
    let device = cpu();
    let transcription =
        Transcription::build(&joint_effect_task(0)).expect("joint-effect transcription");
    let plan = TensorPlan::new(&transcription, 2, 1, device.clone()).expect("tensor plan");

    let p = 0.7f64;
    let action_logit = (p / (1.0 - p)).ln();
    let logits = Tensor::from_vec(
        vec![action_logit, 0.0, 0.0, 0.0],
        (1, 2, transcription.num_actions()),
        &device,
    )
    .expect("action logits");
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
    let forward = plan
        .two_loss_forward(&logits, &temperature)
        .expect("action-only forward pass");

    let action = to_vec3(&forward.action);
    assert!((action[0][0][0] - p).abs() < 1e-12);
    let state = to_vec3(&forward.exact_state_by_step);
    for var in 0..2 {
        let old = transcription.fact(var, 0) as usize;
        let new = transcription.fact(var, 1) as usize;
        assert!(
            (state[0][1][new] - p).abs() < 1e-12,
            "effect on variable {var} received mass {}, expected {p}",
            state[0][1][new]
        );
        assert!(
            (state[0][1][old] - (1.0 - p)).abs() < 1e-12,
            "persistence on variable {var} is {}, expected {}",
            state[0][1][old],
            1.0 - p
        );
    }

    let failed = forward
        .failed_precondition_by_step
        .to_vec2::<f64>()
        .expect("per-step precondition loss");
    assert_eq!(
        failed,
        forward
            .support_only_failed_precondition_by_step
            .to_vec2::<f64>()
            .expect("support-only precondition loss")
    );
    assert!(
        failed[0][0].abs() < 1e-12,
        "an applicable stochastic action has loss {}",
        failed[0][0]
    );

    let causal_add = to_vec3(&forward.causal.add);
    let causal_delete = to_vec3(&forward.causal.delete);
    let causal_demand = to_vec3(&forward.causal.action_demand);
    for var in 0..2 {
        let old = transcription.fact(var, 0) as usize;
        let new = transcription.fact(var, 1) as usize;
        assert!((causal_add[0][0][new] - p).abs() < 1e-12);
        assert!((causal_delete[0][0][old] - p).abs() < 1e-12);
        assert!((causal_demand[0][0][old] - p).abs() < 1e-12);
    }
}

#[test]
fn straight_through_hardening_has_exact_endpoints_and_linear_forward_interpolation() {
    let device = cpu();
    let transcription =
        Transcription::build(&joint_effect_task(0)).expect("joint-effect transcription");
    let plan = TensorPlan::new(&transcription, 2, 1, device.clone()).expect("tensor plan");
    let logits = Tensor::from_vec(
        vec![(0.7f64 / 0.3).ln(), 0.0, (0.4f64 / 0.6).ln(), 0.0],
        (1, plan.horizon, plan.num_actions),
        &device,
    )
    .expect("action logits");
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");

    let wrapper = plan
        .two_loss_forward(&logits, &temperature)
        .expect("wrapper forward");
    let soft = plan
        .two_loss_forward_hardened(&logits, &temperature, 0.0)
        .expect("soft endpoint");
    let midpoint = plan
        .two_loss_forward_hardened(&logits, &temperature, 0.5)
        .expect("midpoint");
    let hard = plan
        .two_loss_forward_hardened(&logits, &temperature, 1.0)
        .expect("hard endpoint");

    let wrapper = to_vec3(&wrapper.action);
    let soft = to_vec3(&soft.action);
    let midpoint = to_vec3(&midpoint.action);
    let hard = to_vec3(&hard.action);
    assert_eq!(
        wrapper, soft,
        "the historical wrapper is exactly alpha zero"
    );
    assert_eq!(hard, vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]]]);
    for row in 0..plan.horizon {
        for action in 0..plan.num_actions {
            let expected = 0.5 * (soft[0][row][action] + hard[0][row][action]);
            assert!(
                (midpoint[0][row][action] - expected).abs() < 1e-12,
                "row {row}, action {action}: {} != {expected}",
                midpoint[0][row][action]
            );
        }
    }

    for alpha in [-f64::EPSILON, 1.0 + f64::EPSILON, f64::NAN] {
        assert!(
            plan.two_loss_forward_hardened(&logits, &temperature, alpha)
                .is_err(),
            "invalid alpha {alpha} must be rejected"
        );
    }
}

#[test]
fn fully_hardened_action_only_recurrence_matches_the_decoded_one_hot_plan() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 3, 1, device.clone()).expect("tensor plan");
    let noop = plan.num_actions - 1;
    let mut values = vec![-0.5f64; plan.horizon * plan.num_actions];
    for (row, action) in [0, 1, noop].into_iter().enumerate() {
        values[row * plan.num_actions + action] = 0.5;
    }
    let logits = Tensor::from_vec(values, (1, plan.horizon, plan.num_actions), &device)
        .expect("action logits");
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
    let forward = plan
        .two_loss_forward_hardened(&logits, &temperature, 1.0)
        .expect("hard action-only forward");

    let actions = to_vec3(&forward.action);
    for (row, expected) in [0, 1, noop].into_iter().enumerate() {
        assert_eq!(actions[0][row][expected], 1.0);
        assert_eq!(actions[0][row].iter().sum::<f64>(), 1.0);
    }
    let states = to_vec3(&forward.exact_state_by_step);
    for (row, value) in [0, 1, 2].into_iter().enumerate() {
        let fact = transcription.fact(0, value) as usize;
        assert_eq!(states[0][row][fact], 1.0, "wrong exact state at row {row}");
    }
    assert_eq!(
        forward
            .failed_precondition
            .sum_all()
            .expect("precondition sum")
            .to_scalar::<f64>()
            .expect("precondition scalar"),
        0.0
    );
    assert_eq!(
        forward
            .terminal_goal_by_goal
            .sum_all()
            .expect("terminal goal sum")
            .to_scalar::<f64>()
            .expect("goal scalar"),
        0.0
    );
}

#[test]
fn fully_hardened_terminal_goal_retains_a_useful_softmax_surrogate_gradient() {
    let device = cpu();
    let transcription =
        Transcription::build(&joint_effect_task(0)).expect("joint-effect transcription");
    let plan = TensorPlan::new(&transcription, 1, 1, device.clone()).expect("tensor plan");
    let logits = candle_core::Var::from_vec(
        vec![0.0f64, 1.0],
        (1, plan.horizon, plan.num_actions),
        &device,
    )
    .expect("variable logits");
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
    let forward = plan
        .two_loss_forward_hardened(logits.as_tensor(), &temperature, 1.0)
        .expect("hard action-only forward");
    assert_eq!(
        to_vec3(&forward.action),
        vec![vec![vec![0.0, 1.0]]],
        "the decoded no-op must be used in the forward pass"
    );
    let loss = forward
        .terminal_goal_by_goal
        .sum_all()
        .expect("terminal goal loss");
    assert_eq!(loss.to_scalar::<f64>().expect("loss scalar"), 2.0);
    let gradients = loss.backward().expect("straight-through backward");
    let gradient = to_vec3(
        gradients
            .get(&logits)
            .expect("terminal goal must reach the action logits"),
    );
    assert!(gradient[0][0][0].is_finite());
    assert!(
        gradient[0][0][0] < -1e-3,
        "goal loss should increase the useful real action logit, got {}",
        gradient[0][0][0]
    );
    assert!(
        gradient[0][0][1] > 1e-3,
        "goal loss should decrease the decoded no-op logit, got {}",
        gradient[0][0][1]
    );
}

/// Propagation uses the exact precondition conjunction, while the auxiliary
/// loss remains additive over missing literals. This catches a reintroduction
/// of either mean-gated or blended effect support.
#[test]
fn false_literal_blocks_all_effects_without_blocking_the_literal_loss_gradient() {
    let device = cpu();
    // `set-both` requires right=0, but this initial state has right=1.
    let transcription =
        Transcription::build(&joint_effect_task(1)).expect("joint-effect transcription");
    let plan = TensorPlan::new(&transcription, 2, 1, device.clone()).expect("tensor plan");

    let p = 0.7f64;
    let action_logit = (p / (1.0 - p)).ln();
    let logits = candle_core::Var::from_vec(
        vec![action_logit, 0.0, 0.0, 0.0],
        (1, 2, transcription.num_actions()),
        &device,
    )
    .expect("action logits");
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
    let forward = plan
        .two_loss_forward(logits.as_tensor(), &temperature)
        .expect("action-only forward pass");

    let state = to_vec3(&forward.exact_state_by_step);
    let left_new = transcription.fact(0, 1) as usize;
    assert_eq!(
        state[0][1][left_new], 0.0,
        "an action with a false precondition must contribute exactly zero effect mass"
    );
    for tensor in [
        &forward.causal.action_demand,
        &forward.causal.add,
        &forward.causal.delete,
    ] {
        assert!(
            tensor
                .narrow(1, 0, 1)
                .expect("first causal row")
                .sum_all()
                .expect("causal mass")
                .to_scalar::<f64>()
                .expect("scalar")
                .abs()
                < 1e-12,
            "an inapplicable action leaked into coherent causal evidence"
        );
    }

    let failed = forward
        .failed_precondition_by_step
        .to_vec2::<f64>()
        .expect("per-step precondition loss");
    assert!(
        (failed[0][0] - p / 2.0).abs() < 1e-12,
        "one of two false literals should contribute p/2, got {}",
        failed[0][0]
    );
    assert_eq!(
        failed,
        forward
            .support_only_failed_precondition_by_step
            .to_vec2::<f64>()
            .expect("support-only false-precondition loss")
    );

    let loss = forward
        .failed_precondition_by_step
        .sum_all()
        .expect("precondition loss");
    let gradients = loss.backward().expect("literal-loss backward");
    let gradient = to_vec3(
        gradients
            .get(&logits)
            .expect("gradient for the action logits"),
    );
    assert!(
        (gradient[0][0][0] - p * (1.0 - p) / 2.0).abs() < 1e-12,
        "literal loss has gradient {}, expected {}",
        gradient[0][0][0],
        p * (1.0 - p) / 2.0
    );
}

#[test]
fn protected_precondition_keeps_the_violation_but_not_the_consumer_gradient() {
    let device = cpu();
    let transcription =
        Transcription::build(&joint_effect_task(1)).expect("joint-effect transcription");
    let plan = TensorPlan::new(&transcription, 2, 1, device.clone()).expect("tensor plan");
    let action_logits = candle_core::Var::from_vec(
        vec![1.0f64, 0.0, 1.0, 0.0],
        (1, 2, transcription.num_actions()),
        &device,
    )
    .expect("action logits");
    let state_logits = candle_core::Var::zeros((1, 2, transcription.num_facts()), DTYPE, &device)
        .expect("state logits");
    let action_temperature = Tensor::ones((1, 2, 1), DTYPE, &device).expect("action temperature");
    let state_temperature = Tensor::ones((1, 2, 1), DTYPE, &device).expect("state temperature");
    let forward = plan
        .forward(
            action_logits.as_tensor(),
            state_logits.as_tensor(),
            &action_temperature,
            &state_temperature,
        )
        .expect("direct forward");
    let live_mask = Tensor::zeros((1, 1, 1), DTYPE, &device).expect("live mask");
    let protected_mask = Tensor::ones((1, 1, 1), DTYPE, &device).expect("protected mask");
    let live = plan
        .protected_precondition_residual(&forward.action, &forward.state, &live_mask)
        .expect("live residual");
    let protected = plan
        .protected_precondition_residual(&forward.action, &forward.state, &protected_mask)
        .expect("protected residual");
    assert_eq!(to_vec3(&live), to_vec3(&protected));

    let live_gradients = live
        .sum_all()
        .expect("live loss")
        .backward()
        .expect("live gradient");
    let live_action = live_gradients
        .get(&action_logits)
        .expect("live consumer gradient")
        .abs()
        .expect("absolute live gradient")
        .sum_all()
        .expect("live gradient norm")
        .to_scalar::<f64>()
        .expect("live scalar");
    assert!(live_action > 0.0);

    let protected_gradients = protected
        .sum_all()
        .expect("protected loss")
        .backward()
        .expect("protected gradient");
    let protected_action = protected_gradients
        .get(&action_logits)
        .expect("zero consumer gradient remains shape-aligned")
        .abs()
        .expect("absolute protected action gradient")
        .sum_all()
        .expect("protected action gradient norm")
        .to_scalar::<f64>()
        .expect("protected action scalar");
    assert_eq!(protected_action, 0.0);
    let support_gradient = protected_gradients
        .get(&state_logits)
        .expect("support-state gradient")
        .abs()
        .expect("absolute support gradient")
        .sum_all()
        .expect("support gradient norm")
        .to_scalar::<f64>()
        .expect("support scalar");
    assert!(support_gradient > 0.0);

    let live_transition = plan
        .protected_transition_residual(&forward.action, &forward.state, &live_mask)
        .expect("live transition residual");
    let protected_transition = plan
        .protected_transition_residual(&forward.action, &forward.state, &protected_mask)
        .expect("protected transition residual");
    for family in 0..4 {
        assert_eq!(
            to_vec3(&live_transition[family]),
            to_vec3(&protected_transition[family]),
            "protection changes gradients, never transition values"
        );
    }
    let live_transition_loss = live_transition
        .iter()
        .try_fold(
            Tensor::zeros((), DTYPE, &device).expect("zero transition loss"),
            |loss, residual| loss + residual.sum_all()?,
        )
        .expect("live transition loss");
    let live_transition_gradients = live_transition_loss
        .backward()
        .expect("live transition gradient");
    let live_transition_action = live_transition_gradients
        .get(&action_logits)
        .expect("live transition action gradient")
        .abs()
        .expect("absolute transition action gradient")
        .sum_all()
        .expect("transition action gradient norm")
        .to_scalar::<f64>()
        .expect("transition action scalar");
    assert!(live_transition_action > 0.0);

    let protected_transition_loss = protected_transition
        .iter()
        .try_fold(
            Tensor::zeros((), DTYPE, &device).expect("zero protected transition loss"),
            |loss, residual| loss + residual.sum_all()?,
        )
        .expect("protected transition loss");
    let protected_transition_gradients = protected_transition_loss
        .backward()
        .expect("protected transition gradient");
    let protected_transition_action = protected_transition_gradients
        .get(&action_logits)
        .expect("zero protected transition action gradient remains shape-aligned")
        .abs()
        .expect("absolute protected transition action gradient")
        .sum_all()
        .expect("protected transition action gradient norm")
        .to_scalar::<f64>()
        .expect("protected transition action scalar");
    assert_eq!(protected_transition_action, 0.0);
    let protected_transition_state = protected_transition_gradients
        .get(&state_logits)
        .expect("protected transition state gradient")
        .abs()
        .expect("absolute protected transition state gradient")
        .sum_all()
        .expect("protected transition state gradient norm")
        .to_scalar::<f64>()
        .expect("protected transition state scalar");
    assert!(protected_transition_state > 0.0);
}

#[test]
fn state_integrality_keeps_a_single_fractional_variable_visible() {
    let device = cpu();
    let transcription =
        Transcription::build(&joint_effect_task(0)).expect("joint-effect transcription");
    let plan = TensorPlan::new(&transcription, 1, 1, device.clone()).expect("tensor plan");

    // Variable zero is uniform while variable one is essentially one-hot.
    let mut state_logits = vec![0.0f64; transcription.num_facts()];
    state_logits[transcription.fact(1, 0) as usize] = 30.0;
    state_logits[transcription.fact(1, 1) as usize] = -30.0;
    let logits = Tensor::from_vec(state_logits, (1, 1, transcription.num_facts()), &device)
        .expect("state logits");
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
    let state = plan
        .state_distribution(&logits, &temperature)
        .expect("state distribution");
    let integrality = plan
        .state_integrality_per_particle(&state)
        .expect("state integrality");

    assert_eq!(integrality.dims(), &[1, 1, 2]);
    let values = to_vec3(&integrality);
    assert!((values[0][0][0] - 0.5).abs() < 1e-12);
    assert!(
        values[0][0][1] < 1e-12,
        "integral variable has penalty {}",
        values[0][0][1]
    );
}

#[test]
fn bottleneck_norm_is_particle_local_and_rejects_invalid_exponents() {
    let device = cpu();
    let residual = Tensor::from_vec(vec![0.0f64, 2.0, 0.0, 0.0, 1.0, 1.0], (2, 1, 3), &device)
        .expect("residuals");
    let norm = bottleneck_norm_per_particle(&residual, 2.0).expect("bottleneck norm");
    let values = norm.to_vec2::<f64>().expect("particle norms");
    assert!((values[0][0] - (4.0f64 / 3.0).sqrt()).abs() < 1e-12);
    assert!((values[1][0] - (2.0f64 / 3.0).sqrt()).abs() < 1e-12);
    assert!(bottleneck_norm_per_particle(&residual, 0.5).is_err());
    assert!(bottleneck_norm_per_particle(&residual, 65.0).is_err());
    assert!(bottleneck_norm_per_particle(&residual, f64::NAN).is_err());

    let zero = candle_core::Var::zeros((1, 2, 3), DTYPE, &device).expect("zero residuals");
    let zero_norm = bottleneck_norm_per_particle(zero.as_tensor(), 8.0).expect("zero norm");
    assert!(scalar(&zero_norm).abs() < 1e-300);
    let gradients = zero_norm
        .sum_all()
        .expect("zero loss")
        .backward()
        .expect("zero backward");
    let gradient = gradients.get(&zero).expect("zero residual gradient");
    assert!(
        gradient
            .flatten_all()
            .expect("flat gradient")
            .to_vec1::<f64>()
            .expect("gradient values")
            .iter()
            .all(|value| value.is_finite() && *value == 0.0)
    );

    let empty = Tensor::zeros((2, 0), DTYPE, &device).expect("empty residual family");
    assert_eq!(
        bottleneck_norm_per_particle(&empty, 8.0)
            .expect("empty norm")
            .to_vec2::<f64>()
            .expect("empty particle norms"),
        vec![vec![0.0], vec![0.0]]
    );
}

#[test]
fn noop_suffix_penalty_detects_only_real_action_after_noop() {
    let device = cpu();
    let transcription =
        Transcription::build(&joint_effect_task(0)).expect("joint-effect transcription");
    let plan = TensorPlan::new(&transcription, 3, 1, device.clone()).expect("tensor plan");
    let noop = transcription.noop_action();
    let one_hot = |choices: &[usize]| {
        let mut values = vec![0.0f64; choices.len() * plan.num_actions];
        for (row, &choice) in choices.iter().enumerate() {
            values[row * plan.num_actions + choice] = 1.0;
        }
        Tensor::from_vec(values, (1, choices.len(), plan.num_actions), &device)
            .expect("action probabilities")
    };
    let suffix = one_hot(&[0, noop, noop]);
    assert!(scalar(&plan.noop_suffix_penalty(&suffix).expect("suffix penalty")) < 1e-12);

    let internal_noop = one_hot(&[noop, 0, noop]);
    assert!(
        scalar(
            &plan
                .noop_suffix_penalty(&internal_noop)
                .expect("internal-noop penalty")
        ) > 0.49
    );
}

#[test]
fn slot_slack_penalty_requires_a_local_blank_without_forcing_a_suffix() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 4, 1, device.clone()).expect("tensor plan");
    let noop = transcription.noop_action();
    let one_hot = |choices: &[usize]| {
        let mut values = vec![0.0f64; choices.len() * plan.num_actions];
        for (row, &choice) in choices.iter().enumerate() {
            values[row * plan.num_actions + choice] = 1.0;
        }
        Tensor::from_vec(values, (1, choices.len(), plan.num_actions), &device)
            .expect("action probabilities")
    };

    let interleaved = one_hot(&[0, 1, noop, 2]);
    assert!(scalar(&plan.slot_slack_penalty(&interleaved, 3).expect("slack")) < 1e-12);

    let dense = one_hot(&[0, 1, 2, 0]);
    assert!(scalar(&plan.slot_slack_penalty(&dense, 3).expect("slack")) > 0.99);

    let suffix_blank = one_hot(&[0, 1, 2, noop]);
    assert!(
        scalar(&plan.slot_slack_penalty(&suffix_blank, 3).expect("slack")) > 0.49,
        "a blank only at the end does not repair the first dense window"
    );
}

#[test]
fn factorized_slots_cover_the_action_simplex_and_keep_empty_slot_gradients() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 1, 1, device.clone()).expect("tensor plan");
    assert_eq!(plan.num_actions, 4, "three real actions plus no-op");

    // Desired full distribution [0.1, 0.2, 0.3, 0.4]. Conditional real
    // probabilities are [1/6, 2/6, 3/6] and occupancy is 0.6.
    let logits = Tensor::from_vec(
        vec![
            (1.0f64 / 6.0).ln(),
            (2.0f64 / 6.0).ln(),
            0.5f64.ln(),
            (0.6f64 / 0.4).ln() - 3.0f64.ln(),
        ],
        (1, 1, 4),
        &device,
    )
    .expect("factorized logits");
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
    let slots = plan
        .factorized_action_distribution(&logits, &temperature)
        .expect("factorized distribution");
    let got = slots.action.to_vec3::<f64>().expect("action probabilities");
    for (actual, expected) in got[0][0].iter().zip([0.1, 0.2, 0.3, 0.4]) {
        assert!((actual - expected).abs() < 1e-12, "{actual} vs {expected}");
    }
    assert!((scalar(&slots.occupancy) - 0.6).abs() < 1e-12);

    // Even a nearly empty slot retains a direct gradient that can open it for
    // a demanded real action. Stable log probabilities stay finite at a much
    // more saturated finite logit as well.
    let variable = candle_core::Var::from_vec(vec![2.0, 0.0, -1.0, -8.0], (1, 1, 4), &device)
        .expect("slot variable");
    let slots = plan
        .factorized_action_distribution(variable.as_tensor(), &temperature)
        .expect("nearly empty slot");
    let loss = slots
        .log_action
        .narrow(2, 0, 1)
        .expect("first action")
        .neg()
        .expect("negative log probability")
        .sum_all()
        .expect("scalar loss");
    let gradients = loss.backward().expect("slot backward");
    let gradient = gradients
        .get(&variable)
        .expect("slot gradient")
        .to_vec3::<f64>()
        .expect("rank-three gradient");
    assert!(
        gradient[0][0][3] < -0.99,
        "gradient descent must increase occupancy: {:?}",
        gradient[0][0]
    );

    let extreme =
        Tensor::from_vec(vec![0.0, 0.0, 0.0, 1000.0], (1, 1, 4), &device).expect("extreme logits");
    let stable = plan
        .factorized_action_distribution(&extreme, &temperature)
        .expect("stable extreme distribution")
        .log_action
        .flatten_all()
        .expect("flat log probabilities")
        .to_vec1::<f64>()
        .expect("log values");
    assert!(stable.iter().all(|value| value.is_finite()), "{stable:?}");
}

#[test]
fn hybrid_slots_change_only_the_reserved_insertion_rows() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 3, 1, device.clone()).expect("tensor plan");
    let logits = Tensor::from_vec(
        vec![
            1.0f64, 0.0, -0.5, 0.2, 0.3, 1.2, -0.7, -0.1, 0.4, -0.2, 0.8, -2.0,
        ],
        (1, 3, 4),
        &device,
    )
    .expect("action logits");
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
    let categorical = plan
        .action_distribution(&logits, &temperature)
        .expect("categorical distribution")
        .to_vec3::<f64>()
        .expect("categorical values");
    let hybrid = plan
        .hybrid_action_distribution(&logits, &temperature, 3)
        .expect("hybrid distribution")
        .action
        .to_vec3::<f64>()
        .expect("hybrid values");
    for row in 0..2 {
        for action in 0..4 {
            assert!((hybrid[0][row][action] - categorical[0][row][action]).abs() < 1e-15);
        }
    }
    assert_ne!(hybrid[0][2], categorical[0][2]);

    let disabled = plan
        .hybrid_action_distribution(&logits, &temperature, 0)
        .expect("disabled hybrid")
        .action
        .to_vec3::<f64>()
        .expect("disabled values");
    for row in 0..3 {
        for action in 0..4 {
            assert!((disabled[0][row][action] - categorical[0][row][action]).abs() < 1e-15);
        }
    }
}

fn causal_chain_task() -> NumericRootTask {
    let variables = vec![
        ExplicitVariable::new(
            2,
            "global".to_string(),
            vec!["holds".to_string(), "default".to_string()],
            Some(0),
            1,
        ),
        ExplicitVariable::new(
            3,
            "position".to_string(),
            vec!["at-0".to_string(), "at-1".to_string(), "at-2".to_string()],
            None,
            0,
        ),
    ];
    let move_01 = Operator::new(
        "move-0-1".to_string(),
        vec![ExplicitFact::propositional(1, 0)],
        vec![Effect::new(Vec::new(), 1, None, 1)],
        Vec::new(),
        1,
    );
    let move_12 = Operator::new(
        "move-1-2".to_string(),
        vec![ExplicitFact::propositional(1, 1)],
        vec![Effect::new(Vec::new(), 1, None, 2)],
        Vec::new(),
        1,
    );
    let move_10 = Operator::new(
        "move-1-0".to_string(),
        vec![ExplicitFact::propositional(1, 1)],
        vec![Effect::new(Vec::new(), 1, None, 0)],
        Vec::new(),
        1,
    );
    NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        Vec::new(),
        vec![ExplicitFact::propositional(1, 2)],
        Vec::new(),
        vec![1, 0],
        Vec::new(),
        vec![move_01, move_12, move_10],
        vec![PropositionalAxiom::new(Vec::new(), 0, 1, 0)],
        Vec::new(),
        Vec::new(),
        ExplicitFact::propositional(0, 0),
    )
}

/// The selected sequence reaches the first goal, then clobbers it while
/// reaching the second. A delete relaxation sees both facts forever; the real
/// rollout must not.
fn clobbered_goals_task(initial_first_goal: bool) -> NumericRootTask {
    let variables = vec![
        ExplicitVariable::new(
            2,
            "global".to_string(),
            vec!["holds".to_string(), "default".to_string()],
            Some(0),
            1,
        ),
        ExplicitVariable::new(
            2,
            "first-goal".to_string(),
            vec!["first-0".to_string(), "first-1".to_string()],
            None,
            0,
        ),
        ExplicitVariable::new(
            2,
            "second-goal".to_string(),
            vec!["second-0".to_string(), "second-1".to_string()],
            None,
            0,
        ),
    ];
    let set_one = Operator::new(
        "set-one".to_string(),
        Vec::new(),
        vec![Effect::new(Vec::new(), 1, None, 1)],
        Vec::new(),
        1,
    );
    let clobber_one_set_two = Operator::new(
        "clobber-one-set-two".to_string(),
        Vec::new(),
        vec![
            Effect::new(Vec::new(), 1, None, 0),
            Effect::new(Vec::new(), 2, None, 1),
        ],
        Vec::new(),
        1,
    );
    NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        Vec::new(),
        vec![
            ExplicitFact::propositional(1, 1),
            ExplicitFact::propositional(2, 1),
        ],
        Vec::new(),
        vec![1, usize::from(initial_first_goal), 0],
        Vec::new(),
        vec![set_one, clobber_one_set_two],
        vec![PropositionalAxiom::new(Vec::new(), 0, 1, 0)],
        Vec::new(),
        Vec::new(),
        ExplicitFact::propositional(0, 0),
    )
}

#[test]
fn delete_aware_terminal_goal_and_noop_pressure_detect_clobbered_goal() {
    let device = cpu();
    let transcription =
        Transcription::build(&clobbered_goals_task(false)).expect("clobbered-goal transcription");
    let plan = TensorPlan::new(&transcription, 3, 1, device.clone()).expect("tensor plan");
    let noop = transcription.num_actions() - 1;
    let logits = Tensor::from_vec(
        peaked_rows(3, transcription.num_actions(), &[0, 1, noop]),
        (1, 3, transcription.num_actions()),
        &device,
    )
    .expect("action logits");
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
    let forward = plan
        .two_loss_forward(&logits, &temperature)
        .expect("action-only forward pass");

    let relaxed = forward
        .relaxed_goal_by_goal
        .to_vec2::<f64>()
        .expect("relaxed goal loss");
    assert!(
        relaxed[0].iter().all(|loss| *loss < 1e-20),
        "the monotone producer relaxation should see both goals: {:?}",
        relaxed[0]
    );
    let terminal = forward
        .terminal_goal_by_goal
        .to_vec2::<f64>()
        .expect("delete-aware terminal goal loss");
    assert!(
        terminal[0][0] > 0.99,
        "clobbered goal loss: {:?}",
        terminal[0]
    );
    assert!(
        terminal[0][1] < 1e-20,
        "finally achieved goal loss: {:?}",
        terminal[0]
    );
    assert!(
        forward
            .premature_noop
            .to_vec2::<f64>()
            .expect("no-op pressure")[0][0]
            > 0.1,
        "the no-op after clobbering a goal must remain penalized"
    );
}

#[test]
fn analytic_goal_survival_matches_one_hot_terminal_order_and_has_threat_gradient() {
    let device = cpu();
    let transcription =
        Transcription::build(&clobbered_goals_task(false)).expect("clobbered-goal transcription");
    let plan = TensorPlan::new(&transcription, 3, 1, device.clone()).expect("tensor plan");
    let noop = transcription.noop_action();
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");

    let producer_then_delete = Tensor::from_vec(
        peaked_rows(3, transcription.num_actions(), &[0, 1, noop]),
        (1, 3, transcription.num_actions()),
        &device,
    )
    .expect("producer then delete logits");
    let clobbered = plan
        .two_loss_forward(&producer_then_delete, &temperature)
        .expect("clobbered forward");
    let clobbered = clobbered
        .surviving_goal_by_goal
        .to_vec2::<f64>()
        .expect("survival loss");
    assert!(
        clobbered[0][0] > 0.99,
        "deleted first goal: {:?}",
        clobbered[0]
    );
    assert!(
        clobbered[0][1] < 1e-20,
        "second goal survives: {:?}",
        clobbered[0]
    );

    let delete_then_producer = Tensor::from_vec(
        peaked_rows(3, transcription.num_actions(), &[1, 0, noop]),
        (1, 3, transcription.num_actions()),
        &device,
    )
    .expect("delete then producer logits");
    let repaired = plan
        .two_loss_forward(&delete_then_producer, &temperature)
        .expect("repaired forward")
        .surviving_goal_by_goal
        .to_vec2::<f64>()
        .expect("survival loss");
    assert!(
        repaired[0].iter().all(|loss| *loss < 1e-20),
        "{:?}",
        repaired[0]
    );

    let initial_transcription =
        Transcription::build(&clobbered_goals_task(true)).expect("initial-goal transcription");
    let initial_plan =
        TensorPlan::new(&initial_transcription, 3, 1, device.clone()).expect("initial plan");
    let noops = Tensor::from_vec(
        peaked_rows(
            3,
            initial_transcription.num_actions(),
            &[initial_transcription.noop_action(); 3],
        ),
        (1, 3, initial_transcription.num_actions()),
        &device,
    )
    .expect("noop logits");
    let initial_loss = initial_plan
        .two_loss_forward(&noops, &temperature)
        .expect("initial forward")
        .surviving_goal_by_goal
        .to_vec2::<f64>()
        .expect("survival loss");
    assert!(initial_loss[0][0] < 1e-20, "initial goal must survive");

    let fractional = Tensor::from_vec(
        vec![0.3f64, -0.2, 0.1, -0.4, 0.7, -0.5, 0.2, -0.1, 0.4],
        (1, 3, transcription.num_actions()),
        &device,
    )
    .expect("fractional logits");
    for loss in plan
        .two_loss_forward(&fractional, &temperature)
        .expect("fractional forward")
        .surviving_goal_by_goal
        .to_vec2::<f64>()
        .expect("bounded survival loss")[0]
        .iter()
    {
        assert!(
            (0.0..=1.0).contains(loss),
            "survival loss outside [0,1]: {loss}"
        );
    }

    let threatening_logits = candle_core::Var::from_vec(
        vec![5.0f64, -5.0, -5.0, -2.0, 2.0, -2.0, -5.0, -5.0, 5.0],
        (1, 3, transcription.num_actions()),
        &device,
    )
    .expect("threatening logits");
    let threat_loss = plan
        .two_loss_forward(threatening_logits.as_tensor(), &temperature)
        .expect("threat forward")
        .surviving_goal_by_goal
        .narrow(1, 0, 1)
        .expect("first-goal loss")
        .sum_all()
        .expect("scalar threat loss");
    let gradient = to_vec3(
        threat_loss
            .backward()
            .expect("threat gradient")
            .get(&threatening_logits)
            .expect("logit gradient"),
    );
    assert!(
        gradient[0][1][1] > 1e-4,
        "increasing the threatening action must increase surviving-goal loss: {}",
        gradient[0][1][1]
    );
}

#[test]
fn deadline_support_separates_raw_producers_from_applicable_survival() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 3, 1, device.clone()).expect("tensor plan");
    let noop = transcription.noop_action();
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
    let peaked = |actions: &[usize]| {
        Tensor::from_vec(
            peaked_rows(3, plan.num_actions, actions),
            (1, 3, plan.num_actions),
            &device,
        )
        .expect("peaked logits")
    };
    let demand_fact = transcription.fact(0, 1) as usize;
    let mut demand = vec![0.0f64; 4 * plan.num_facts];
    demand[2 * plan.num_facts + demand_fact] = 1.0;
    let demand = Tensor::from_vec(demand, (1, 4, plan.num_facts), &device).expect("demand");
    let mut mask = vec![0.0f64; 4 * 3];
    mask[2 * 3] = 1.0;
    mask[2 * 3 + 1] = 1.0;
    let mask = Tensor::from_vec(mask, (1, 4, 3), &device).expect("source mask");

    let valid = plan
        .two_loss_forward_hardened(&peaked(&[0, noop, 1]), &temperature, 1.0)
        .expect("valid producer plan");
    let valid_support = plan
        .deadline_support_forward(&valid.action, &valid.causal, &demand, &mask, &[2])
        .expect("valid deadline support");
    assert!(scalar(&valid_support.raw_loss) < 1e-12);
    assert!(scalar(&valid_support.supported_loss) < 1e-12);

    let threatened = plan
        .two_loss_forward_hardened(&peaked(&[0, 2, 1]), &temperature, 1.0)
        .expect("threatened producer plan");
    let threatened_support = plan
        .deadline_support_forward(&threatened.action, &threatened.causal, &demand, &mask, &[2])
        .expect("threatened deadline support");
    assert!(
        scalar(&threatened_support.raw_loss) < 1e-12,
        "raw discovery deliberately ignores threats"
    );
    assert!(
        scalar(&threatened_support.supported_loss) > 20.0,
        "the proof lane must reject a producer deleted before its deadline"
    );

    // A repair suffix begins after move-0-1 deleted the initially true at-0
    // fact. The task initial state would fabricate evidence; the exact suffix
    // boundary must correctly report that a new producer is absent.
    let initial_fact = transcription.fact(0, 0) as usize;
    let mut boundary_demand = vec![0.0f64; 4 * plan.num_facts];
    boundary_demand[2 * plan.num_facts + initial_fact] = 1.0;
    let boundary_demand =
        Tensor::from_vec(boundary_demand, (1, 4, plan.num_facts), &device).expect("demand");
    let boundary_state = valid
        .exact_state_by_step
        .narrow(1, 1, 1)
        .expect("state after the first action");
    let from_task_initial = plan
        .deadline_support_forward(&valid.action, &valid.causal, &boundary_demand, &mask, &[2])
        .expect("legacy initial boundary");
    let from_exact_boundary = plan
        .deadline_support_forward_from_boundary(
            &valid.action,
            &valid.causal,
            &boundary_state,
            &boundary_demand,
            &mask,
            &[2],
        )
        .expect("exact repair boundary");
    assert!(scalar(&from_task_initial.raw_loss) < 1e-12);
    assert!(
        scalar(&from_exact_boundary.raw_loss) > 20.0,
        "a fact deleted before the repair suffix needs a new producer"
    );
}

#[test]
fn backward_goal_bridge_is_zero_on_a_supported_chain_and_repairs_its_missing_link() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 3, 1, device.clone()).expect("tensor plan");
    let noop = transcription.noop_action();
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
    let demand_fact = transcription.fact(0, 2) as usize;
    let mut demand = vec![0.0f64; plan.num_facts];
    demand[demand_fact] = 1.0;
    let demand = Tensor::from_vec(demand, (1, plan.num_facts), &device).expect("goal demand");
    let active = Tensor::ones((1, 3, 1), DTYPE, &device).expect("active suffix");
    let boundary =
        Tensor::from_vec(vec![1.0f64, 0.0, 0.0], (1, 3, 1), &device).expect("repair boundary");

    let valid_logits = Tensor::from_vec(
        peaked_rows(3, plan.num_actions, &[0, 1, noop]),
        (1, 3, plan.num_actions),
        &device,
    )
    .expect("valid logits");
    let valid = plan
        .two_loss_forward_hardened(&valid_logits, &temperature, 1.0)
        .expect("valid chain");
    let valid_bridge = plan
        .backward_goal_bridge(
            &valid.action,
            &valid.exact_state_by_step,
            &demand,
            &active,
            &boundary,
        )
        .expect("valid bridge");
    assert_eq!(scalar(&valid_bridge.loss), 0.0);

    // Row one contains the goal achiever, but its prerequisite producer is
    // undecided at row zero. The bridge must prefer that producer over no-op.
    let mut logits = peaked_rows(3, plan.num_actions, &[noop, 1, noop]);
    logits[0] = 0.0;
    logits[noop] = 0.0;
    let logits = candle_core::Var::from_vec(logits, (1, 3, plan.num_actions), &device)
        .expect("repair logits");
    let repairing = plan
        .two_loss_forward(logits.as_tensor(), &temperature)
        .expect("repairing chain");
    let bridge = plan
        .backward_goal_bridge(
            &repairing.action,
            &repairing.exact_state_by_step,
            &demand,
            &active,
            &boundary,
        )
        .expect("repair bridge");
    assert!(scalar(&bridge.loss) > 0.0);
    let gradient = bridge
        .loss
        .sum_all()
        .expect("scalar bridge loss")
        .backward()
        .expect("bridge gradient")
        .get(&logits)
        .expect("action gradient")
        .to_vec3::<f64>()
        .expect("rank-three gradient");
    assert!(
        gradient[0][0][0] < gradient[0][0][noop],
        "gradient descent must prefer the missing prerequisite producer: {:?}",
        gradient[0][0]
    );
    assert!(
        gradient[0][1][1] < gradient[0][1][noop],
        "induced prerequisites must not reverse the goal-producer gradient: {:?}",
        gradient[0][1]
    );

    let support_values = repairing
        .exact_state_by_step
        .flatten_all()
        .expect("flat support state")
        .to_vec1::<f64>()
        .expect("support values");
    let support = candle_core::Var::from_vec(support_values, (1, 3, plan.num_facts), &device)
        .expect("lifted support variable");
    let lifted_bridge = plan
        .backward_goal_bridge(
            &repairing.action,
            support.as_tensor(),
            &demand,
            &active,
            &boundary,
        )
        .expect("lifted-state bridge");
    let support_gradient = lifted_bridge
        .loss
        .sum_all()
        .expect("scalar lifted bridge")
        .backward()
        .expect("lifted bridge gradient")
        .get(&support)
        .expect("support-state gradient")
        .to_vec3::<f64>()
        .expect("rank-three support gradient");
    let missing_link_fact = transcription.fact(0, 1) as usize;
    assert!(
        support_gradient[0][0][missing_link_fact] < 0.0,
        "bridge must expose a direct local gradient to its missing support fact: {}",
        support_gradient[0][0][missing_link_fact]
    );
}

#[test]
fn backward_causal_flow_exposes_a_whole_chain_without_probability_decay() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 3, 1, device.clone()).expect("tensor plan");
    let noop = transcription.noop_action();
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
    let goal_fact = transcription.fact(0, 2) as usize;
    let prerequisite_fact = transcription.fact(0, 1) as usize;
    let mut demand = vec![0.0f64; plan.num_facts];
    demand[goal_fact] = 1.0;
    let demand = Tensor::from_vec(demand, (1, plan.num_facts), &device).expect("goal demand");
    let active = Tensor::ones((1, 3, 1), DTYPE, &device).expect("active suffix");
    let boundary =
        Tensor::from_vec(vec![1.0f64, 0.0, 0.0], (1, 3, 1), &device).expect("repair boundary");
    let links = link_tensor(
        &plan,
        &[
            // The terminal goal is supplied by action row one.
            (3, goal_fact, 2),
            // That action's prerequisite is supplied by action row zero.
            (1, prerequisite_fact, 1),
        ],
    );
    let link_temperature = Tensor::ones((1, 1, 1, 1), DTYPE, &device).expect("link temperature");
    let no_delete = Tensor::zeros((1, 3, plan.num_facts), DTYPE, &device).expect("no threats");

    let valid_logits = Tensor::from_vec(
        peaked_rows(3, plan.num_actions, &[0, 1, noop]),
        (1, 3, plan.num_actions),
        &device,
    )
    .expect("valid logits");
    let valid = plan
        .two_loss_forward_hardened(&valid_logits, &temperature, 1.0)
        .expect("valid chain");
    let valid_flow = plan
        .backward_causal_flow(
            &valid.action,
            &no_delete,
            &valid.exact_state_by_step,
            &demand,
            &active,
            &boundary,
            &links,
            &link_temperature,
            0,
        )
        .expect("valid causal flow");
    assert!(
        scalar(&valid_flow.loss) < 1e-20,
        "valid chain has zero flow loss"
    );
    let suffix_active =
        Tensor::from_vec(vec![0.0f64, 1.0, 1.0], (1, 3, 1), &device).expect("suffix-active rows");
    let suffix_boundary =
        Tensor::from_vec(vec![0.0f64, 1.0, 0.0], (1, 3, 1), &device).expect("suffix boundary");
    let valid_suffix_flow = plan
        .backward_causal_flow(
            &valid.action,
            &no_delete,
            &valid.exact_state_by_step,
            &demand,
            &suffix_active,
            &suffix_boundary,
            &links,
            &link_temperature,
            1,
        )
        .expect("nonzero-boundary causal flow accepts narrowed tensor views");
    assert!(scalar(&valid_suffix_flow.loss) < 1e-20);

    // Both required producer rows start completely undecided. The normalized
    // causal responsibilities must expose both ends of the two-action chain in
    // one backward pass instead of waiting for the later action to saturate.
    let logits = candle_core::Var::zeros((1, 3, plan.num_actions), DTYPE, &device)
        .expect("undecided action logits");
    let action = plan
        .action_distribution(logits.as_tensor(), &temperature)
        .expect("soft actions");
    let undecided = plan
        .two_loss_forward(logits.as_tensor(), &temperature)
        .expect("undecided recurrent states");
    let flow = plan
        .backward_causal_flow(
            &action,
            &no_delete,
            &undecided.exact_state_by_step,
            &demand,
            &active,
            &boundary,
            &links,
            &link_temperature,
            0,
        )
        .expect("repair causal flow");
    let gradient = flow
        .loss
        .sum_all()
        .expect("scalar flow loss")
        .backward()
        .expect("flow gradient")
        .get(&logits)
        .expect("action gradient")
        .to_vec3::<f64>()
        .expect("rank-three gradient");
    assert!(
        gradient[0][0][0] < gradient[0][0][noop],
        "earlier prerequisite producer receives a simultaneous preference: {:?}",
        gradient[0][0]
    );
    assert!(
        gradient[0][1][1] < gradient[0][1][noop],
        "later goal producer receives a simultaneous preference: {:?}",
        gradient[0][1]
    );
}

#[test]
fn temporal_tokens_move_action_identity_through_a_doubly_stochastic_schedule() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 4, 1, device.clone()).expect("tensor plan");
    let noop = transcription.noop_action();
    let action_temperature = Tensor::ones((1, 4, 1), DTYPE, &device).expect("action temperature");
    let schedule_temperature =
        Tensor::ones((1, 1, 1), DTYPE, &device).expect("schedule temperature");
    let token_logits = Tensor::from_vec(
        peaked_rows(4, plan.num_actions, &[0, 1, noop, 2]),
        (1, 4, plan.num_actions),
        &device,
    )
    .expect("token actions");

    // Token 2 (the no-op) moves left while token 1 keeps its action identity
    // and moves right. This is the continuous insertion primitive the direct
    // row parameterization lacks.
    let token_to_row = [0usize, 2, 1, 3];
    let mut schedule = vec![-30.0f64; 16];
    for (token, &row) in token_to_row.iter().enumerate() {
        schedule[token * 4 + row] = 30.0;
    }
    let schedule = Tensor::from_vec(schedule, (1, 4, 4), &device).expect("schedule");
    let temporal = plan
        .temporal_token_distribution(
            &token_logits,
            &schedule,
            &action_temperature,
            &schedule_temperature,
            &Tensor::ones((1, 1, 1), DTYPE, &device).expect("schedule gate"),
            12,
        )
        .expect("temporal tokens");
    let decoded = temporal
        .action
        .argmax(2)
        .expect("row argmax")
        .to_vec2::<u32>()
        .expect("decoded rows");
    assert_eq!(decoded[0], vec![0, noop as u32, 1, 2]);

    let assignment = temporal
        .assignment
        .to_vec3::<f64>()
        .expect("assignment matrix");
    for token in 0..4 {
        assert!((assignment[0][token].iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
    for row in 0..4 {
        let column_sum = (0..4).map(|token| assignment[0][token][row]).sum::<f64>();
        assert!((column_sum - 1.0).abs() < 1e-12);
    }
}

#[test]
fn temporal_token_schedule_has_a_direct_insertion_gradient() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 4, 1, device.clone()).expect("tensor plan");
    let noop = transcription.noop_action();
    let action_temperature = Tensor::ones((1, 4, 1), DTYPE, &device).expect("action temperature");
    let schedule_temperature =
        Tensor::ones((1, 1, 1), DTYPE, &device).expect("schedule temperature");
    let token_logits = Tensor::from_vec(
        peaked_rows(4, plan.num_actions, &[0, 1, noop, 2]),
        (1, 4, plan.num_actions),
        &device,
    )
    .expect("token actions");
    let mut identity = vec![0.0f64; 16];
    for row in 0..4 {
        identity[row * 4 + row] = 2.0;
    }
    let schedule = candle_core::Var::from_vec(identity, (1, 4, 4), &device).expect("schedule");
    let temporal = plan
        .temporal_token_distribution(
            &token_logits,
            schedule.as_tensor(),
            &action_temperature,
            &schedule_temperature,
            &Tensor::ones((1, 1, 1), DTYPE, &device).expect("schedule gate"),
            12,
        )
        .expect("temporal tokens");
    let loss = temporal
        .log_action
        .narrow(1, 1, 1)
        .expect("target row")
        .narrow(2, noop, 1)
        .expect("target noop")
        .neg()
        .expect("negative log likelihood")
        .sum_all()
        .expect("scalar loss");
    let gradient = loss
        .backward()
        .expect("schedule gradient")
        .get(&schedule)
        .expect("live schedule gradient")
        .to_vec3::<f64>()
        .expect("schedule gradient shape");
    assert!(
        gradient[0][2][1] < 0.0,
        "gradient descent must move the no-op token directly into the insertion row: {:?}",
        gradient[0][2]
    );
}

#[test]
fn locked_temporal_tokens_are_exactly_the_direct_plan() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 4, 1, device.clone()).expect("tensor plan");
    let action_temperature = Tensor::ones((1, 4, 1), DTYPE, &device).expect("temperature");
    let schedule_temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
    let token_logits = Tensor::from_vec(
        (0..4 * plan.num_actions)
            .map(|index| (index as f64 * 0.37).sin())
            .collect::<Vec<_>>(),
        (1, 4, plan.num_actions),
        &device,
    )
    .expect("token logits");
    let arbitrary_schedule = Tensor::from_vec(
        (0..16)
            .map(|index| (index as f64 * 1.13).cos() * 4.0)
            .collect::<Vec<_>>(),
        (1, 4, 4),
        &device,
    )
    .expect("schedule");
    let temporal = plan
        .temporal_token_distribution(
            &token_logits,
            &arbitrary_schedule,
            &action_temperature,
            &schedule_temperature,
            &Tensor::zeros((1, 1, 1), DTYPE, &device).expect("locked gate"),
            12,
        )
        .expect("locked temporal plan");
    let direct = plan
        .action_distribution(&token_logits, &action_temperature)
        .expect("direct plan");
    let maximum_error = (temporal.action - direct)
        .expect("difference")
        .abs()
        .expect("absolute difference")
        .max_all()
        .expect("maximum")
        .to_scalar::<f64>()
        .expect("scalar");
    assert!(maximum_error < 1e-12, "locked schedule changed the plan");
}

fn peaked_rows(rows: usize, width: usize, choices: &[usize]) -> Vec<f64> {
    assert_eq!(choices.len(), rows);
    let mut logits = vec![-30.0f64; rows * width];
    for (row, &choice) in choices.iter().enumerate() {
        assert!(choice < width);
        logits[row * width + choice] = 30.0;
    }
    logits
}

fn chain_forward(
    plan: &TensorPlan,
    transcription: &Transcription,
    action_choices: &[usize],
    state_values: &[usize],
) -> Forward {
    assert_eq!(plan.particles, 1);
    assert_eq!(action_choices.len(), plan.horizon);
    assert_eq!(state_values.len(), plan.horizon);
    let device = plan.device();
    let action_logits = Tensor::from_vec(
        peaked_rows(plan.horizon, plan.num_actions, action_choices),
        (1, plan.horizon, plan.num_actions),
        device,
    )
    .expect("action logits");
    let state_facts: Vec<usize> = state_values
        .iter()
        .map(|&value| transcription.fact(0, value) as usize)
        .collect();
    let state_logits = Tensor::from_vec(
        peaked_rows(plan.horizon, plan.num_facts, &state_facts),
        (1, plan.horizon, plan.num_facts),
        device,
    )
    .expect("state logits");
    let temperature = Tensor::ones((1, 1, 1), DTYPE, device).expect("temperature");
    plan.forward(&action_logits, &state_logits, &temperature, &temperature)
        .expect("chain forward")
}

fn link_logit_values(plan: &TensorPlan, selected: &[(usize, usize, usize)]) -> Vec<f64> {
    assert_eq!(plan.particles, 1);
    let [_, consumers, facts, sources] = plan.causal_link_shape();
    let mut logits = vec![-30.0f64; consumers * facts * sources];
    for &(consumer, fact, source) in selected {
        assert!(consumer < consumers && fact < facts && source < sources);
        logits[(consumer * facts + fact) * sources + source] = 30.0;
    }
    logits
}

fn link_tensor(plan: &TensorPlan, selected: &[(usize, usize, usize)]) -> Tensor {
    let shape = plan.causal_link_shape();
    Tensor::from_vec(link_logit_values(plan, selected), &shape, plan.device()).expect("link logits")
}

fn link_temperature(plan: &TensorPlan) -> Tensor {
    Tensor::ones((plan.particles, 1, 1, 1), DTYPE, plan.device()).expect("link temperature")
}

fn scalar(tensor: &Tensor) -> f64 {
    assert_eq!(tensor.elem_count(), 1, "expected exactly one value");
    tensor
        .sum_all()
        .expect("single-value reduction")
        .to_scalar::<f64>()
        .expect("scalar tensor")
}

#[test]
fn valid_last_achiever_links_have_near_zero_losses_and_exact_triangular_mask() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 3, 1, device).expect("tensor plan");
    let pos0 = transcription.fact(0, 0) as usize;
    let pos1 = transcription.fact(0, 1) as usize;
    let pos2 = transcription.fact(0, 2) as usize;
    let noop = plan.num_actions - 1;
    let forward = chain_forward(&plan, &transcription, &[0, 1, noop], &[1, 2, 2]);
    let logits = link_tensor(
        &plan,
        &[(0, pos0, 0), (1, pos1, 1), (plan.horizon, pos2, 2)],
    );
    let links = plan
        .causal_link_forward(&forward, &logits, &link_temperature(&plan))
        .expect("causal links");

    assert_eq!(links.link.dims(), &plan.causal_link_shape());
    assert!(scalar(&links.source_loss) < 1e-12);
    assert!(scalar(&links.threat_loss) < 1e-12);
    assert!(scalar(&links.link_integrality) < 1e-12);

    let values = to_vec4(&links.link);
    for consumer in 0..=plan.horizon {
        for fact in 0..plan.num_facts {
            for source in consumer + 1..=plan.horizon {
                assert_eq!(
                    values[0][consumer][fact][source], 0.0,
                    "future/self source {source} leaked into consumer {consumer}, fact {fact}"
                );
            }
        }
    }
}

#[test]
fn recurrent_causal_input_proves_the_same_valid_chain_without_state_logits() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 3, 1, device.clone()).expect("tensor plan");
    let pos0 = transcription.fact(0, 0) as usize;
    let pos1 = transcription.fact(0, 1) as usize;
    let pos2 = transcription.fact(0, 2) as usize;
    let noop = plan.num_actions - 1;
    let action_logits = Tensor::from_vec(
        peaked_rows(plan.horizon, plan.num_actions, &[0, 1, noop]),
        (1, plan.horizon, plan.num_actions),
        &device,
    )
    .expect("action logits");
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
    let recurrent = plan
        .two_loss_forward(&action_logits, &temperature)
        .expect("recurrent forward");
    let links = plan
        .causal_link_forward_from_input(
            &recurrent.causal,
            &link_tensor(
                &plan,
                &[(0, pos0, 0), (1, pos1, 1), (plan.horizon, pos2, 2)],
            ),
            &link_temperature(&plan),
        )
        .expect("recurrent causal links");

    assert!(scalar(&links.source_loss) < 1e-12);
    assert!(scalar(&links.threat_loss) < 1e-12);
    assert!(scalar(&links.link_integrality) < 1e-12);

    let roundoff_input = CausalLinkInput {
        action_demand: ((&recurrent.causal.action_demand * (1.0 + 4e-16)).unwrap() - 2e-16)
            .unwrap(),
        add: ((&recurrent.causal.add * (1.0 + 4e-16)).unwrap() - 2e-16).unwrap(),
        delete: ((&recurrent.causal.delete * (1.0 + 4e-16)).unwrap() - 2e-16).unwrap(),
    };
    plan.causal_link_forward_from_input(
        &roundoff_input,
        &link_tensor(
            &plan,
            &[(0, pos0, 0), (1, pos1, 1), (plan.horizon, pos2, 2)],
        ),
        &link_temperature(&plan),
    )
    .expect("ulp-scale probability drift is projected to the semantic interval");

    let invalid_input = CausalLinkInput {
        action_demand: (&recurrent.causal.action_demand + 0.01).unwrap(),
        add: recurrent.causal.add.clone(),
        delete: recurrent.causal.delete.clone(),
    };
    assert!(
        plan.causal_link_forward_from_input(
            &invalid_input,
            &link_tensor(
                &plan,
                &[(0, pos0, 0), (1, pos1, 1), (plan.horizon, pos2, 2)],
            ),
            &link_temperature(&plan),
        )
        .is_err(),
        "materially invalid probabilities must still fail fast"
    );
}

#[test]
fn missing_sources_and_intervening_deletes_are_positive() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 3, 1, device).expect("tensor plan");
    let pos0 = transcription.fact(0, 0) as usize;
    let pos1 = transcription.fact(0, 1) as usize;
    let pos2 = transcription.fact(0, 2) as usize;
    let noop = plan.num_actions - 1;

    let valid_forward = chain_forward(&plan, &transcription, &[0, 1, noop], &[1, 2, 2]);
    let missing_logits = link_tensor(
        &plan,
        &[
            (0, pos0, 0),
            (1, pos1, 1),
            // The initial state does not contain the terminal goal.
            (plan.horizon, pos2, 0),
        ],
    );
    let missing = plan
        .causal_link_forward(&valid_forward, &missing_logits, &link_temperature(&plan))
        .expect("missing-source links");
    assert!(
        scalar(&missing.source_loss) > 0.2,
        "missing producer was not charged: {}",
        scalar(&missing.source_loss)
    );
    let max_source = scalar(&missing.max_source_violation);
    assert!(
        max_source > 2.9 * scalar(&missing.source_loss),
        "one missing source was diluted: max={max_source}, normalized={}",
        scalar(&missing.source_loss)
    );
    assert!(max_source > 10.0, "missing source maximum is {max_source}");

    // Row 1 deletes pos1 after row 0 establishes it. Naming row 0 as the
    // source of row 2's pos1 precondition must therefore expose a threat.
    let threatened_forward = chain_forward(&plan, &transcription, &[0, 2, 1], &[1, 0, 2]);
    let threatened_logits = link_tensor(
        &plan,
        &[
            (0, pos0, 0),
            (1, pos1, 1),
            (2, pos1, 1),
            (plan.horizon, pos2, 3),
        ],
    );
    let threatened = plan
        .causal_link_forward(
            &threatened_forward,
            &threatened_logits,
            &link_temperature(&plan),
        )
        .expect("threatened links");
    assert!(
        scalar(&threatened.threat_loss) > 0.2,
        "intervening delete was not charged: {}",
        scalar(&threatened.threat_loss)
    );
    let max_threat = scalar(&threatened.max_threat_violation);
    assert!(
        max_threat > 3.9 * scalar(&threatened.threat_loss),
        "one intervening threat was diluted: max={max_threat}, normalized={}",
        scalar(&threatened.threat_loss)
    );
    assert!(max_threat > 1.0 - 1e-12, "threat maximum is {max_threat}");
}

#[test]
fn terminal_goal_demand_is_constant_even_when_the_soft_goal_is_false() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 3, 1, device).expect("tensor plan");
    let pos2 = transcription.fact(0, 2) as usize;
    let noop = plan.num_actions - 1;
    let forward = chain_forward(&plan, &transcription, &[noop, noop, noop], &[0, 0, 0]);
    assert!(to_vec3(&forward.goal)[0][0][0] > 1.0 - 1e-12);

    let links = plan
        .causal_link_forward(&forward, &link_tensor(&plan, &[]), &link_temperature(&plan))
        .expect("causal links");
    let demand = to_vec3(&links.demand);
    for fact in 0..plan.num_facts {
        let expected = f64::from(fact == pos2);
        assert_eq!(demand[0][plan.horizon][fact], expected);
    }
}

#[test]
fn exact_goal_weight_scales_causal_pressure_instead_of_normalizing_away() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 3, 1, device.clone()).expect("tensor plan");
    let pos2 = transcription.fact(0, 2) as usize;
    let noop = plan.num_actions - 1;
    let forward = chain_forward(&plan, &transcription, &[noop, noop, noop], &[0, 0, 0]);
    let link_logits = link_tensor(&plan, &[]);
    let temperature = link_temperature(&plan);
    let baseline = plan
        .causal_link_forward(&forward, &link_logits, &temperature)
        .expect("baseline links");

    let mut extra_values = vec![0.0f64; (plan.horizon + 1) * plan.num_facts];
    extra_values[plan.horizon * plan.num_facts + pos2] = 9.0;
    let extra = Tensor::from_vec(extra_values, (1, plan.horizon + 1, plan.num_facts), &device)
        .expect("extra demand");
    let focused = plan
        .causal_link_forward_with_demand(&forward, &link_logits, &temperature, &extra)
        .expect("focused links");

    let baseline_loss = scalar(&baseline.source_loss);
    let focused_loss = scalar(&focused.source_loss);
    assert!(
        (focused_loss - 10.0 * baseline_loss).abs() < 1e-10,
        "tenfold exact demand should create tenfold source pressure, got \
         baseline={baseline_loss}, focused={focused_loss}"
    );
}

#[test]
fn log_source_support_backpropagates_to_producer_actions_and_link_logits() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 3, 1, device.clone()).expect("tensor plan");
    let pos0 = transcription.fact(0, 0) as usize;
    let pos1 = transcription.fact(0, 1) as usize;
    let pos2 = transcription.fact(0, 2) as usize;
    let noop = plan.num_actions - 1;

    // Row 1 supplies only fractional goal mass, while the terminal link asks
    // it for the entire goal. Log source support must pull up that producer.
    let p = 0.35f64;
    let mut action_values = vec![-30.0f64; plan.horizon * plan.num_actions];
    action_values[0] = 30.0;
    action_values[plan.num_actions + 1] = (p / (1.0 - p)).ln();
    action_values[plan.num_actions + noop] = 0.0;
    action_values[2 * plan.num_actions + noop] = 30.0;
    let action_logits =
        candle_core::Var::from_vec(action_values, (1, plan.horizon, plan.num_actions), &device)
            .expect("variable action logits");
    let state_facts = [pos1, pos2, pos2];
    let state_logits = Tensor::from_vec(
        peaked_rows(plan.horizon, plan.num_facts, &state_facts),
        (1, plan.horizon, plan.num_facts),
        &device,
    )
    .expect("state logits");
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
    let forward = plan
        .forward(
            action_logits.as_tensor(),
            &state_logits,
            &temperature,
            &temperature,
        )
        .expect("fractional producer forward");
    let links = plan
        .causal_link_forward(
            &forward,
            &link_tensor(
                &plan,
                &[(0, pos0, 0), (1, pos1, 1), (plan.horizon, pos2, 2)],
            ),
            &link_temperature(&plan),
        )
        .expect("fractional producer links");
    let gradients = links.source_loss.backward().expect("action backward");
    let action_gradient = to_vec3(
        gradients
            .get(&action_logits)
            .expect("gradient for producer action logits"),
    );
    assert!(
        action_gradient[0][1][1] < -1e-3,
        "producer action gradient should increase its logit, got {}",
        action_gradient[0][1][1]
    );

    // With an integral producer but a uniform terminal link, source loss must
    // move link mass away from nonexistent sources and toward row 1.
    let integral_forward = chain_forward(&plan, &transcription, &[0, 1, noop], &[1, 2, 2]);
    let mut link_values = link_logit_values(&plan, &[(0, pos0, 0), (1, pos1, 1)]);
    let sources = plan.horizon + 1;
    for source in 0..sources {
        link_values[(plan.horizon * plan.num_facts + pos2) * sources + source] = 0.0;
    }
    let shape = plan.causal_link_shape();
    let link_logits =
        candle_core::Var::from_vec(link_values, &shape, &device).expect("variable link logits");
    let links = plan
        .causal_link_forward(
            &integral_forward,
            link_logits.as_tensor(),
            &link_temperature(&plan),
        )
        .expect("uniform terminal links");
    let gradients = links.source_loss.backward().expect("link backward");
    let link_gradient = to_vec4(
        gradients
            .get(&link_logits)
            .expect("gradient for link logits"),
    );
    assert!(
        link_gradient[0][plan.horizon][pos2][2] < -1e-3,
        "last achiever link should be increased, got {}",
        link_gradient[0][plan.horizon][pos2][2]
    );
    assert!(
        link_gradient[0][plan.horizon][pos2][0] > 1e-3,
        "missing initial source should be decreased, got {}",
        link_gradient[0][plan.horizon][pos2][0]
    );
}

#[test]
fn causal_losses_are_normalized_independently_for_two_particles() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 3, 2, device.clone()).expect("tensor plan");
    let pos0 = transcription.fact(0, 0) as usize;
    let pos1 = transcription.fact(0, 1) as usize;
    let pos2 = transcription.fact(0, 2) as usize;
    let noop = plan.num_actions - 1;

    let one_action = peaked_rows(plan.horizon, plan.num_actions, &[0, 1, noop]);
    let mut action_values = one_action.clone();
    action_values.extend_from_slice(&one_action);
    let action_logits = Tensor::from_vec(
        action_values,
        (plan.particles, plan.horizon, plan.num_actions),
        &device,
    )
    .expect("two-particle action logits");
    let state_facts = [pos1, pos2, pos2];
    let one_state = peaked_rows(plan.horizon, plan.num_facts, &state_facts);
    let mut state_values = one_state.clone();
    state_values.extend_from_slice(&one_state);
    let state_logits = Tensor::from_vec(
        state_values,
        (plan.particles, plan.horizon, plan.num_facts),
        &device,
    )
    .expect("two-particle state logits");
    let temperature = Tensor::ones((plan.particles, 1, 1), DTYPE, &device).expect("temperature");
    let forward = plan
        .forward(&action_logits, &state_logits, &temperature, &temperature)
        .expect("two-particle forward");

    let [particles, consumers, facts, sources] = plan.causal_link_shape();
    let mut link_values = vec![-30.0f64; particles * consumers * facts * sources];
    let mut select = |particle: usize, consumer: usize, fact: usize, source: usize| {
        link_values[((particle * consumers + consumer) * facts + fact) * sources + source] = 30.0;
    };
    for particle in 0..particles {
        select(particle, 0, pos0, 0);
        select(particle, 1, pos1, 1);
    }
    // Particle zero names the real row-1 achiever. Particle one names the
    // false initial goal fact; its violation must not be divided by particle
    // zero's active demand.
    select(0, plan.horizon, pos2, 2);
    select(1, plan.horizon, pos2, 0);
    let link_logits =
        Tensor::from_vec(link_values, &plan.causal_link_shape(), &device).expect("link logits");
    let link_temperature =
        Tensor::ones((particles, 1, 1, 1), DTYPE, &device).expect("link temperature");
    let links = plan
        .causal_link_forward(&forward, &link_logits, &link_temperature)
        .expect("two-particle causal links");

    for output in [
        &links.source_loss,
        &links.threat_loss,
        &links.link_integrality,
        &links.max_source_violation,
        &links.max_threat_violation,
        &links.active_consumer_mass,
    ] {
        assert_eq!(output.dims(), &[particles, 1]);
    }
    let source = links.source_loss.to_vec2::<f64>().expect("source loss");
    assert!(source[0][0] < 1e-12);
    assert!(
        source[1][0] > 0.3,
        "particle-local normalization was diluted to {}",
        source[1][0]
    );
    let active = links
        .active_consumer_mass
        .to_vec2::<f64>()
        .expect("active mass");
    assert!((active[0][0] - active[1][0]).abs() < 1e-12);
}

#[test]
fn some_producer_probability_uses_all_horizon_rows() {
    let device = cpu();
    let transcription = Transcription::build(&causal_chain_task()).expect("chain transcription");
    let plan = TensorPlan::new(&transcription, 3, 1, device.clone()).expect("tensor plan");
    let noop = plan.num_actions - 1;
    let probabilities = [0.2f64, 0.3f64];
    let mut values = vec![-30.0f64; plan.horizon * plan.num_actions];
    for (row, &probability) in probabilities.iter().enumerate() {
        values[row * plan.num_actions + 1] = (probability / (1.0 - probability)).ln();
        values[row * plan.num_actions + noop] = 0.0;
    }
    values[2 * plan.num_actions + noop] = 30.0;
    let logits = Tensor::from_vec(values, (1, plan.horizon, plan.num_actions), &device)
        .expect("action logits");
    let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
    let forward = plan
        .two_loss_forward(&logits, &temperature)
        .expect("two-loss forward");
    let action = to_vec3(&forward.action);
    let expected = 1.0
        - (0..plan.horizon)
            .map(|row| 1.0 - action[0][row][1])
            .product::<f64>();
    let got = forward
        .some_goal_producer_probability
        .to_vec2::<f64>()
        .expect("producer probability")[0][0];
    assert!(
        (got - expected).abs() < 1e-12,
        "q_g is {got}, expected {expected}"
    );
    let loss = forward
        .some_goal_producer_loss
        .to_vec2::<f64>()
        .expect("producer loss")[0][0];
    assert!((loss + expected.ln()).abs() < 1e-12);
}
