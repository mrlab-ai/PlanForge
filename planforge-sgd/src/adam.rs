//! Adam, hand-written so that moments can be reset on a slice.
//!
//! `candle_nn::AdamW` cannot do that, and plateau escape needs it: when a
//! particle stalls, the fix is to remelt one `(particle, time)` window — raise
//! its temperature, perturb its logits — and a stale second moment there would
//! damp the very exploration the remelt is trying to buy.
//!
//! Weight decay is deliberately absent rather than set to zero: these are plan
//! logits, not model weights, and shrinking them toward zero has no meaning.

use candle_core::{Result as CandleResult, Tensor, Var, backprop::GradStore};

/// Adam hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct AdamParams {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    /// Gradient-norm clip. In global mode it is applied across all parameters
    /// jointly; in per-particle mode it is applied across all parameters of
    /// each particle jointly.
    pub grad_clip: f64,
    /// Enable clipping independently along the leading particle axis.
    ///
    /// Every optimized variable must then have this exact leading dimension.
    /// Keeping the count explicit prevents a rank-compatible but semantically
    /// unrelated tensor from silently being treated as a population.
    pub particles: Option<usize>,
}

impl Default for AdamParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.04,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            grad_clip: 30.0,
            particles: None,
        }
    }
}

struct Slot {
    variable: Var,
    first_moment: Tensor,
    second_moment: Tensor,
    /// Elementwise β powers. They are elementwise, rather than one global
    /// update counter, because a local remelt resets only part of a tensor.
    first_power: Tensor,
    second_power: Tensor,
}

pub struct Adam {
    params: AdamParams,
    slots: Vec<Slot>,
}

impl Adam {
    pub fn new(variables: Vec<Var>, params: AdamParams) -> CandleResult<Self> {
        if let Some(particles) = params.particles {
            if particles == 0 {
                candle_core::bail!("Adam per-particle clipping requires at least one particle");
            }
            for variable in &variables {
                let dims = variable.dims();
                if dims.is_empty() || dims[0] != particles {
                    candle_core::bail!(
                        "Adam per-particle clipping requires leading dimension {particles}, got {:?}",
                        dims
                    );
                }
            }
        }
        let slots = variables
            .into_iter()
            .map(|variable| {
                let first_moment =
                    Tensor::zeros(variable.shape(), variable.dtype(), variable.device())?;
                let second_moment =
                    Tensor::zeros(variable.shape(), variable.dtype(), variable.device())?;
                let first_power =
                    Tensor::ones(variable.shape(), variable.dtype(), variable.device())?;
                let second_power =
                    Tensor::ones(variable.shape(), variable.dtype(), variable.device())?;
                Ok(Slot {
                    variable,
                    first_moment,
                    second_moment,
                    first_power,
                    second_power,
                })
            })
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(Self { params, slots })
    }

    /// Global gradient norm over every parameter, returned as a progress
    /// diagnostic even when clipping uses separate particle norms.
    fn global_norm(&self, grads: &GradStore) -> CandleResult<f64> {
        let mut total = 0f64;
        for slot in &self.slots {
            if let Some(gradient) = grads.get(&slot.variable) {
                total += gradient.sqr()?.sum_all()?.to_scalar::<f64>()?;
            }
        }
        Ok(total.sqrt())
    }

    /// Scale factors for a joint norm over every parameter belonging to each
    /// particle. The output is `[particles]`, one factor per leading slice.
    fn per_particle_clip_scales(
        &self,
        grads: &GradStore,
        particles: usize,
    ) -> CandleResult<Tensor> {
        debug_assert!(particles > 0, "validated when Adam was constructed");
        let first = self
            .slots
            .first()
            .expect("an Adam optimizer must own at least one variable");
        let mut squared_norm =
            Tensor::zeros(particles, first.variable.dtype(), first.variable.device())?;
        for slot in &self.slots {
            let Some(gradient) = grads.get(&slot.variable) else {
                continue;
            };
            let dims = gradient.dims();
            if dims.is_empty() || dims[0] != particles {
                candle_core::bail!(
                    "Adam per-particle gradient has leading dimension {:?}, expected {particles}",
                    dims.first()
                );
            }
            let contribution = if dims.len() == 1 {
                gradient.sqr()?
            } else {
                gradient.sqr()?.flatten_from(1)?.sum(1)?
            };
            squared_norm = (&squared_norm + contribution)?;
        }
        let norm = squared_norm.sqrt()?;
        // `max(norm, clip)` makes the zero-gradient case exactly scale one,
        // without evaluating a division by zero behind a conditional.
        let clip = (norm.ones_like()? * self.params.grad_clip)?;
        clip.broadcast_div(&norm.maximum(&clip)?)
    }

    /// One Adam step. Returns the pre-clip global gradient norm.
    pub fn step(&mut self, grads: &GradStore) -> CandleResult<f64> {
        let norm = self.global_norm(grads)?;
        let particle_scales = match (self.params.particles, self.params.grad_clip > 0.0) {
            (Some(particles), true) => Some(self.per_particle_clip_scales(grads, particles)?),
            _ => None,
        };
        let global_scale = if particle_scales.is_none()
            && self.params.grad_clip > 0.0
            && norm > self.params.grad_clip
        {
            self.params.grad_clip / norm
        } else {
            1.0
        };

        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;

        for slot in self.slots.iter_mut() {
            let Some(gradient) = grads.get(&slot.variable) else {
                continue;
            };
            // Detach before anything is stored. A gradient tensor still carries
            // the autograd graph of the step that produced it, so keeping a
            // moment derived from it alive would pin that whole graph and every
            // iteration would chain onto the last.
            let gradient = if let Some(scales) = &particle_scales {
                let mut scale_dims = vec![scales.dim(0)?];
                scale_dims.extend(std::iter::repeat_n(1, gradient.rank() - 1));
                gradient
                    .detach()
                    .broadcast_mul(&scales.reshape(scale_dims)?)?
            } else {
                (gradient.detach() * global_scale)?
            };

            let next_first =
                ((&slot.first_moment * beta1)? + (&gradient * (1.0 - beta1))?)?.detach();
            let next_second =
                ((&slot.second_moment * beta2)? + (gradient.sqr()? * (1.0 - beta2))?)?.detach();
            let next_first_power = (&slot.first_power * beta1)?.detach();
            let next_second_power = (&slot.second_power * beta2)?.detach();

            let corrected_first =
                (&next_first / (next_first_power.ones_like()? - &next_first_power)?)?;
            let corrected_second =
                (&next_second / (next_second_power.ones_like()? - &next_second_power)?)?;
            let update = (corrected_first / (corrected_second.sqrt()? + self.params.epsilon)?)?;
            let next = (slot.variable.as_tensor() - (update * self.params.learning_rate)?)?;
            slot.variable.set(&next.detach())?;

            slot.first_moment = next_first;
            slot.second_moment = next_second;
            slot.first_power = next_first_power;
            slot.second_power = next_second_power;
        }
        Ok(norm)
    }

    /// Zero both moments wherever `keep` is zero.
    ///
    /// Every `keep` mask must have exactly its parameter's shape and contain 0
    /// on the remelted coordinates and 1 elsewhere. Requiring exact shapes
    /// prevents an accidentally broad reset from silently broadcasting.
    pub fn reset_moments_where(&mut self, keep: &[Tensor]) -> CandleResult<()> {
        if keep.len() != self.slots.len() {
            candle_core::bail!(
                "expected one mask per parameter ({}), got {}",
                self.slots.len(),
                keep.len()
            );
        }
        for (slot, mask) in self.slots.iter_mut().zip(keep) {
            if mask.dims() != slot.variable.dims() {
                candle_core::bail!(
                    "Adam reset mask shape {:?} does not match parameter shape {:?}",
                    mask.dims(),
                    slot.variable.dims()
                );
            }
            slot.first_moment = (&slot.first_moment * mask)?;
            slot.second_moment = (&slot.second_moment * mask)?;
            let reset = (mask.ones_like()? - mask)?;
            slot.first_power = ((&slot.first_power * mask)? + &reset)?.detach();
            slot.second_power = ((&slot.second_power * mask)? + reset)?.detach();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn a_reset_slice_gets_fresh_bias_correction_at_late_optimizer_age() {
        let device = Device::Cpu;
        let variable = Var::zeros(2, DType::F64, &device).expect("variable");
        let learning_rate = 0.1;
        let mut optimizer = Adam::new(
            vec![variable.clone()],
            AdamParams {
                learning_rate,
                grad_clip: 0.0,
                ..AdamParams::default()
            },
        )
        .expect("optimizer");

        for _ in 0..100 {
            let loss = variable.as_tensor().sum_all().expect("loss");
            optimizer
                .step(&loss.backward().expect("gradient"))
                .expect("step");
        }
        optimizer
            .reset_moments_where(&[Tensor::from_vec(vec![0.0f64, 1.0], 2, &device).expect("mask")])
            .expect("partial reset");

        let before = variable.as_tensor().to_vec1::<f64>().expect("before");
        let loss = variable.as_tensor().sum_all().expect("loss");
        optimizer
            .step(&loss.backward().expect("gradient"))
            .expect("step");
        let after = variable.as_tensor().to_vec1::<f64>().expect("after");

        let deltas = [before[0] - after[0], before[1] - after[1]];
        let expected = learning_rate / (1.0 + AdamParams::default().epsilon);
        for (index, delta) in deltas.into_iter().enumerate() {
            assert!(
                (delta - expected).abs() < 1e-10,
                "element {index} moved {delta}; a fresh and a continuously-aged \
                 constant-gradient Adam element must both move {expected}"
            );
        }
        assert!(
            (deltas[0] - deltas[1]).abs() < 1e-12,
            "the reset and continuously-aged slices must receive the same step"
        );
    }

    #[test]
    fn per_particle_clipping_does_not_shrink_another_particles_gradient() {
        let device = Device::Cpu;
        let variable = Var::zeros((2, 1), DType::F64, &device).expect("variable");
        let mut optimizer = Adam::new(
            vec![variable.clone()],
            AdamParams {
                learning_rate: 1.0,
                beta1: 0.0,
                beta2: 0.0,
                epsilon: 1.0,
                grad_clip: 2.0,
                particles: Some(2),
            },
        )
        .expect("optimizer");
        let coefficients =
            Tensor::from_vec(vec![100.0f64, 1.0], (2, 1), &device).expect("coefficients");
        let loss = variable
            .as_tensor()
            .broadcast_mul(&coefficients)
            .expect("weighted loss")
            .sum_all()
            .expect("scalar loss");
        optimizer
            .step(&loss.backward().expect("gradient"))
            .expect("step");

        let values = variable.as_tensor().to_vec2::<f64>().expect("values");
        // Particle 0 has norm 100 and is clipped to gradient 2, whereas
        // particle 1 retains gradient 1. With beta=0 and epsilon=1 their
        // respective Adam updates are 2/(2+1) and 1/(1+1).
        assert!((values[0][0] + 2.0 / 3.0).abs() < 1e-12);
        assert!((values[1][0] + 1.0 / 2.0).abs() < 1e-12);
    }

    #[test]
    fn per_particle_clipping_is_joint_over_all_parameters() {
        let device = Device::Cpu;
        let first = Var::zeros((2, 1), DType::F64, &device).expect("first variable");
        let second = Var::zeros((2, 1), DType::F64, &device).expect("second variable");
        let mut optimizer = Adam::new(
            vec![first.clone(), second.clone()],
            AdamParams {
                learning_rate: 1.0,
                beta1: 0.0,
                beta2: 0.0,
                epsilon: 1.0,
                grad_clip: 4.0,
                particles: Some(2),
            },
        )
        .expect("optimizer");
        let first_coefficients =
            Tensor::from_vec(vec![3.0f64, 0.0], (2, 1), &device).expect("first coefficients");
        let second_coefficients =
            Tensor::from_vec(vec![4.0f64, 1.0], (2, 1), &device).expect("second coefficients");
        let loss = (first
            .as_tensor()
            .broadcast_mul(&first_coefficients)
            .expect("first loss")
            + second
                .as_tensor()
                .broadcast_mul(&second_coefficients)
                .expect("second loss"))
        .expect("combined loss")
        .sum_all()
        .expect("scalar loss");
        optimizer
            .step(&loss.backward().expect("gradient"))
            .expect("step");

        let first_values = first.as_tensor().to_vec2::<f64>().expect("first values");
        let second_values = second.as_tensor().to_vec2::<f64>().expect("second values");
        // Particle 0's joint norm is sqrt(3² + 4²) = 5, so both tensors use
        // the same 4/5 scale. Particle 1's norm is one and remains unclipped.
        assert!((first_values[0][0] + 2.4 / 3.4).abs() < 1e-12);
        assert!((second_values[0][0] + 3.2 / 4.2).abs() < 1e-12);
        assert!((second_values[1][0] + 1.0 / 2.0).abs() < 1e-12);
    }

    #[test]
    fn per_particle_clipping_rejects_a_variable_without_the_particle_axis() {
        let device = Device::Cpu;
        let variable = Var::zeros(3, DType::F64, &device).expect("variable");
        let result = Adam::new(
            vec![variable],
            AdamParams {
                particles: Some(2),
                ..AdamParams::default()
            },
        );
        let error = match result {
            Ok(_) => panic!("mismatched leading dimension must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("leading dimension 2"));
    }
}
