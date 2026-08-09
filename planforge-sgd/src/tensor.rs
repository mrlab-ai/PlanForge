//! The differentiable transcription, as candle tensors.
//!
//! Everything here mirrors [`crate::residuals`] exactly; that module is the
//! reference semantics and this one is differentially tested against it. If the
//! two ever disagree, the `f64` version is right by definition.
//!
//! # Layout
//!
//! Two parameter tensors, batched over particles:
//!
//! * `Z : [M, H, A]` — action logits;
//! * `U : [M, H, F]` — state logits for rows `1..=H`. Row 0 is a constant, so
//!   the initial state is fixed by construction rather than by a penalty.
//!
//! Facts are kept in one flat list rather than padded to a rectangular
//! `[variables, max_domain]`, because domain sizes are small and uneven (blocks
//! has both 2 and 5). Everything that needs per-variable structure goes through
//! precomputed index tensors and one segmented-sum op.
//!
//! # Segmented reductions
//!
//! [`SegSum`] computes `out[..., s] = Σ_{i : seg[i] = s} x[..., i]` with the
//! gradient `grad_x[..., i] = grad_out[..., seg[i]]`. With it:
//!
//! * a **segmented softmax** gives the per-variable state distributions;
//! * **`Add` and `Chg`** are the same per-effect contributions summed under two
//!   different segmentations.
//!
//! [`SegProd`] computes exact products, including empty products and exact zero
//! factors. It implements action-precondition conjunctions, conditional-effect
//! conjunctions, and last-write-wins suffix products without the positive leak
//! introduced by clamping before `log`.
//!
//! Cost is `O(nnz)` of the incidence structure — no dense `A × F` product
//! appears anywhere.

use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, DType, Device, Layout, Result as CandleResult, Shape, Tensor,
};
use std::collections::BTreeSet;

use crate::transcription::Transcription;

/// The dtype used throughout. `f64` is deliberate: residuals are bounded by one
/// but the segmented softmax is computed unnormalized, and `f64` plus logit
/// clipping keeps that safe without a separate segment-max pass.
pub const DTYPE: DType = DType::F64;

/// Segmented sum over the last dimension.
#[derive(Debug, Clone)]
pub struct SegSum {
    num_segments: usize,
    segment_of: Vec<u32>,
}

impl SegSum {
    pub fn new(num_segments: usize, segment_of: Vec<u32>) -> Self {
        debug_assert!(
            segment_of.iter().all(|&s| (s as usize) < num_segments),
            "segment id out of range"
        );
        Self {
            num_segments,
            segment_of,
        }
    }
}

impl CustomOp1 for SegSum {
    fn name(&self) -> &'static str {
        "seg-sum"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> CandleResult<(CpuStorage, Shape)> {
        let dims = layout.shape().dims();
        let inner = *dims.last().expect("seg-sum needs at least one dimension");
        if inner != self.segment_of.len() {
            candle_core::bail!(
                "seg-sum: last dimension {inner} does not match {} segment ids",
                self.segment_of.len()
            );
        }
        let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("seg-sum requires a contiguous input".to_string())
        })?;
        let source = match storage {
            CpuStorage::F64(values) => &values[start..end],
            _ => candle_core::bail!("seg-sum expects an f64 input"),
        };

        // Derive the row count from the leading dimensions rather than dividing
        // by `inner`: a task whose operators have no effects at all gives a
        // legitimately empty last dimension, and every segment is then zero.
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let mut out = vec![0f64; rows * self.num_segments];
        for row in 0..rows {
            let src = &source[row * inner..(row + 1) * inner];
            let dst = &mut out[row * self.num_segments..(row + 1) * self.num_segments];
            for (index, &value) in src.iter().enumerate() {
                dst[self.segment_of[index] as usize] += value;
            }
        }

        let mut shape = dims.to_vec();
        *shape.last_mut().expect("checked above") = self.num_segments;
        Ok((CpuStorage::F64(out), Shape::from(shape)))
    }

    fn bwd(&self, _arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> CandleResult<Option<Tensor>> {
        // Each input slot contributed to exactly one segment, so the gradient is
        // a gather of the output gradient.
        let indices = Tensor::from_slice(&self.segment_of, self.segment_of.len(), res.device())?;
        Ok(Some(grad_res.index_select(&indices, grad_res.rank() - 1)?))
    }
}

/// Segmented product over the last dimension.
///
/// Empty segments have product one. The custom backward is important at the
/// boundary: `exp(sum(log(clamp(x))))` leaks positive mass through an exact
/// zero, while `product / x` loses the correct derivative when exactly one
/// factor is zero. Planning states contain exact zeros in the fixed initial row,
/// so both shortcuts give the wrong semantics where it matters most.
#[derive(Debug, Clone)]
pub struct SegProd {
    num_segments: usize,
    segment_of: Vec<u32>,
}

impl SegProd {
    pub fn new(num_segments: usize, segment_of: Vec<u32>) -> Self {
        debug_assert!(
            segment_of.iter().all(|&s| (s as usize) < num_segments),
            "segment id out of range"
        );
        Self {
            num_segments,
            segment_of,
        }
    }
}

impl CustomOp1 for SegProd {
    fn name(&self) -> &'static str {
        "seg-prod"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> CandleResult<(CpuStorage, Shape)> {
        let dims = layout.shape().dims();
        let inner = *dims.last().expect("seg-prod needs at least one dimension");
        if inner != self.segment_of.len() {
            candle_core::bail!(
                "seg-prod: last dimension {inner} does not match {} segment ids",
                self.segment_of.len()
            );
        }
        let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("seg-prod requires a contiguous input".to_string())
        })?;
        let source = match storage {
            CpuStorage::F64(values) => &values[start..end],
            _ => candle_core::bail!("seg-prod expects an f64 input"),
        };

        let rows: usize = dims[..dims.len() - 1].iter().product();
        let mut out = vec![1f64; rows * self.num_segments];
        for row in 0..rows {
            let src = &source[row * inner..(row + 1) * inner];
            let dst = &mut out[row * self.num_segments..(row + 1) * self.num_segments];
            for (index, &value) in src.iter().enumerate() {
                dst[self.segment_of[index] as usize] *= value;
            }
        }

        let mut shape = dims.to_vec();
        *shape.last_mut().expect("checked above") = self.num_segments;
        Ok((CpuStorage::F64(out), Shape::from(shape)))
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> CandleResult<Option<Tensor>> {
        Ok(Some(arg.apply_op2_no_bwd(
            &grad_res.contiguous()?,
            &SegProdBackward {
                num_segments: self.num_segments,
                segment_of: self.segment_of.clone(),
            },
        )?))
    }
}

/// First derivative of [`SegProd`].
///
/// This is a custom binary op because the zero cases cannot be expressed as
/// `product / factor`: with one zero, the derivative at that zero is the
/// product of all nonzero factors; with two zeros, every derivative is zero.
#[derive(Debug, Clone)]
struct SegProdBackward {
    num_segments: usize,
    segment_of: Vec<u32>,
}

impl CustomOp2 for SegProdBackward {
    fn name(&self) -> &'static str {
        "seg-prod-backward"
    }

    fn cpu_fwd(
        &self,
        arg_storage: &CpuStorage,
        arg_layout: &Layout,
        grad_storage: &CpuStorage,
        grad_layout: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        let arg_dims = arg_layout.shape().dims();
        let grad_dims = grad_layout.shape().dims();
        let inner = *arg_dims
            .last()
            .expect("seg-prod backward needs an input dimension");
        let grad_inner = *grad_dims
            .last()
            .expect("seg-prod backward needs a gradient dimension");
        if inner != self.segment_of.len() {
            candle_core::bail!(
                "seg-prod backward: input dimension {inner} does not match {} segment ids",
                self.segment_of.len()
            );
        }
        if grad_inner != self.num_segments
            || arg_dims[..arg_dims.len() - 1] != grad_dims[..grad_dims.len() - 1]
        {
            candle_core::bail!(
                "seg-prod backward: incompatible input {:?} and gradient {:?}",
                arg_dims,
                grad_dims
            );
        }
        let (arg_start, arg_end) = arg_layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("seg-prod backward requires a contiguous input".to_string())
        })?;
        let (grad_start, grad_end) = grad_layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("seg-prod backward requires a contiguous gradient".to_string())
        })?;
        let args = match arg_storage {
            CpuStorage::F64(values) => &values[arg_start..arg_end],
            _ => candle_core::bail!("seg-prod backward expects an f64 input"),
        };
        let grads = match grad_storage {
            CpuStorage::F64(values) => &values[grad_start..grad_end],
            _ => candle_core::bail!("seg-prod backward expects an f64 gradient"),
        };

        let rows: usize = arg_dims[..arg_dims.len() - 1].iter().product();
        let mut out = vec![0f64; rows * inner];
        let mut zero_count = vec![0usize; self.num_segments];
        let mut nonzero_product = vec![1f64; self.num_segments];
        for row in 0..rows {
            zero_count.fill(0);
            nonzero_product.fill(1.0);
            let args = &args[row * inner..(row + 1) * inner];
            let grads = &grads[row * self.num_segments..(row + 1) * self.num_segments];
            let out = &mut out[row * inner..(row + 1) * inner];

            for (index, &value) in args.iter().enumerate() {
                let segment = self.segment_of[index] as usize;
                if value == 0.0 {
                    zero_count[segment] += 1;
                } else {
                    nonzero_product[segment] *= value;
                }
            }
            for (index, (&value, derivative)) in args.iter().zip(out.iter_mut()).enumerate() {
                let segment = self.segment_of[index] as usize;
                let local = match zero_count[segment] {
                    0 => nonzero_product[segment] / value,
                    1 if value == 0.0 => nonzero_product[segment],
                    _ => 0.0,
                };
                *derivative = grads[segment] * local;
            }
        }

        Ok((CpuStorage::F64(out), Shape::from(arg_dims.to_vec())))
    }
}

/// Index tensors and constants derived once from a [`Transcription`].
#[derive(Debug)]
pub struct TensorPlan {
    pub horizon: usize,
    pub particles: usize,
    pub num_actions: usize,
    pub num_facts: usize,
    pub num_variables: usize,
    pub num_preconditions: usize,
    pub num_groups: usize,
    pub num_effects: usize,
    pub num_goals: usize,

    device: Device,

    /// Transcription variable of each fact, as `u32` indices.
    var_of_fact: Tensor,
    /// Action and fact of each precondition incidence.
    pre_action: Tensor,
    pre_fact: Tensor,
    /// Per effect, the action of the group it belongs to.
    effect_action: Tensor,
    /// Canonical `(action, fact)` producer incidences used by optimistic
    /// verifier-triggered insertion discovery.
    producer_action: Tensor,
    producer_fact: Tensor,
    /// Producer-pair index for every precondition incidence of that pair's
    /// action. Together with `seg_producer_precondition_to_fact`, this is the
    /// sparse producer-to-prerequisite relation used by backward causal flow.
    producer_precondition_pair: Tensor,
    /// Optimistic structural add incidence, shape `[1, A, F]`.
    producer_matrix: Tensor,
    /// Canonical action-precondition incidence, shape `[1, A, F]`.
    precondition_matrix: Tensor,
    /// Per group, the action whose conditional writes the group contains.
    group_action: Tensor,
    /// Condition segmentation: which effect each condition fact belongs to.
    cond_fact: Tensor,
    cond_effect: Vec<u32>,
    /// Group segmentation of effects, used for "some effect fires".
    effect_group: Vec<u32>,
    /// For each `(earlier effect, later effect)` pair in one write group, the
    /// later effect and the earlier effect whose suffix product it contributes
    /// to. This implements last-write-wins without assuming one write per group.
    suffix_later_effect: Tensor,
    suffix_owner_effect: Vec<u32>,
    /// Goal facts.
    goal_fact: Tensor,
    precondition_count: Tensor,

    /// Segment maps for the custom op.
    seg_precondition_to_action: Vec<u32>,
    seg_precondition_to_fact: Vec<u32>,
    seg_fact_to_var: Vec<u32>,
    seg_effect_to_fact: Vec<u32>,
    seg_producer_to_fact: Vec<u32>,
    seg_producer_precondition_to_fact: Vec<u32>,
    seg_group_to_var: Vec<u32>,

    /// The fixed initial state row, shape `[1, 1, F]`.
    initial_row: Tensor,
    /// Constant terminal fact demand, shape `[1, 1, F]`.
    goal_demand_row: Tensor,
    /// Triangular causal-link mask, shape `[1, H + 1, 1, H + 1]`.
    causal_link_mask: Tensor,
}

/// Why a transcription cannot be handed to the tensor layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorPlanError {
    /// The horizon must leave at least one action slot.
    EmptyHorizon,
    /// Optimizing zero particles is not a thing.
    NoParticles,
}

impl std::fmt::Display for TensorPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyHorizon => write!(f, "horizon must be at least 1"),
            Self::NoParticles => write!(f, "particles must be at least 1"),
        }
    }
}

impl std::error::Error for TensorPlanError {}

/// Residuals of one forward pass. Shapes are `[M, H, ...]`, except the goal
/// family which lives on the terminal row only.
#[derive(Debug, Clone)]
pub struct Forward {
    /// Action distributions, `[M, H, A]`.
    pub action: Tensor,
    /// Stable logarithm of `action`, `[M, H, A]`.
    pub action_log_probability: Tensor,
    /// State distributions, `[M, H + 1, F]`.
    pub state: Tensor,
    /// Actual last-write addition mass, `[M, H, F]`.
    pub add: Tensor,
    /// Actual last-write deletion mass, `[M, H, F]`.
    pub delete: Tensor,
    /// Precondition residuals, `[M, H, Npre]`.
    pub precondition: Tensor,
    /// The four transition residuals, each `[M, H, F]`.
    pub transition: [Tensor; 4],
    /// Goal residuals, `[M, 1, Ngoal]`.
    pub goal: Tensor,
}

/// One factorized distribution over optional action slots.
///
/// The first `A - 1` optimizer coordinates parameterize a conditional
/// distribution over real operators. The final coordinate parameterizes slot
/// occupancy. The assembled `action` tensor retains the transcription's
/// ordinary `[real actions..., no-op]` layout, so every downstream STRIPS
/// equation remains unchanged.
#[derive(Debug, Clone)]
pub struct SlotDistribution {
    /// Full categorical distribution, `[M, H, A]`.
    pub action: Tensor,
    /// Conditional distribution over real operators, `[M, H, A - 1]`.
    pub real_action: Tensor,
    /// Probability that the slot contains a real operator, `[M, H, 1]`.
    pub occupancy: Tensor,
    /// Stable logarithm of `action`, `[M, H, A]`.
    pub log_action: Tensor,
}

/// Action identities carried by latent tokens and assigned to temporal rows.
///
/// `assignment[m,k,t]` is the mass assigning token `k` to execution row `t`.
/// Sinkhorn normalization makes the matrix doubly stochastic, so tokens can
/// move continuously without duplicating a role or exceeding one token per
/// row. At a permutation matrix and one-hot token actions this is exactly the
/// ordinary bounded-plan representation.
#[derive(Debug, Clone)]
pub struct TemporalTokenDistribution {
    pub action: Tensor,
    pub log_action: Tensor,
    pub token_action: Tensor,
    /// Exact/hard execution assignment with the chosen straight-through
    /// derivative.
    pub assignment: Tensor,
    /// Globally normalized continuous assignment used by conjunctive temporal
    /// losses. Keeping this separate avoids products of different hard one-hot
    /// rows becoming identically zero in both value and gradient.
    pub soft_assignment: Tensor,
}

/// The two losses used by the action-only planner.
///
/// `relaxed_goal` rolls the action distribution through a delete relaxation:
/// facts, once reached, remain reached. `failed_precondition` rolls the same
/// distribution through the real delete-aware transition and charges action
/// mass for every precondition that is false there. Both are kept per particle
/// so their weights can follow different points of the alternating schedule.
#[derive(Debug, Clone)]
pub struct TwoLossForward {
    pub action: Tensor,
    /// Delete-aware probabilistic states before each action, `[M, H, F]`.
    pub exact_state_by_step: Tensor,
    /// Shape `[M, 1]`.
    pub relaxed_goal: Tensor,
    /// Terminal plus peak-producer loss for each goal, `[M, G]`.
    pub relaxed_goal_by_goal: Tensor,
    /// Delete-aware terminal loss for each goal, `[M, G]`.
    ///
    /// This is computed from the applicability-supported recurrent state, not
    /// the monotone producer relaxation. It therefore remains positive when a
    /// later action clobbers an earlier achieved goal.
    pub terminal_goal_by_goal: Tensor,
    /// Analytic delete-aware survival loss for each goal, `[M, G]`.
    ///
    /// Every initial or added goal fact is a candidate source. A source counts
    /// only if no later applicability-supported delete reaches that fact; a
    /// noisy-or combines the surviving candidates. At an applicable one-hot
    /// trajectory this equals [`Self::terminal_goal_by_goal`], while its
    /// source-to-threat paths are much shorter than the full recurrence.
    pub surviving_goal_by_goal: Tensor,
    /// Probability that the initial state or some action supplies each goal,
    /// `[M, G]`.
    pub some_goal_producer_probability: Tensor,
    /// Negative log of [`Self::some_goal_producer_probability`], `[M, G]`.
    pub some_goal_producer_loss: Tensor,
    /// Goal-directed no-op pressure, `[M, 1]`.
    pub premature_noop: Tensor,
    /// Shape `[M, 1]`.
    pub failed_precondition: Tensor,
    /// Shape `[M, H]`, retained so exact first-failure feedback can emphasize
    /// one row without selecting a replacement action.
    pub failed_precondition_by_step: Tensor,
    /// Same forward residual as [`Self::failed_precondition_by_step`], but the
    /// selected consumer action is detached. During goal-chain repair this
    /// routes applicability pressure into supporting earlier actions instead
    /// of deleting the newly introduced achiever.
    pub support_only_failed_precondition_by_step: Tensor,
    /// Shape `[M, 1]`, used only to make the decoded rows decisive.
    pub action_integrality: Tensor,
    /// Applicability-supported causal evidence from the same delete-aware
    /// recurrent rollout. This is the coherent input for causal links attached
    /// to an action-only proof copy: no independent state tensor can fabricate
    /// an addition, deletion, or consumer demand.
    pub causal: CausalLinkInput,
}

/// Causal-link evidence produced by one coherent rollout.
///
/// All three tensors are applicability-supported. In particular, an action
/// with a false precondition contributes neither a consumer obligation nor an
/// effect source. Shapes are `[M, H, F]`.
#[derive(Debug, Clone)]
pub struct CausalLinkInput {
    pub action_demand: Tensor,
    pub add: Tensor,
    pub delete: Tensor,
}

/// Sparse verifier-demanded producer evidence at explicit deadlines.
#[derive(Debug, Clone)]
pub struct DeadlineSupportForward {
    /// Optimistic structural-producer loss, `[M, 1]`.
    pub raw_loss: Tensor,
    /// Applicability-supported, delete-aware survival loss, `[M, 1]`.
    pub supported_loss: Tensor,
    /// Demand-weighted mean raw evidence, `[M, 1]`.
    pub raw_evidence: Tensor,
    /// Demand-weighted mean supported evidence, `[M, 1]`.
    pub supported_evidence: Tensor,
}

/// Continuous backward chaining from verifier-missing goals.
#[derive(Debug, Clone)]
pub struct BackwardBridgeForward {
    /// Boundary support plus goal-relevant local applicability, `[M, 1]`.
    pub loss: Tensor,
    /// Requirements left unsupported at the exact repair boundary, `[M, 1]`.
    pub boundary_loss: Tensor,
    /// False preconditions of actions relevant to the backward chain, `[M, 1]`.
    pub relevant_precondition_loss: Tensor,
}

/// Fact-grouped latent causal-link witnesses.
///
/// Consumer rows `0..H` contain the precondition demand of the action in that
/// row; row `H` contains constant terminal-goal demand. Source row zero is the
/// initial state and source row `s + 1` is the actual last-write addition mass
/// of action row `s`. The triangular mask therefore permits exactly the
/// strictly earlier action sources for every action consumer.
#[derive(Debug, Clone)]
pub struct CausalLinkForward {
    /// Fact demand per consumer, `[M, H + 1, F]`.
    pub demand: Tensor,
    /// Masked source distribution, `[M, H + 1, F, H + 1]`.
    pub link: Tensor,
    /// Demand-weighted unsupported-source hinge per particle, `[M, 1]`.
    pub source_loss: Tensor,
    /// Demand-weighted intervening-delete loss per particle, `[M, 1]`.
    pub threat_loss: Tensor,
    /// Demand-weighted link integrality per particle, `[M, 1]`.
    pub link_integrality: Tensor,
    /// Largest demand-weighted unsupported-source violation before demand
    /// normalization, `[M, 1]`.
    pub max_source_violation: Tensor,
    /// Largest demand-weighted intervening-delete violation before demand
    /// normalization, `[M, 1]`.
    pub max_threat_violation: Tensor,
    /// Total active consumer mass per particle, `[M, 1]` and detached.
    pub active_consumer_mass: Tensor,
}

/// Smooth bottleneck norm over every non-particle coordinate.
///
/// Returns `(mean_i |x_i|^p)^(1/p)` with shape `[M, 1]`. Unlike a detached
/// top-k selection, every residual retains a gradient, while larger `p`
/// increasingly concentrates pressure on the worst violation.
pub fn bottleneck_norm_per_particle(residual: &Tensor, p: f64) -> CandleResult<Tensor> {
    if residual.rank() < 2 {
        candle_core::bail!(
            "bottleneck norm expects a particle axis and at least one residual axis, got shape {:?}",
            residual.dims()
        );
    }
    if !p.is_finite() || !(1.0..=64.0).contains(&p) {
        candle_core::bail!("bottleneck exponent must be finite and in [1, 64], got {p}");
    }
    let particles = residual.dim(0)?;
    let powered = residual.abs()?.powf(p)?.flatten_from(1)?;
    if powered.dim(1)? == 0 {
        return Tensor::zeros((particles, 1), residual.dtype(), residual.device());
    }
    // The derivative of x^(1/p) is infinite at zero for p > 1. Candle's
    // generic pow backward would consequently multiply `inf * 0` through the
    // preceding x^p and produce NaNs for a perfectly satisfied family. This
    // machine-small translation preserves the norm to floating-point accuracy,
    // keeps exact zero at zero, and makes the whole backward finite.
    let epsilon = f64::MIN_POSITIVE;
    ((powered.mean(1)? + epsilon)?.powf(1.0 / p)? - epsilon.powf(1.0 / p))?.reshape((particles, 1))
}

impl TensorPlan {
    pub fn new(
        transcription: &Transcription,
        horizon: usize,
        particles: usize,
        device: Device,
    ) -> Result<Self, TensorPlanError> {
        if horizon == 0 {
            return Err(TensorPlanError::EmptyHorizon);
        }
        if particles == 0 {
            return Err(TensorPlanError::NoParticles);
        }

        let num_facts = transcription.num_facts();
        let num_effects = transcription.num_effects();
        let num_groups = transcription.num_groups();

        // Per effect, the action of its group.
        let effect_action: Vec<u32> = (0..num_effects)
            .map(|effect| {
                let group = transcription.effect_group()[effect] as usize;
                transcription.group_action()[group]
            })
            .collect();

        // For effect i, the last-write mass contains Π_{j>i}(1-φ_j) over
        // later effects in the same `(action, variable)` group.
        let mut suffix_later_effect = Vec::new();
        let mut suffix_owner_effect = Vec::new();
        for group in 0..num_groups {
            let effects = transcription.group_effects(group);
            for (position, &owner) in effects.iter().enumerate() {
                for &later in &effects[position + 1..] {
                    suffix_later_effect.push(later);
                    suffix_owner_effect.push(owner);
                }
            }
        }

        let u32_tensor = |values: &[u32]| -> Result<Tensor, TensorPlanError> {
            Tensor::from_slice(values, values.len(), &device).map_err(|error| {
                panic!("building an index tensor cannot fail: {error}");
            })
        };
        let mut precondition_count = vec![0f64; transcription.num_actions()];
        for &action in transcription.pre_action() {
            precondition_count[action as usize] += 1.0;
        }
        for count in &mut precondition_count {
            if *count == 0.0 {
                *count = 1.0;
            }
        }

        let producer_pairs = effect_action
            .iter()
            .copied()
            .zip(transcription.effect_fact().iter().copied())
            .collect::<BTreeSet<_>>();
        let producer_action = producer_pairs
            .iter()
            .map(|&(action, _)| action)
            .collect::<Vec<_>>();
        let producer_fact = producer_pairs
            .iter()
            .map(|&(_, fact)| fact)
            .collect::<Vec<_>>();
        let mut producer_precondition_pair = Vec::new();
        let mut producer_precondition_fact = Vec::new();
        for (pair, &(action, _)) in producer_pairs.iter().enumerate() {
            for (&precondition_action, &fact) in transcription
                .pre_action()
                .iter()
                .zip(transcription.pre_fact())
            {
                if precondition_action == action {
                    producer_precondition_pair.push(pair as u32);
                    producer_precondition_fact.push(fact);
                }
            }
        }
        let mut producer_matrix = vec![0f64; transcription.num_actions() * num_facts];
        for &(action, fact) in &producer_pairs {
            producer_matrix[action as usize * num_facts + fact as usize] = 1.0;
        }
        let mut precondition_matrix = vec![0f64; transcription.num_actions() * num_facts];
        for (&action, &fact) in transcription
            .pre_action()
            .iter()
            .zip(transcription.pre_fact())
        {
            precondition_matrix[action as usize * num_facts + fact as usize] = 1.0;
        }
        let seg_producer_to_fact = producer_pairs
            .iter()
            .map(|&(_, fact)| fact)
            .collect::<Vec<_>>();

        let mut initial = vec![0f64; num_facts];
        for &fact in transcription.initial_fact() {
            initial[fact as usize] = 1.0;
        }
        let initial_row = Tensor::from_vec(initial, (1, 1, num_facts), &device)
            .expect("building the initial row cannot fail");

        let mut goal_demand = vec![0f64; num_facts];
        for &fact in transcription.goal_facts() {
            goal_demand[fact as usize] = 1.0;
        }
        let goal_demand_row = Tensor::from_vec(goal_demand, (1, 1, num_facts), &device)
            .expect("building terminal goal demand cannot fail");

        // Source zero is the initial state; source `s + 1` is action row `s`.
        // Thus source <= consumer permits the initial state and only strictly
        // earlier actions. Every consumer has at least source zero, so the
        // masked softmax below always has a nonempty support.
        let link_rows = horizon + 1;
        let causal_link_mask: Vec<f64> = (0..link_rows)
            .flat_map(|consumer| (0..link_rows).map(move |source| f64::from(source <= consumer)))
            .collect();
        let causal_link_mask =
            Tensor::from_vec(causal_link_mask, (1, link_rows, 1, link_rows), &device)
                .expect("building the causal-link mask cannot fail");

        Ok(Self {
            horizon,
            particles,
            num_actions: transcription.num_actions(),
            num_facts,
            num_variables: transcription.num_variables(),
            num_preconditions: transcription.pre_action().len(),
            num_groups,
            num_effects,
            num_goals: transcription.goal_facts().len(),
            var_of_fact: u32_tensor(transcription.var_of_fact())?,
            pre_action: u32_tensor(transcription.pre_action())?,
            pre_fact: u32_tensor(transcription.pre_fact())?,
            effect_action: u32_tensor(&effect_action)?,
            producer_action: u32_tensor(&producer_action)?,
            producer_fact: u32_tensor(&producer_fact)?,
            producer_precondition_pair: u32_tensor(&producer_precondition_pair)?,
            producer_matrix: Tensor::from_vec(
                producer_matrix,
                (1, transcription.num_actions(), num_facts),
                &device,
            )
            .expect("building producer incidence cannot fail"),
            precondition_matrix: Tensor::from_vec(
                precondition_matrix,
                (1, transcription.num_actions(), num_facts),
                &device,
            )
            .expect("building precondition incidence cannot fail"),
            group_action: u32_tensor(transcription.group_action())?,
            cond_fact: u32_tensor(transcription.cond_fact())?,
            cond_effect: transcription.cond_effect().to_vec(),
            effect_group: transcription.effect_group().to_vec(),
            suffix_later_effect: u32_tensor(&suffix_later_effect)?,
            suffix_owner_effect,
            goal_fact: u32_tensor(transcription.goal_facts())?,
            precondition_count: Tensor::from_vec(
                precondition_count,
                (1, 1, transcription.num_actions()),
                &device,
            )
            .expect("building precondition counts cannot fail"),
            seg_precondition_to_action: transcription.pre_action().to_vec(),
            seg_precondition_to_fact: transcription.pre_fact().to_vec(),
            seg_fact_to_var: transcription.var_of_fact().to_vec(),
            seg_effect_to_fact: transcription.effect_fact().to_vec(),
            seg_producer_to_fact,
            seg_producer_precondition_to_fact: producer_precondition_fact,
            seg_group_to_var: transcription.group_var().to_vec(),
            device,
            initial_row,
            goal_demand_row,
            causal_link_mask,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Expected dense causal-link-logit shape `[M, H + 1, F, H + 1]`.
    pub fn causal_link_shape(&self) -> [usize; 4] {
        [
            self.particles,
            self.horizon + 1,
            self.num_facts,
            self.horizon + 1,
        ]
    }

    /// Action distributions from logits at a per-particle temperature.
    ///
    /// The temperature is a `[M, 1, 1]` tensor rather than a scalar so each
    /// particle can sit at a different point of the annealing cycle. That is what
    /// lets some particles melt while others crystallize, which gives the
    /// population genuine diversity without any ranking or selection between
    /// particles.
    pub fn action_distribution(
        &self,
        logits: &Tensor,
        temperature: &Tensor,
    ) -> CandleResult<Tensor> {
        candle_nn::ops::softmax(&logits.broadcast_div(temperature)?, 2)
    }

    /// Separate action identity (tokens) from temporal position (assignment).
    ///
    /// Both logit tensors use the horizon as their token dimension. Repeated
    /// row/column normalization is differentiable and stays strictly positive
    /// for finite logits; the final discrete limit is a permutation of the
    /// token actions, not an alias-expanded action set.
    pub fn temporal_token_distribution(
        &self,
        token_action_logits: &Tensor,
        schedule_logits: &Tensor,
        action_temperature: &Tensor,
        schedule_temperature: &Tensor,
        schedule_gate: &Tensor,
        sinkhorn_iterations: usize,
    ) -> CandleResult<TemporalTokenDistribution> {
        assert_eq!(
            token_action_logits.dims(),
            &[self.particles, self.horizon, self.num_actions]
        );
        assert_eq!(
            schedule_logits.dims(),
            &[self.particles, self.horizon, self.horizon]
        );
        assert_eq!(
            action_temperature.dims(),
            &[self.particles, self.horizon, 1]
        );
        assert_eq!(schedule_temperature.dims(), &[self.particles, 1, 1]);
        assert_eq!(schedule_gate.dims(), &[self.particles, 1, 1]);
        if sinkhorn_iterations == 0 {
            candle_core::bail!("temporal token Sinkhorn iterations must be at least one");
        }
        if schedule_temperature.min_all()?.to_scalar::<f64>()? <= 0.0 {
            candle_core::bail!("temporal token schedule temperature must be positive");
        }

        let token_action = self.action_distribution(token_action_logits, action_temperature)?;
        let scaled_schedule = schedule_logits.broadcast_div(schedule_temperature)?;
        let mut assignment = candle_nn::ops::softmax(&scaled_schedule, 2)?;
        for _ in 0..sinkhorn_iterations {
            assignment =
                assignment.broadcast_div(&assignment.sum_keepdim(2)?.clamp(1e-300, f64::MAX)?)?;
            // Finish on execution columns: every scheduled action row must be
            // a convex combination of token distributions even before
            // Sinkhorn has converged to a doubly stochastic fixed point.
            assignment =
                assignment.broadcast_div(&assignment.sum_keepdim(1)?.clamp(1e-300, f64::MAX)?)?;
        }
        let identity = Tensor::eye(self.horizon, DTYPE, &self.device)?.unsqueeze(0)?;
        let locked = (schedule_gate.ones_like()? - schedule_gate)?;
        assignment =
            ((assignment.broadcast_mul(schedule_gate)?) + identity.broadcast_mul(&locked)?)?;
        let action = assignment.transpose(1, 2)?.matmul(&token_action)?;
        let log_action = action.clamp(1e-300, 1.0)?.log()?;
        Ok(TemporalTokenDistribution {
            action,
            log_action,
            token_action,
            soft_assignment: assignment.clone(),
            assignment,
        })
    }

    pub fn temporal_assignment_integrality(&self, assignment: &Tensor) -> CandleResult<Tensor> {
        assert_eq!(
            assignment.dims(),
            &[self.particles, self.horizon, self.horizon]
        );
        // Execution columns are normalized exactly by the final Sinkhorn
        // step, so their categorical impurity is non-negative even while the
        // token-row marginal is still converging.
        let sum_squares = assignment.sqr()?.sum_keepdim(1)?.transpose(1, 2)?;
        sum_squares.ones_like()? - sum_squares
    }

    /// Factorized optional-slot distribution.
    ///
    /// This is only a reparameterization of the interior of the full action
    /// simplex: `occupancy` is the total real-action mass and `real_action` is
    /// that mass normalized over real operators. Stable log probabilities are
    /// built before exponentiation so a saturated occupancy never turns a
    /// finite ranking or KL objective into `log(0)`.
    pub fn factorized_action_distribution(
        &self,
        logits: &Tensor,
        temperature: &Tensor,
    ) -> CandleResult<SlotDistribution> {
        assert_eq!(
            logits.dims(),
            &[self.particles, self.horizon, self.num_actions],
            "factorized action logits have shape [M, H, A]"
        );
        let scaled = logits.broadcast_div(temperature)?;
        if self.num_actions == 1 {
            let real_action =
                Tensor::zeros((self.particles, self.horizon, 0), DTYPE, &self.device)?;
            let occupancy = Tensor::zeros((self.particles, self.horizon, 1), DTYPE, &self.device)?;
            let action = Tensor::ones((self.particles, self.horizon, 1), DTYPE, &self.device)?;
            let log_action = Tensor::zeros((self.particles, self.horizon, 1), DTYPE, &self.device)?;
            return Ok(SlotDistribution {
                action,
                real_action,
                occupancy,
                log_action,
            });
        }

        let real_logits = scaled.narrow(2, 0, self.num_actions - 1)?;
        // Zero occupancy coordinate must reproduce the legacy neutral
        // categorical prior: every real action and the no-op have equal mass.
        // Since `q` is conditional over `A - 1` real actions, its total mass
        // needs log-odds `ln(A - 1)` against the single no-op. Keeping this
        // offset outside temperature scaling preserves that invariant for
        // every positive temperature.
        let occupancy_logit =
            (scaled.narrow(2, self.num_actions - 1, 1)? + ((self.num_actions - 1) as f64).ln())?;
        let log_real_action = candle_nn::ops::log_softmax(&real_logits, 2)?;

        // log(sigmoid(x)) = min(x, 0) - log(1 + exp(-abs(x))).  Expressing
        // min(x, 0) as x - relu(x) keeps the whole operation differentiable
        // in Candle and bounds the exponential argument by zero.
        let log_sigmoid = |value: &Tensor| -> CandleResult<Tensor> {
            let minimum = (value - value.relu()?)?;
            let correction = ((value.abs()?.neg()?.exp()? + 1.0)?).log()?;
            minimum - correction
        };
        let log_occupancy = log_sigmoid(&occupancy_logit)?;
        let log_noop = log_sigmoid(&occupancy_logit.neg()?)?;
        let log_real_mass = log_real_action.broadcast_add(&log_occupancy)?;
        let log_action = Tensor::cat(&[&log_real_mass, &log_noop], 2)?;
        let real_action = log_real_action.exp()?;
        let occupancy = log_occupancy.exp()?;
        let action = log_action.exp()?;
        Ok(SlotDistribution {
            action,
            real_action,
            occupancy,
            log_action,
        })
    }

    /// Hybrid distribution: ordinary categorical anchor rows with periodic
    /// factorized insertion rows.
    ///
    /// Making every row optional gives precondition loss a global escape route
    /// through occupancy. Restricting that coordinate system to explicit slack
    /// rows preserves the proven categorical geometry elsewhere while still
    /// making an insertion an `O(1)` change in a nearby row.
    pub fn hybrid_action_distribution(
        &self,
        logits: &Tensor,
        temperature: &Tensor,
        slack_window: usize,
    ) -> CandleResult<SlotDistribution> {
        if slack_window == 1 {
            candle_core::bail!("slot slack window must be zero or at least 2");
        }
        let scaled = logits.broadcast_div(temperature)?;
        let categorical_log = candle_nn::ops::log_softmax(&scaled, 2)?;
        let categorical_action = categorical_log.exp()?;
        let categorical_occupancy = if self.num_actions == 1 {
            Tensor::zeros((self.particles, self.horizon, 1), DTYPE, &self.device)?
        } else {
            categorical_action
                .narrow(2, 0, self.num_actions - 1)?
                .sum_keepdim(2)?
        };
        let categorical_real = if self.num_actions == 1 {
            Tensor::zeros((self.particles, self.horizon, 0), DTYPE, &self.device)?
        } else {
            categorical_action
                .narrow(2, 0, self.num_actions - 1)?
                .broadcast_div(&categorical_occupancy)?
        };
        if slack_window == 0 || self.horizon < slack_window {
            return Ok(SlotDistribution {
                action: categorical_action,
                real_action: categorical_real,
                occupancy: categorical_occupancy,
                log_action: categorical_log,
            });
        }

        let factorized = self.factorized_action_distribution(logits, temperature)?;
        let slack = Tensor::from_vec(
            (0..self.horizon)
                .map(|row| f64::from((row + 1) % slack_window == 0))
                .collect::<Vec<_>>(),
            (1, self.horizon, 1),
            &self.device,
        )?
        .broadcast_as((self.particles, self.horizon, 1))?;
        let anchors = (slack.ones_like()? - &slack)?;
        let full_slack = slack.broadcast_as((self.particles, self.horizon, self.num_actions))?;
        let full_anchors =
            anchors.broadcast_as((self.particles, self.horizon, self.num_actions))?;
        let real_slack = slack.broadcast_as((
            self.particles,
            self.horizon,
            self.num_actions.saturating_sub(1),
        ))?;
        let real_anchors = anchors.broadcast_as((
            self.particles,
            self.horizon,
            self.num_actions.saturating_sub(1),
        ))?;
        Ok(SlotDistribution {
            action: ((categorical_action * &full_anchors)? + (factorized.action * &full_slack)?)?,
            real_action: ((categorical_real * &real_anchors)?
                + (factorized.real_action * &real_slack)?)?,
            occupancy: ((categorical_occupancy * &anchors)? + (factorized.occupancy * &slack)?)?,
            log_action: ((categorical_log * &full_anchors)?
                + (factorized.log_action * &full_slack)?)?,
        })
    }

    /// State distributions from logits at temperature `temperature`, with the
    /// fixed initial row prepended.
    ///
    /// This is a *segmented* softmax: each finite-domain variable gets its own
    /// normalization, so "one value per variable" is structural.
    pub fn state_distribution(
        &self,
        logits: &Tensor,
        temperature: &Tensor,
    ) -> CandleResult<Tensor> {
        let scaled = logits.broadcast_div(temperature)?;
        let unnormalized = scaled.exp()?;
        let per_variable = unnormalized.contiguous()?.apply_op1(SegSum::new(
            self.num_variables,
            self.seg_fact_to_var.clone(),
        ))?;
        let denominator = per_variable.index_select(&self.var_of_fact, 2)?;
        let rows = (unnormalized / denominator)?;
        let initial = self
            .initial_row
            .broadcast_as((self.particles, 1, self.num_facts))?
            .contiguous()?;
        Tensor::cat(&[&initial, &rows], 1)
    }

    /// Per-effect condition products `φ`, shape `[M, H, Neff]`.
    fn condition_products(&self, current: &Tensor) -> CandleResult<Tensor> {
        let steps = current.dim(1)?;
        if self.cond_effect.is_empty() {
            // No conditional effects: every φ is 1. Skipping the log/exp round
            // trip here is not just an optimization, it avoids clamping noise.
            return Tensor::ones(
                (self.particles, steps, self.num_effects),
                DTYPE,
                &self.device,
            );
        }
        let gathered = current.index_select(&self.cond_fact, 2)?;
        // An effect with no conditions has an empty segment, whose product is
        // one. Unlike a clamped log-product this preserves exact zeros.
        gathered
            .contiguous()?
            .apply_op1(SegProd::new(self.num_effects, self.cond_effect.clone()))
    }

    /// Smooth conjunction of operator preconditions, one value per action.
    ///
    /// This is propagation semantics, not a shaping surrogate: an action can
    /// fire only on the joint event that every precondition holds. An action
    /// without preconditions has support exactly one. Gradient starvation from
    /// multiple false literals is handled separately by
    /// [`Self::literal_precondition_loss`], never by leaking effects.
    fn action_support(&self, state: &Tensor) -> CandleResult<Tensor> {
        if self.num_preconditions == 0 {
            return Tensor::ones(
                (self.particles, state.dim(1)?, self.num_actions),
                DTYPE,
                &self.device,
            );
        }
        let selected = state.contiguous()?.index_select(&self.pre_fact, 2)?;
        selected.contiguous()?.apply_op1(SegProd::new(
            self.num_actions,
            self.seg_precondition_to_action.clone(),
        ))
    }

    /// Expected fraction of false preconditions of the sampled action.
    ///
    /// Each missing literal contributes independently:
    ///
    /// `Σ_a P[a] / |pre(a)| · Σ_{f∈pre(a)} (1 - S[f])`.
    ///
    /// The per-action normalization keeps the value in `[0,1]` and avoids
    /// penalizing high-arity actions merely for having more incidences. This
    /// additive loss has a gradient for every false literal even when the exact
    /// product used to propagate effects is zero.
    fn literal_precondition_loss(&self, state: &Tensor, action: &Tensor) -> CandleResult<Tensor> {
        if self.num_preconditions == 0 {
            return Tensor::zeros((self.particles, state.dim(1)?), DTYPE, &self.device);
        }
        let selected_fact = state.contiguous()?.index_select(&self.pre_fact, 2)?;
        let selected_action = action.contiguous()?.index_select(&self.pre_action, 2)?;
        let action_arity = self.precondition_count.index_select(&self.pre_action, 2)?;
        ((selected_action * (selected_fact.ones_like()? - selected_fact)?)?
            .broadcast_div(&action_arity))?
        .sum(2)
    }

    /// Fact demand induced by action mass, with duplicate `(action, fact)`
    /// preconditions already canonicalized by the transcription.
    fn action_fact_demand(&self, action: &Tensor) -> CandleResult<Tensor> {
        if self.num_preconditions == 0 {
            return Tensor::zeros(
                (self.particles, action.dim(1)?, self.num_facts),
                DTYPE,
                &self.device,
            );
        }
        action
            .contiguous()?
            .index_select(&self.pre_action, 2)?
            .contiguous()?
            .apply_op1(SegSum::new(
                self.num_facts,
                self.seg_precondition_to_fact.clone(),
            ))
    }

    /// Add and change masses for one or more rows.
    fn effect_masses(&self, state: &Tensor, action: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let phi = self.condition_products(&state.contiguous()?)?;

        // An effect contributes only where it fires and no later effect in its
        // `(action, variable)` group fires. This is the same last-write-wins
        // suffix product as the scalar reference.
        let suffix_no_write = if self.suffix_owner_effect.is_empty() {
            Tensor::ones(
                (self.particles, state.dim(1)?, self.num_effects),
                DTYPE,
                &self.device,
            )?
        } else {
            (phi.ones_like()? - &phi)?
                .contiguous()?
                .index_select(&self.suffix_later_effect, 2)?
                .contiguous()?
                .apply_op1(SegProd::new(
                    self.num_effects,
                    self.suffix_owner_effect.clone(),
                ))?
        };
        let last_write = (&phi * suffix_no_write)?;
        let effect_mass =
            (action.contiguous()?.index_select(&self.effect_action, 2)? * last_write)?;
        let add = effect_mass
            .contiguous()?
            .apply_op1(SegSum::new(self.num_facts, self.seg_effect_to_fact.clone()))?;

        // Change mass is the probability that at least one effect in a group
        // fires, multiplied once by the joint action event.
        let no_write = (phi.ones_like()? - phi)?
            .contiguous()?
            .apply_op1(SegProd::new(self.num_groups, self.effect_group.clone()))?;
        let group_fires = (no_write.ones_like()? - no_write)?;
        let group_mass = (action.contiguous()?.index_select(&self.group_action, 2)? * group_fires)?;
        let change_per_variable = group_mass.contiguous()?.apply_op1(SegSum::new(
            self.num_variables,
            self.seg_group_to_var.clone(),
        ))?;
        let change = change_per_variable.index_select(&self.var_of_fact, 2)?;
        Ok((add, change))
    }

    /// Producer evidence for sparse fact demands at explicit consumer rows.
    ///
    /// `demand[m,d,f]` requests fact `f` before action row `d`; row `H` is the
    /// terminal goal deadline. `source_mask[m,d,s]` decides which action rows
    /// may serve that deadline. The engine constructs this mask from a
    /// continuous repair window, never from a selected replacement operator.
    pub fn deadline_support_forward(
        &self,
        action: &Tensor,
        causal: &CausalLinkInput,
        demand: &Tensor,
        source_mask: &Tensor,
        active_deadlines: &[usize],
    ) -> CandleResult<DeadlineSupportForward> {
        let initial = self
            .initial_row
            .broadcast_as((self.particles, 1, self.num_facts))?;
        self.deadline_support_forward_from_boundary(
            action,
            causal,
            &initial,
            demand,
            source_mask,
            active_deadlines,
        )
    }

    /// Deadline support relative to an exact verifier-proven repair boundary.
    /// Facts deleted before this boundary are not fabricated from the task's
    /// initial state.
    pub fn deadline_support_forward_from_boundary(
        &self,
        action: &Tensor,
        causal: &CausalLinkInput,
        boundary_state: &Tensor,
        demand: &Tensor,
        source_mask: &Tensor,
        active_deadlines: &[usize],
    ) -> CandleResult<DeadlineSupportForward> {
        assert_eq!(
            action.dims(),
            &[self.particles, self.horizon, self.num_actions],
            "deadline support receives a full action distribution"
        );
        if active_deadlines.windows(2).any(|pair| pair[0] >= pair[1])
            || active_deadlines
                .iter()
                .any(|&deadline| deadline > self.horizon)
        {
            candle_core::bail!(
                "active deadlines must be strictly increasing and in 0..={}, got {:?}",
                self.horizon,
                active_deadlines
            );
        }
        if active_deadlines.is_empty() {
            let zero = Tensor::zeros((self.particles, 1), DTYPE, &self.device)?;
            return Ok(DeadlineSupportForward {
                raw_loss: zero.clone(),
                supported_loss: zero.clone(),
                raw_evidence: zero.clone(),
                supported_evidence: zero,
            });
        }
        assert_eq!(
            causal.add.dims(),
            &[self.particles, self.horizon, self.num_facts]
        );
        assert_eq!(causal.delete.dims(), causal.add.dims());
        assert_eq!(
            boundary_state.dims(),
            &[self.particles, 1, self.num_facts],
            "deadline boundary has one exact state per particle"
        );
        assert_eq!(
            demand.dims(),
            &[self.particles, self.horizon + 1, self.num_facts],
            "deadline demand has one row for every action consumer and the terminal goals"
        );
        assert_eq!(
            source_mask.dims(),
            &[self.particles, self.horizon + 1, self.horizon],
            "deadline source mask has shape [M, H + 1, H]"
        );

        let raw_add = if self.seg_producer_to_fact.is_empty() {
            Tensor::zeros(
                (self.particles, self.horizon, self.num_facts),
                DTYPE,
                &self.device,
            )?
        } else {
            action
                .contiguous()?
                .index_select(&self.producer_action, 2)?
                .contiguous()?
                .apply_op1(SegSum::new(
                    self.num_facts,
                    self.seg_producer_to_fact.clone(),
                ))?
        };
        let initial = boundary_state.clone();
        let epsilon = 1e-12;
        let mut raw_rows = Vec::with_capacity(active_deadlines.len());
        let mut supported_rows = Vec::with_capacity(active_deadlines.len());
        let mut demand_rows = Vec::with_capacity(active_deadlines.len());

        for &deadline in active_deadlines {
            let mut no_raw_source =
                Tensor::ones((self.particles, 1, self.num_facts), DTYPE, &self.device)?;
            let mut no_supported_source = no_raw_source.clone();
            let mut survival = no_raw_source.clone();
            for source in (0..deadline).rev() {
                let mask = source_mask
                    .narrow(1, deadline, 1)?
                    .narrow(2, source, 1)?
                    .reshape((self.particles, 1, 1))?;
                let raw = raw_add.narrow(1, source, 1)?.broadcast_mul(&mask)?;
                no_raw_source = (&no_raw_source * (raw.ones_like()? - raw)?)?;

                let supported =
                    (causal.add.narrow(1, source, 1)? * &survival)?.broadcast_mul(&mask)?;
                no_supported_source =
                    (&no_supported_source * (supported.ones_like()? - supported)?)?;
                survival = (&survival
                    * (causal.delete.narrow(1, source, 1)?.ones_like()?
                        - causal.delete.narrow(1, source, 1)?)?)?;
            }

            let raw_initial_missing = (initial.ones_like()? - &initial)?;
            let raw_evidence =
                (no_raw_source.ones_like()? - (no_raw_source * raw_initial_missing)?)?;
            let surviving_initial = (&initial * &survival)?;
            let supported_evidence = (no_supported_source.ones_like()?
                - (no_supported_source * (surviving_initial.ones_like()? - surviving_initial)?)?)?;
            raw_rows.push(raw_evidence);
            supported_rows.push(supported_evidence);
            demand_rows.push(demand.narrow(1, deadline, 1)?);
        }

        let raw_evidence_by_deadline = Tensor::cat(&raw_rows, 1)?;
        let supported_evidence_by_deadline = Tensor::cat(&supported_rows, 1)?;
        let active_demand = Tensor::cat(&demand_rows, 1)?;
        let demand_mass = active_demand.sum(2)?.sum(1)?.reshape((self.particles, 1))?;
        let denominator = demand_mass.clamp(1.0, f64::MAX)?;
        let weighted_mean = |evidence: &Tensor| -> CandleResult<Tensor> {
            (evidence * &active_demand)?
                .sum(2)?
                .sum(1)?
                .reshape((self.particles, 1))?
                .broadcast_div(&denominator)
        };
        let weighted_loss = |evidence: &Tensor| -> CandleResult<Tensor> {
            let safe = ((evidence * (1.0 - epsilon))? + epsilon)?;
            (safe.log()?.neg()? * &active_demand)?
                .sum(2)?
                .sum(1)?
                .reshape((self.particles, 1))?
                .broadcast_div(&denominator)
        };
        Ok(DeadlineSupportForward {
            raw_loss: weighted_loss(&raw_evidence_by_deadline)?,
            supported_loss: weighted_loss(&supported_evidence_by_deadline)?,
            raw_evidence: weighted_mean(&raw_evidence_by_deadline)?,
            supported_evidence: weighted_mean(&supported_evidence_by_deadline)?,
        })
    }

    /// Propagate verifier-missing terminal facts backwards through every soft
    /// action, without selecting an operator or extracting a relaxed plan.
    ///
    /// `support_state_by_step` is supplied by the engine's lifted-to-recurrent
    /// continuation. `active_rows[m,t]` is a suffix in which repair may change actions;
    /// `boundary_rows[m,t]` names its unique first row.  A required fact is
    /// discharged by structural producer mass at row `t`; the preconditions of
    /// every proportionally relevant producer become requirements at `t`.
    /// Conditional effects are intentionally optimistic here, just as in the
    /// raw producer objective. Exact recurrent losses and replay remain the
    /// validity certificate.
    pub fn backward_goal_bridge(
        &self,
        action: &Tensor,
        support_state_by_step: &Tensor,
        terminal_demand: &Tensor,
        active_rows: &Tensor,
        boundary_rows: &Tensor,
    ) -> CandleResult<BackwardBridgeForward> {
        assert_eq!(
            terminal_demand.dims(),
            &[self.particles, self.num_facts],
            "backward bridge demand has shape [M, F]"
        );
        assert_eq!(
            active_rows.dims(),
            &[self.particles, self.horizon, 1],
            "backward bridge active mask has shape [M, H, 1]"
        );
        assert_eq!(
            boundary_rows.dims(),
            active_rows.dims(),
            "backward bridge boundary mask matches the active mask"
        );
        assert_eq!(
            action.dims(),
            &[self.particles, self.horizon, self.num_actions]
        );
        assert_eq!(
            support_state_by_step.dims(),
            &[self.particles, self.horizon, self.num_facts]
        );

        let tolerance = 1e-12;
        for (name, tensor) in [
            ("terminal demand", terminal_demand),
            ("active-row mask", active_rows),
            ("boundary-row mask", boundary_rows),
        ] {
            let minimum = tensor.min_all()?.to_scalar::<f64>()?;
            let maximum = tensor.max_all()?.to_scalar::<f64>()?;
            if minimum < -tolerance || maximum > 1.0 + tolerance {
                candle_core::bail!(
                    "backward bridge {name} must lie in [0, 1], got [{minimum}, {maximum}]"
                );
            }
        }

        let demand = terminal_demand.clamp(0.0, 1.0)?;
        let demand_mass = demand
            .sum(1)?
            .reshape((self.particles, 1))?
            .clamp(1.0, f64::MAX)?;
        let active_mass = active_rows
            .sum(1)?
            .reshape((self.particles, 1))?
            .clamp(1.0, f64::MAX)?;
        let mut requirement = demand.reshape((self.particles, 1, self.num_facts))?;
        let mut boundary_residual = Tensor::zeros((self.particles, 1), DTYPE, &self.device)?;
        let mut relevant_precondition = boundary_residual.clone();

        for timestep in (0..self.horizon).rev() {
            let active = active_rows.narrow(1, timestep, 1)?;
            let action = action.narrow(1, timestep, 1)?;
            let state = support_state_by_step.narrow(1, timestep, 1)?;

            // An action is relevant in proportion to both its probability and
            // its overlap with the current requirement. Clamping the overlap
            // makes one action's relevance a probability even if it adds
            // several simultaneously required facts.
            let overlap = requirement
                .broadcast_mul(&self.producer_matrix)?
                .sum(2)?
                .clamp(0.0, 1.0)?;
            let committed = (action.reshape((self.particles, self.num_actions))? * overlap)?
                .broadcast_mul(&active.reshape((self.particles, 1))?)?;

            // A producer replaces one required effect by possibly many
            // prerequisites. If those induced facts backpropagate into the
            // same producer mass, a simple sum loss has derivative
            // `-1 + missing_preconditions` and can prefer deleting precisely
            // the goal achiever that exposed the useful chain. Freeze the
            // commitment only on the demand-induction branch. The live
            // producer evidence below still rewards the action for discharging
            // its required effect, while the detached prerequisite demand
            // sends gradients to earlier producers and support states.
            let committed_demand = committed.detach();

            let producer = action
                .transpose(1, 2)?
                .broadcast_mul(&self.producer_matrix)?
                .sum(1)?
                .reshape((self.particles, 1, self.num_facts))?
                .clamp(0.0, 1.0)?;
            let unmet = (&requirement * (producer.ones_like()? - producer)?)?;
            let induced_precondition = committed_demand
                .reshape((self.particles, self.num_actions, 1))?
                .broadcast_mul(&self.precondition_matrix)?
                .sum(1)?
                .reshape((self.particles, 1, self.num_facts))?
                .clamp(0.0, 1.0)?;
            let proposed = (unmet.ones_like()?
                - ((unmet.ones_like()? - unmet)?
                    * (induced_precondition.ones_like()? - induced_precondition)?)?)?;
            requirement = (proposed.broadcast_mul(&active)?
                + requirement.broadcast_mul(&(active.ones_like()? - &active)?)?)?;

            let false_precondition = self
                .precondition_matrix
                .broadcast_mul(&(state.ones_like()? - &state)?)?
                .sum(2)?
                .broadcast_div(&self.precondition_count.reshape((1, self.num_actions))?)?;
            relevant_precondition = (relevant_precondition
                + (committed_demand * false_precondition)?
                    .sum(1)?
                    .reshape((self.particles, 1))?)?;

            let boundary = boundary_rows.narrow(1, timestep, 1)?;
            boundary_residual = (boundary_residual
                + ((&requirement * (state.ones_like()? - state)?)?
                    .sum(2)?
                    .broadcast_mul(&boundary.reshape((self.particles, 1))?))?)?;
        }

        let boundary_loss = boundary_residual.broadcast_div(&demand_mass)?;
        let relevant_precondition_loss = relevant_precondition.broadcast_div(&active_mass)?;
        let loss = (&boundary_loss + &relevant_precondition_loss)?;
        Ok(BackwardBridgeForward {
            loss,
            boundary_loss,
            relevant_precondition_loss,
        })
    }

    /// Backward causal flow from verifier-missing goals through latent temporal
    /// sources and structural producer responsibilities.
    ///
    /// Unlike [`Self::backward_goal_bridge`], this recurrence does not scale a
    /// prerequisite chain by the probability of every producer on the path.
    /// For each demanded fact, the causal-link simplex allocates responsibility
    /// among earlier rows, and a second normalized simplex allocates that row's
    /// responsibility among all operators that can add the fact. Consequently
    /// an arbitrarily small but positive producer mass still exposes its full
    /// prerequisite structure while the source-support residual separately
    /// increases the producer mass. No operator or source is selected
    /// discretely.
    pub fn backward_causal_flow(
        &self,
        action: &Tensor,
        delete: &Tensor,
        exact_state_by_step: &Tensor,
        terminal_demand: &Tensor,
        active_rows: &Tensor,
        boundary_rows: &Tensor,
        link_logits: &Tensor,
        link_temperature: &Tensor,
        first_active_row: usize,
    ) -> CandleResult<BackwardBridgeForward> {
        assert!(
            first_active_row <= self.horizon,
            "first active causal-flow row lies inside the horizon"
        );
        assert_eq!(
            action.dims(),
            &[self.particles, self.horizon, self.num_actions]
        );
        assert_eq!(
            delete.dims(),
            &[self.particles, self.horizon, self.num_facts]
        );
        assert_eq!(
            exact_state_by_step.dims(),
            &[self.particles, self.horizon, self.num_facts]
        );
        assert_eq!(terminal_demand.dims(), &[self.particles, self.num_facts]);
        assert_eq!(active_rows.dims(), &[self.particles, self.horizon, 1]);
        assert_eq!(boundary_rows.dims(), active_rows.dims());
        let link_shape = self.causal_link_shape();
        assert_eq!(link_logits.dims(), link_shape);
        assert_eq!(link_temperature.dims(), &[self.particles, 1, 1, 1]);
        if link_temperature.min_all()?.to_scalar::<f64>()? <= 0.0 {
            candle_core::bail!("backward causal-flow link temperature must be positive");
        }
        const TOLERANCE: f64 = 1e-12;
        for (name, tensor) in [
            ("action", action),
            ("delete", delete),
            ("exact state", exact_state_by_step),
            ("terminal demand", terminal_demand),
            ("active-row mask", active_rows),
            ("boundary-row mask", boundary_rows),
        ] {
            let minimum = tensor.min_all()?.to_scalar::<f64>()?;
            let maximum = tensor.max_all()?.to_scalar::<f64>()?;
            if !minimum.is_finite()
                || !maximum.is_finite()
                || minimum < -TOLERANCE
                || maximum > 1.0 + TOLERANCE
            {
                candle_core::bail!(
                    "backward causal-flow {name} must lie in [0, 1], got [{minimum}, {maximum}]"
                );
            }
        }

        // Source zero is the exact state at the unique repair boundary. The
        // first repair action must be applicable in that state in every valid
        // suffix, so its structural effects are gated exactly. Later rows stay
        // optimistic: their prerequisite structure is what this backward
        // recurrence is learning.
        let boundary_state = exact_state_by_step
            .broadcast_mul(boundary_rows)?
            .sum(1)?
            .reshape((self.particles, 1, self.num_facts))?;
        let boundary_support = self.action_support(&boundary_state)?;
        let mut first_row_mask = vec![0.0f64; self.horizon];
        if first_active_row < self.horizon {
            first_row_mask[first_active_row] = 1.0;
        }
        let first_row_mask = Tensor::from_vec(first_row_mask, (1, self.horizon, 1), &self.device)?;
        let passthrough = (first_row_mask.ones_like()? - &first_row_mask)?.broadcast_as((
            self.particles,
            self.horizon,
            self.num_actions,
        ))?;
        let gated = first_row_mask.broadcast_mul(&boundary_support)?;
        let bridge_action = (action * (passthrough + gated)?)?;
        let producer_probability = bridge_action.index_select(&self.producer_action, 2)?;
        let structural_add = producer_probability.contiguous()?.apply_op1(SegSum::new(
            self.num_facts,
            self.seg_producer_to_fact.clone(),
        ))?;
        let source = Tensor::cat(&[&boundary_state, &structural_add], 1)?;
        let source_by_link = source.transpose(1, 2)?;

        // The static triangular mask enforces strict causal order. The dynamic
        // source mask additionally confines repair to the verifier-opened
        // suffix; source zero remains available as its exact boundary state.
        let boundary_source = Tensor::ones((self.particles, 1), DTYPE, &self.device)?;
        let active_source = active_rows.squeeze(2)?;
        let source_active = Tensor::cat(&[&boundary_source, &active_source], 1)?.reshape((
            self.particles,
            1,
            1,
            self.horizon + 1,
        ))?;
        let mask = self
            .causal_link_mask
            .broadcast_as(&link_shape)?
            .broadcast_mul(&source_active)?;
        let scaled = link_logits.broadcast_div(link_temperature)?;
        let invalid = (mask.ones_like()? - &mask)?;
        let masked_logits = ((scaled * &mask)? + (invalid * -1e300)?)?;
        let row_max = masked_logits.max_keepdim(3)?;
        let unnormalized = masked_logits
            .broadcast_sub(&row_max)?
            .exp()?
            .broadcast_mul(&mask)?;
        let link = unnormalized.broadcast_div(&unnormalized.sum_keepdim(3)?)?;

        // prefix[t] contains deletes strictly before consumer row t. This is
        // the same source/consumer convention as `causal_link_forward`.
        let zero_prefix = Tensor::zeros((self.particles, 1, self.num_facts), DTYPE, &self.device)?;
        let delete_prefix = Tensor::cat(&[&zero_prefix, &delete.cumsum(1)?], 1)?;
        let intervening_delete = delete_prefix
            .unsqueeze(3)?
            .broadcast_sub(&delete_prefix.transpose(1, 2)?.unsqueeze(1)?)?
            .relu()?
            .broadcast_mul(&mask)?;

        let zero_demand = Tensor::zeros((self.particles, self.num_facts), DTYPE, &self.device)?;
        let mut demand_by_consumer = vec![zero_demand; self.horizon + 1];
        demand_by_consumer[self.horizon] = terminal_demand.clamp(0.0, 1.0)?;
        let mut source_residual = Tensor::zeros((self.particles, 1), DTYPE, &self.device)?;
        let mut threat_residual = source_residual.clone();

        for consumer in (first_active_row..=self.horizon).rev() {
            let demand = demand_by_consumer[consumer].clamp(0.0, 1.0)?;
            let consumer_link = link.narrow(1, consumer, 1)?.squeeze(1)?;
            let support = (&consumer_link * &source_by_link)?.sum(2)?;
            source_residual = (source_residual
                + ((&demand * (support.ones_like()? - support)?)?)
                    .sqr()?
                    .sum(1)?
                    .reshape((self.particles, 1))?)?;
            threat_residual = (threat_residual
                + (&consumer_link * intervening_delete.narrow(1, consumer, 1)?.squeeze(1)?)?
                    .broadcast_mul(&demand.unsqueeze(2)?)?
                    .sum(2)?
                    .sum(1)?
                    .reshape((self.particles, 1))?)?;

            // Source `s` denotes action row `s - 1`. Its normalized producer
            // responsibilities sum to one for every fact with structural
            // producer mass, so prerequisite demand does not vanish with
            // causal depth. The source-support residual above owns the
            // orthogonal task of increasing the absolute producer mass.
            let source_rows = consumer.saturating_sub(first_active_row);
            if source_rows > 0 {
                let source_weight = consumer_link
                    .narrow(2, first_active_row + 1, source_rows)?
                    .transpose(1, 2)?
                    .broadcast_mul(&demand.unsqueeze(1)?)?;
                let pair_probability = producer_probability
                    .narrow(1, first_active_row, source_rows)?
                    .contiguous()?;
                let pair_mass = structural_add
                    .narrow(1, first_active_row, source_rows)?
                    .contiguous()?
                    .index_select(&self.producer_fact, 2)?
                    .clamp(1e-12, f64::MAX)?;
                let pair_weight = source_weight
                    .contiguous()?
                    .index_select(&self.producer_fact, 2)?;
                let pair_use = pair_probability
                    .broadcast_div(&pair_mass)?
                    .broadcast_mul(&pair_weight)?;
                let precondition_use = pair_use
                    .index_select(&self.producer_precondition_pair, 2)?
                    .contiguous()?;
                let induced_by_row = precondition_use
                    .apply_op1(SegSum::new(
                        self.num_facts,
                        self.seg_producer_precondition_to_fact.clone(),
                    ))?
                    .clamp(0.0, 1.0)?;
                for offset in 0..source_rows {
                    let row = first_active_row + offset;
                    let induced = induced_by_row.narrow(1, offset, 1)?.squeeze(1)?;
                    let existing = &demand_by_consumer[row];
                    demand_by_consumer[row] = (existing.ones_like()?
                        - ((existing.ones_like()? - existing)?
                            * (induced.ones_like()? - induced)?)?)?;
                }
            }
        }

        let normalizer = terminal_demand
            .sum(1)?
            .reshape((self.particles, 1))?
            .clamp(1.0, f64::MAX)?;
        let boundary_loss = source_residual.broadcast_div(&normalizer)?;
        let relevant_precondition_loss = threat_residual.broadcast_div(&normalizer)?;
        let loss = (&boundary_loss + &relevant_precondition_loss)?;
        Ok(BackwardBridgeForward {
            loss,
            boundary_loss,
            relevant_precondition_loss,
        })
    }

    /// Roll one action tensor through the two complementary planning models.
    ///
    /// The real rollout supplies the failed-precondition loss and includes
    /// deletes. The relaxed rollout gates an action's additions by the product
    /// of its relaxed preconditions and never deletes a reached fact. This is
    /// deliberately action-only: intermediate states are consequences of the
    /// plan tensor, not independent variables that can absorb inconsistency.
    pub fn two_loss_forward(
        &self,
        action_logits: &Tensor,
        action_temperature: &Tensor,
    ) -> CandleResult<TwoLossForward> {
        self.two_loss_forward_hardened(action_logits, action_temperature, 0.0)
    }

    /// Action-only recurrence from an already normalized execution tensor.
    /// Used by temporal tokens, whose Sinkhorn assignment is part of the
    /// action parameterization and must not be passed through a second softmax.
    pub fn two_loss_forward_from_action(
        &self,
        action: Tensor,
        alpha: f64,
    ) -> CandleResult<TwoLossForward> {
        self.two_loss_forward_from_soft_action(action, alpha)
    }

    /// Roll the action-only models forward with bounded straight-through
    /// hardening.
    ///
    /// `alpha = 0` is the ordinary soft action distribution. `alpha = 1` uses
    /// exactly the same one-hot rows as argmax decoding in the forward pass.
    /// Intermediate values linearly interpolate between those endpoints. The
    /// one-hot correction is detached, so every value of `alpha` retains the
    /// softmax surrogate gradient with respect to `action_logits`:
    ///
    /// `P_used = P_soft + alpha * stop_gradient(one_hot(argmax(P_soft)) - P_soft)`.
    ///
    /// This is deliberately confined to the action-only recurrence. It is a
    /// biased straight-through estimator at nonzero `alpha`, not the gradient
    /// of the hard argmax objective.
    pub fn two_loss_forward_hardened(
        &self,
        action_logits: &Tensor,
        action_temperature: &Tensor,
        alpha: f64,
    ) -> CandleResult<TwoLossForward> {
        let soft_action = self.action_distribution(action_logits, action_temperature)?;
        self.two_loss_forward_from_soft_action(soft_action, alpha)
    }

    /// Action-only recurrence using factorized optional slots.
    pub fn two_loss_forward_factorized_hardened(
        &self,
        action_logits: &Tensor,
        action_temperature: &Tensor,
        alpha: f64,
    ) -> CandleResult<TwoLossForward> {
        let soft_action = self
            .factorized_action_distribution(action_logits, action_temperature)?
            .action;
        self.two_loss_forward_from_soft_action(soft_action, alpha)
    }

    /// Action-only recurrence with categorical anchor rows and periodic
    /// factorized insertion rows.
    pub fn two_loss_forward_hybrid_hardened(
        &self,
        action_logits: &Tensor,
        action_temperature: &Tensor,
        slack_window: usize,
        alpha: f64,
    ) -> CandleResult<TwoLossForward> {
        let soft_action = self
            .hybrid_action_distribution(action_logits, action_temperature, slack_window)?
            .action;
        self.two_loss_forward_from_soft_action(soft_action, alpha)
    }

    fn two_loss_forward_from_soft_action(
        &self,
        soft_action: Tensor,
        alpha: f64,
    ) -> CandleResult<TwoLossForward> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            candle_core::bail!(
                "straight-through hardening alpha must be finite and in [0, 1], got {alpha}"
            );
        }
        let decoded = soft_action.argmax_keepdim(2)?;
        let action_ids = Tensor::arange(0u32, self.num_actions as u32, &self.device)?.reshape((
            1,
            1,
            self.num_actions,
        ))?;
        let hard_action = decoded
            .broadcast_as((self.particles, self.horizon, self.num_actions))?
            .eq(&action_ids.broadcast_as((self.particles, self.horizon, self.num_actions))?)?
            .to_dtype(DTYPE)?;
        let straight_through_correction = (&hard_action - &soft_action)?.detach();
        let action = (&soft_action + (straight_through_correction * alpha)?)?;
        let initial = self
            .initial_row
            .broadcast_as((self.particles, 1, self.num_facts))?
            .contiguous()?;
        let mut exact = initial.clone();
        let mut relaxed = initial.clone();
        let mut failed = Tensor::zeros((self.particles, 1), DTYPE, &self.device)?;
        let mut failed_by_step = Vec::with_capacity(self.horizon);
        let mut support_only_failed_by_step = Vec::with_capacity(self.horizon);
        let mut exact_states = Vec::with_capacity(self.horizon);
        let mut causal_action_demand = Vec::with_capacity(self.horizon);
        let mut causal_add = Vec::with_capacity(self.horizon);
        let mut causal_delete = Vec::with_capacity(self.horizon);
        let mut goal_producer_rows = Vec::with_capacity(self.horizon);
        let mut no_goal_producer =
            Tensor::ones((self.particles, self.num_goals), DTYPE, &self.device)?;
        let mut premature_noop = Tensor::zeros((self.particles, 1), DTYPE, &self.device)?;

        for timestep in 0..self.horizon {
            exact_states.push(exact.clone());
            let exact_before_action = exact.clone();
            let selected_action = action.narrow(1, timestep, 1)?.contiguous()?;

            let failure = self.literal_precondition_loss(&exact, &selected_action)?;
            failed = (&failed + &failure)?;
            failed_by_step.push(failure);
            support_only_failed_by_step
                .push(self.literal_precondition_loss(&exact, &selected_action.detach())?);

            let exact_support = self.action_support(&exact)?;
            let supported_exact_action = (&selected_action * exact_support)?;
            let (exact_add, exact_change) = self.effect_masses(&exact, &supported_exact_action)?;
            let exact_delete = (&exact_change - &exact_add)?;
            causal_action_demand.push(self.action_fact_demand(&supported_exact_action)?);
            causal_add.push(exact_add.clone());
            causal_delete.push(exact_delete);
            exact = ((&exact * (exact_change.ones_like()? - exact_change)?)? + exact_add)?;

            // Producer pressure deliberately sees raw selected-action mass:
            // it requests that some producer of each goal exist before that
            // producer's prerequisites have been learned. Applicability still
            // gates the actual relaxed transition below, so raw producer mass
            // can shape the plan but can never fabricate achieved support.
            let (producer_add, _) = self.effect_masses(&relaxed, &selected_action)?;
            let goal_producer = producer_add.index_select(&self.goal_fact, 2)?;
            no_goal_producer = (&no_goal_producer
                * (goal_producer.ones_like()? - &goal_producer)?
                    .reshape((self.particles, self.num_goals))?)?;
            goal_producer_rows.push(goal_producer);
            let support = self.action_support(&relaxed)?;
            let supported_action = (&selected_action * support)?;
            let (relaxed_add, _) = self.effect_masses(&relaxed, &supported_action)?;
            // No-op pressure follows the real recurrent state. A monotone
            // relaxation would treat a goal that was subsequently clobbered
            // as satisfied and incorrectly make later no-ops free.
            let current_goal = exact_before_action
                .contiguous()?
                .index_select(&self.goal_fact, 2)?;
            let goal_missing = (current_goal.ones_like()? - current_goal)?.mean(2)?;
            let noop_mass = selected_action
                .narrow(2, self.num_actions - 1, 1)?
                .reshape((self.particles, 1))?;
            premature_noop = (&premature_noop + (noop_mass * goal_missing)?)?;
            relaxed = (&relaxed + ((relaxed.ones_like()? - &relaxed)? * relaxed_add)?)?;
        }

        let goal_support = relaxed.contiguous()?.index_select(&self.goal_fact, 2)?;
        let terminal_loss = (goal_support.ones_like()? - goal_support)?
            .reshape((self.particles, self.num_goals))?;
        let exact_goal_support = exact.contiguous()?.index_select(&self.goal_fact, 2)?;
        let terminal_goal_by_goal = (exact_goal_support.ones_like()? - exact_goal_support)?
            .reshape((self.particles, self.num_goals))?;
        let exact_add = Tensor::cat(&causal_add, 1)?;
        let exact_delete = Tensor::cat(&causal_delete, 1)?;
        let goal_add = exact_add.index_select(&self.goal_fact, 2)?;
        let goal_delete = exact_delete.index_select(&self.goal_fact, 2)?;
        // `survival_after[t] = prod_{u>t} (1 - delete[u])`. Building it
        // backwards makes the candidate at row t depend directly on every
        // later threat without routing its gradient through all earlier state
        // recurrences.
        let mut survival_after =
            Tensor::ones((self.particles, 1, self.num_goals), DTYPE, &self.device)?;
        let mut no_surviving_source =
            Tensor::ones((self.particles, self.num_goals), DTYPE, &self.device)?;
        for timestep in (0..self.horizon).rev() {
            let add = goal_add.narrow(1, timestep, 1)?;
            let surviving_add =
                (&add * &survival_after)?.reshape((self.particles, self.num_goals))?;
            no_surviving_source =
                (&no_surviving_source * (surviving_add.ones_like()? - surviving_add)?)?;
            let delete = goal_delete.narrow(1, timestep, 1)?;
            survival_after = (&survival_after * (delete.ones_like()? - delete)?)?;
        }
        let initially_surviving = (initial.index_select(&self.goal_fact, 2)? * &survival_after)?
            .reshape((self.particles, self.num_goals))?;
        no_surviving_source =
            (&no_surviving_source * (initially_surviving.ones_like()? - initially_surviving)?)?;
        // The accumulated product is precisely `1 - evidence`: it is the
        // probability-like event that neither an initial source nor any added
        // source survives to the terminal row.
        let surviving_goal_by_goal = no_surviving_source;
        let initial_goal = initial.index_select(&self.goal_fact, 2)?;
        let producer_peak = Tensor::cat(&goal_producer_rows, 1)?.max(1)?;
        let producer_evidence = initial_goal
            .reshape((self.particles, self.num_goals))?
            .maximum(&producer_peak)?;
        let producer_loss = (producer_evidence.ones_like()? - producer_evidence)?;
        // `q_g = 1 - (1 - I_g) prod_t (1 - A_tg)` is the probability that a
        // goal is initially true or has at least one (raw) producer. Unlike a
        // max-only objective, `-log(q_g)` gives every producer row a gradient.
        let initial_goal = initial_goal.reshape((self.particles, self.num_goals))?;
        let no_goal_evidence = ((initial_goal.ones_like()? - initial_goal)? * no_goal_producer)?;
        let some_goal_producer_probability = (no_goal_evidence.ones_like()? - no_goal_evidence)?;
        let some_goal_producer_loss = some_goal_producer_probability
            .clamp(1e-300, 1.0)?
            .log()?
            .neg()?;
        let relaxed_goal_by_goal =
            ((terminal_loss + (producer_loss * 2.0)?)? + &some_goal_producer_loss)?;
        let premature_noop = (premature_noop * 0.25)?;
        let relaxed_goal =
            (relaxed_goal_by_goal.sum(1)?.reshape((self.particles, 1))? + &premature_noop)?;
        let action_integrality = self
            .action_integrality_per_particle(&action)?
            .sum(1)?
            .reshape((self.particles, 1))?;

        Ok(TwoLossForward {
            action,
            exact_state_by_step: Tensor::cat(&exact_states, 1)?,
            relaxed_goal,
            relaxed_goal_by_goal,
            terminal_goal_by_goal,
            surviving_goal_by_goal,
            some_goal_producer_probability,
            some_goal_producer_loss,
            premature_noop,
            failed_precondition: failed,
            failed_precondition_by_step: Tensor::cat(&failed_by_step, 1)?,
            support_only_failed_precondition_by_step: Tensor::cat(&support_only_failed_by_step, 1)?,
            action_integrality,
            causal: CausalLinkInput {
                action_demand: Tensor::cat(&causal_action_demand, 1)?,
                add: exact_add,
                delete: exact_delete,
            },
        })
    }

    /// Build fact-grouped latent causal-link witnesses.
    ///
    /// `link_logits[m, t, f, s]` chooses a source for fact `f` demanded by
    /// consumer row `t`. Source zero is the fixed initial state; source `s + 1`
    /// is the actual last-write addition mass of action row `s`. A structural
    /// triangular mask makes future and self-support exactly impossible.
    ///
    /// This layer is a differentiable proof witness, not another action alias:
    /// it must name an extant source and pay for every intervening delete. The
    /// three returned losses are divided by that particle's active demand, so
    /// neither inactive slots nor another particle can dilute them.
    pub fn causal_link_forward(
        &self,
        forward: &Forward,
        link_logits: &Tensor,
        link_temperature: &Tensor,
    ) -> CandleResult<CausalLinkForward> {
        let extra_demand = Tensor::zeros(
            (self.particles, self.horizon + 1, self.num_facts),
            DTYPE,
            &self.device,
        )?;
        self.causal_link_forward_with_demand(forward, link_logits, link_temperature, &extra_demand)
    }

    /// Build causal links from applicability-supported evidence emitted by a
    /// recurrent action-only rollout.
    pub fn causal_link_forward_from_input(
        &self,
        input: &CausalLinkInput,
        link_logits: &Tensor,
        link_temperature: &Tensor,
    ) -> CandleResult<CausalLinkForward> {
        let extra_demand = Tensor::zeros(
            (self.particles, self.horizon + 1, self.num_facts),
            DTYPE,
            &self.device,
        )?;
        self.causal_link_forward_from_input_with_demand(
            input,
            link_logits,
            link_temperature,
            &extra_demand,
        )
    }

    /// As [`Self::causal_link_forward`], with additional non-negative demand
    /// supplied by exact verifier feedback.
    ///
    /// A first failed precondition is a concrete causal obligation even when
    /// the soft action distribution subsequently moves mass away from the
    /// rejected action. Keeping that demand explicit prevents the optimizer
    /// from "repairing" a failure merely by erasing the action that exposed it.
    pub fn causal_link_forward_with_demand(
        &self,
        forward: &Forward,
        link_logits: &Tensor,
        link_temperature: &Tensor,
        extra_demand: &Tensor,
    ) -> CandleResult<CausalLinkForward> {
        let input = CausalLinkInput {
            action_demand: self.action_fact_demand(&forward.action)?,
            add: forward.add.clone(),
            delete: forward.delete.clone(),
        };
        self.causal_link_forward_from_input_with_demand(
            &input,
            link_logits,
            link_temperature,
            extra_demand,
        )
    }

    /// As [`Self::causal_link_forward_from_input`], with additional
    /// non-negative demand supplied by exact verifier feedback.
    pub fn causal_link_forward_from_input_with_demand(
        &self,
        input: &CausalLinkInput,
        link_logits: &Tensor,
        link_temperature: &Tensor,
        extra_demand: &Tensor,
    ) -> CandleResult<CausalLinkForward> {
        let link_shape = self.causal_link_shape();
        if link_logits.dims() != link_shape {
            candle_core::bail!(
                "causal-link logits have shape {:?}, expected {:?}",
                link_logits.dims(),
                link_shape
            );
        }
        let temperature_shape = [self.particles, 1, 1, 1];
        if link_temperature.dims() != temperature_shape {
            candle_core::bail!(
                "causal-link temperature has shape {:?}, expected {:?}",
                link_temperature.dims(),
                temperature_shape
            );
        }
        if link_temperature.min_all()?.to_scalar::<f64>()? <= 0.0 {
            candle_core::bail!("causal-link temperature must be strictly positive");
        }
        let demand_shape = [self.particles, self.horizon + 1, self.num_facts];
        if extra_demand.dims() != demand_shape {
            candle_core::bail!(
                "extra causal demand has shape {:?}, expected {:?}",
                extra_demand.dims(),
                demand_shape
            );
        }
        if extra_demand.min_all()?.to_scalar::<f64>()? < 0.0 {
            candle_core::bail!("extra causal demand must be non-negative");
        }
        let effect_shape = [self.particles, self.horizon, self.num_facts];
        if input.action_demand.dims() != effect_shape
            || input.add.dims() != effect_shape
            || input.delete.dims() != effect_shape
        {
            candle_core::bail!(
                "causal-link input tensors have incompatible demand/add/delete shapes: {:?}, {:?}, {:?}; expected {:?}",
                input.action_demand.dims(),
                input.add.dims(),
                input.delete.dims(),
                effect_shape
            );
        }
        const PROBABILITY_ROUNDOFF_TOLERANCE: f64 = 1e-12;
        for (name, tensor) in [
            ("action demand", &input.action_demand),
            ("addition", &input.add),
            ("deletion", &input.delete),
        ] {
            let minimum = tensor.min_all()?.to_scalar::<f64>()?;
            let maximum = tensor.max_all()?.to_scalar::<f64>()?;
            if !minimum.is_finite()
                || !maximum.is_finite()
                || minimum < -PROBABILITY_ROUNDOFF_TOLERANCE
                || maximum > 1.0 + PROBABILITY_ROUNDOFF_TOLERANCE
            {
                candle_core::bail!(
                    "causal-link {name} must lie in [0, 1], got [{minimum}, {maximum}]"
                );
            }
        }
        // Products and complements of valid f64 probabilities can land a few
        // ulps outside the closed interval. After the strict tolerance check,
        // projection is the unique semantically correct representation of the
        // same probability; larger violations still fail above.
        let action_demand = input.action_demand.clamp(0.0, 1.0)?;
        let add = input.add.clamp(0.0, 1.0)?;
        let delete = input.delete.clamp(0.0, 1.0)?;

        // Action consumers demand their precondition facts. Terminal goals are
        // constants: making demand depend on a soft terminal state would let an
        // optimizer erase the obligation by making the goal false.
        let goal_demand = self
            .goal_demand_row
            .broadcast_as((self.particles, 1, self.num_facts))?
            .contiguous()?;
        let base_demand = Tensor::cat(&[&action_demand, &goal_demand], 1)?;
        let demand = (&base_demand + extra_demand)?;

        // Stable masked softmax. Optimizer logits are finite; multiplication by
        // the exact mask after exp makes future/self entries bitwise zero.
        let mask = self
            .causal_link_mask
            .broadcast_as(&link_shape)?
            .contiguous()?;
        let scaled = link_logits.broadcast_div(link_temperature)?;
        // Candle's CPU `where` kernel does not support f64. Arithmetic masking
        // is exact here because validated/clipped logits are finite and the
        // post-exp multiplication below forces invalid entries to bitwise zero.
        let invalid = (mask.ones_like()? - &mask)?;
        let masked_logits = ((scaled * &mask)? + (invalid * -1e300)?)?;
        let row_max = masked_logits.max_keepdim(3)?;
        let unnormalized = masked_logits
            .broadcast_sub(&row_max)?
            .exp()?
            .broadcast_mul(&mask)?;
        let link = unnormalized.broadcast_div(&unnormalized.sum_keepdim(3)?)?;

        let initial = self
            .initial_row
            .broadcast_as((self.particles, 1, self.num_facts))?
            .contiguous()?;
        let source = Tensor::cat(&[&initial, &add], 1)?;
        let source_by_link = source
            .transpose(1, 2)?
            .unsqueeze(1)?
            .broadcast_as(&link_shape)?;
        let demand_by_link = demand.unsqueeze(3)?.broadcast_as(&link_shape)?;

        // Each active link must name a source that actually exists. Use the
        // negative log of expected source support, not
        // `relu(demand * link - source)`: the latter can make a missing source
        // free by diffusing one witness over many tiny link probabilities.
        // The logarithm also amplifies the producer gradient precisely when
        // source evidence is scarce. `SOURCE_FLOOR` keeps that gradient finite;
        // the affine floor preserves loss zero at support one.
        //
        // Sources are proof facts rather than consumable resources, so
        // independent consumers may correctly reuse the same achiever.
        const SOURCE_FLOOR: f64 = 1e-6;
        let used_link = (&link * &demand_by_link)?;
        let source_support = (&link * source_by_link)?.sum(3)?;
        let source_violation = ((source_support * (1.0 - SOURCE_FLOOR))? + SOURCE_FLOOR)?
            .log()?
            .neg()?
            .broadcast_mul(&demand)?;

        // prefix[t] contains deletes in rows strictly before consumer t.
        // Subtracting prefix[source] leaves exactly the rows after the named
        // producer and before the consumer (or all prior rows for source zero).
        let zero_prefix = Tensor::zeros((self.particles, 1, self.num_facts), DTYPE, &self.device)?;
        let delete_prefix = Tensor::cat(&[&zero_prefix, &delete.cumsum(1)?], 1)?;
        let consumer_prefix = delete_prefix.unsqueeze(3)?;
        let source_prefix = delete_prefix.transpose(1, 2)?.unsqueeze(1)?;
        let intervening_delete = consumer_prefix
            .broadcast_sub(&source_prefix)?
            .relu()?
            .broadcast_mul(&mask)?;
        let threat = (&used_link * intervening_delete)?;

        let active_consumer_mass = demand
            .sum(2)?
            .sum(1)?
            .reshape((self.particles, 1))?
            .detach();
        // Normalize by the unweighted obligations, not by verifier weights.
        // Dividing by weighted demand would cancel projected weight ascent:
        // multiplying one failed fact or every missing goal by 80 would leave
        // the causal loss essentially unchanged. The detached baseline keeps
        // task-size scaling stable while exact-feedback weights genuinely
        // change the loss landscape.
        let normalizer = base_demand
            .sum(2)?
            .sum(1)?
            .reshape((self.particles, 1))?
            .detach()
            .clamp(1.0, f64::MAX)?;
        let source_loss = source_violation
            .sum(2)?
            .sum(1)?
            .reshape((self.particles, 1))?
            .broadcast_div(&normalizer)?;
        let threat_loss = threat
            .sum(3)?
            .sum(2)?
            .sum(1)?
            .reshape((self.particles, 1))?
            .broadcast_div(&normalizer)?;
        let link_sum_squares = link.sqr()?.sum(3)?;
        let link_fractionality = (link_sum_squares.ones_like()? - link_sum_squares)?.relu()?;
        let link_integrality = (link_fractionality * &demand)?
            .sum(2)?
            .sum(1)?
            .reshape((self.particles, 1))?
            .broadcast_div(&normalizer)?;
        // Keep exact live maxima in addition to normalized aggregate losses.
        // One decisive unsupported source or threat must remain observable even
        // when thousands of other obligations are already satisfied.
        let max_source_violation = source_violation
            .max(2)?
            .max(1)?
            .reshape((self.particles, 1))?;
        let max_threat_violation = threat
            .max(3)?
            .max(2)?
            .max(1)?
            .reshape((self.particles, 1))?;

        Ok(CausalLinkForward {
            demand,
            link,
            source_loss,
            threat_loss,
            link_integrality,
            max_source_violation,
            max_threat_violation,
            active_consumer_mass,
        })
    }

    /// Action-integrality penalty `1 − Σ_a P²` per `(particle, timestep)`, zero
    /// exactly on one-hot action rows. Shape `[M, H, 1]`.
    ///
    /// Kept per particle so the weight can follow each particle's own annealing
    /// phase: forcing a melting particle to commit at the same time defeats the
    /// point of melting it.
    pub fn action_integrality_per_particle(&self, action: &Tensor) -> CandleResult<Tensor> {
        let sum_squares = action.sqr()?.sum_keepdim(2)?;
        sum_squares.ones_like()? - sum_squares
    }

    /// Symmetry-breaking residual that makes no-op mass a suffix.
    ///
    /// Internal no-ops are removable from every deterministic classical plan,
    /// so requiring `P(noop,t) <= P(noop,t+1)` loses no bounded solutions. At
    /// integrality it is a differentiable STOP encoding: real actions form a
    /// prefix and no-op pads the remainder.
    pub fn noop_suffix_penalty(&self, action: &Tensor) -> CandleResult<Tensor> {
        if self.horizon <= 1 {
            return Tensor::zeros((self.particles, 1), DTYPE, &self.device);
        }
        let noop = action.narrow(2, self.num_actions - 1, 1)?;
        let current = noop.narrow(1, 0, self.horizon - 1)?;
        let next = noop.narrow(1, 1, self.horizon - 1)?;
        (current - next)?
            .relu()?
            .mean(1)?
            .reshape((self.particles, 1))
    }

    /// Local insertion-slack residual, one scalar per particle.
    ///
    /// A zero residual means every complete window retains at least one unit
    /// of no-op mass. This is a soft geometric prior, not a bounded-plan
    /// restriction: it is scheduled to zero before the final polish tail.
    pub fn slot_slack_penalty(&self, action: &Tensor, window: usize) -> CandleResult<Tensor> {
        if window == 0 || self.horizon < window {
            return Tensor::zeros((self.particles, 1), DTYPE, &self.device);
        }
        if window < 2 {
            candle_core::bail!("slot slack window must be zero or at least 2, got {window}");
        }
        assert_eq!(
            action.dims(),
            &[self.particles, self.horizon, self.num_actions],
            "slot slack receives a full action distribution"
        );
        let noop = action.narrow(2, self.num_actions - 1, 1)?;
        let occupancy = (noop.ones_like()? - noop)?;
        let mut residuals = Vec::with_capacity(self.horizon - window + 1);
        for start in 0..=self.horizon - window {
            let active = occupancy.narrow(1, start, window)?.sum(1)?;
            residuals.push((active - (window - 1) as f64)?.relu()?.sqr()?);
        }
        let refs = residuals.iter().collect::<Vec<_>>();
        Tensor::cat(&refs, 1)?.mean(1)?.reshape((self.particles, 1))
    }

    /// State-integrality penalty `1 − Σ_d S²` for every individual variable
    /// and `(particle, timestep)`. Shape `[M, H, V]`.
    ///
    /// Row 0 is skipped: it is the fixed initial state, integral by
    /// construction. Keeping `V` intact is essential for bottleneck losses: a
    /// single fractional variable must not disappear into a variable mean.
    pub fn state_integrality_per_particle(&self, state: &Tensor) -> CandleResult<Tensor> {
        let rows = state.narrow(1, 1, self.horizon)?.contiguous()?;
        let per_variable = rows.sqr()?.contiguous()?.apply_op1(SegSum::new(
            self.num_variables,
            self.seg_fact_to_var.clone(),
        ))?;
        per_variable.ones_like()? - per_variable
    }

    /// Direct-transcription precondition residual with optional consumer
    /// gradient protection per particle.
    ///
    /// `protect_consumer[m]=1` leaves the forward constraint `P[a] <= S[f]`
    /// unchanged but detaches `P` on that particle, so repair gradients can
    /// only increase support. Zero recovers the ordinary live constraint.
    pub fn protected_precondition_residual(
        &self,
        action: &Tensor,
        state: &Tensor,
        protect_consumer: &Tensor,
    ) -> CandleResult<Tensor> {
        assert_eq!(
            action.dims(),
            &[self.particles, self.horizon, self.num_actions]
        );
        assert_eq!(
            state.dims(),
            &[self.particles, self.horizon + 1, self.num_facts]
        );
        let protected_action = self.action_with_gradient_protection(action, protect_consumer)?;
        let current = state.narrow(1, 0, self.horizon)?.contiguous()?;
        let selected_action = protected_action.index_select(&self.pre_action, 2)?;
        let selected_fact = current.index_select(&self.pre_fact, 2)?;
        (selected_action - selected_fact)?.relu()
    }

    /// Direct-transcription transition residuals with optional action-gradient
    /// protection per particle.
    ///
    /// The forward values are identical to [`Forward::transition`]. With a
    /// protection value of one, however, a newly introduced goal achiever
    /// cannot reduce its effect/state mismatch by deleting itself; gradients
    /// instead move the adjacent lifted state rows to express that transition.
    pub fn protected_transition_residual(
        &self,
        action: &Tensor,
        state: &Tensor,
        protect_action: &Tensor,
    ) -> CandleResult<[Tensor; 4]> {
        assert_eq!(
            action.dims(),
            &[self.particles, self.horizon, self.num_actions]
        );
        assert_eq!(
            state.dims(),
            &[self.particles, self.horizon + 1, self.num_facts]
        );
        let protected_action = self.action_with_gradient_protection(action, protect_action)?;
        let current = state.narrow(1, 0, self.horizon)?.contiguous()?;
        let next = state.narrow(1, 1, self.horizon)?.contiguous()?;
        let (add, change) = self.effect_masses(&current, &protected_action)?;
        let delete = (&change - &add)?;
        Ok([
            (&add - &next)?.relu()?,
            ((&next + &delete)? - 1.0)?.relu()?,
            (((&current - &delete)? - &next)?).relu()?,
            (((&next - &current)? - &add)?).relu()?,
        ])
    }

    fn action_with_gradient_protection(
        &self,
        action: &Tensor,
        protection: &Tensor,
    ) -> CandleResult<Tensor> {
        assert_eq!(
            protection.dims(),
            &[self.particles, 1, 1],
            "action-gradient protection has one scalar per particle"
        );
        let minimum = protection.min_all()?.to_scalar::<f64>()?;
        let maximum = protection.max_all()?.to_scalar::<f64>()?;
        if minimum < 0.0 || maximum > 1.0 {
            candle_core::bail!(
                "action-gradient protection must lie in [0, 1], got [{minimum}, {maximum}]"
            );
        }
        action.broadcast_mul(&(protection.ones_like()? - protection)?)?
            + action.detach().broadcast_mul(protection)?
    }

    /// One forward pass: distributions and every residual family.
    pub fn forward(
        &self,
        action_logits: &Tensor,
        state_logits: &Tensor,
        action_temperature: &Tensor,
        state_temperature: &Tensor,
    ) -> CandleResult<Forward> {
        let scaled = action_logits.broadcast_div(action_temperature)?;
        let action_log_probability = candle_nn::ops::log_softmax(&scaled, 2)?;
        // Preserve the historical softmax arithmetic exactly. The separately
        // computed stable log probabilities are used only by ranking/KL-like
        // objectives and must not perturb a categorical ablation's trajectory.
        let action = self.action_distribution(action_logits, action_temperature)?;
        self.forward_from_action(
            action,
            action_log_probability,
            state_logits,
            state_temperature,
        )
    }

    /// One direct-transcription forward pass using factorized optional slots.
    pub fn forward_factorized(
        &self,
        action_logits: &Tensor,
        state_logits: &Tensor,
        action_temperature: &Tensor,
        state_temperature: &Tensor,
    ) -> CandleResult<Forward> {
        let slots = self.factorized_action_distribution(action_logits, action_temperature)?;
        self.forward_from_action(
            slots.action,
            slots.log_action,
            state_logits,
            state_temperature,
        )
    }

    /// Direct-transcription forward pass with categorical anchor rows and
    /// periodic factorized insertion rows.
    pub fn forward_hybrid(
        &self,
        action_logits: &Tensor,
        state_logits: &Tensor,
        action_temperature: &Tensor,
        state_temperature: &Tensor,
        slack_window: usize,
    ) -> CandleResult<Forward> {
        let slots =
            self.hybrid_action_distribution(action_logits, action_temperature, slack_window)?;
        self.forward_from_action(
            slots.action,
            slots.log_action,
            state_logits,
            state_temperature,
        )
    }

    /// Direct transcription from an already normalized action tensor.
    pub fn forward_from_action_distribution(
        &self,
        action: Tensor,
        log_action: Tensor,
        state_logits: &Tensor,
        state_temperature: &Tensor,
    ) -> CandleResult<Forward> {
        self.forward_from_action(action, log_action, state_logits, state_temperature)
    }

    fn forward_from_action(
        &self,
        action: Tensor,
        action_log_probability: Tensor,
        state_logits: &Tensor,
        state_temperature: &Tensor,
    ) -> CandleResult<Forward> {
        let state = self.state_distribution(state_logits, state_temperature)?;

        // `narrow` yields a view, and `index_select` requires contiguous input.
        let current = state.narrow(1, 0, self.horizon)?.contiguous()?;
        let next = state.narrow(1, 1, self.horizon)?.contiguous()?;

        // Preconditions: P[t,a] <= S[t,f] for each incidence.
        let precondition = {
            let selected_action = action.index_select(&self.pre_action, 2)?;
            let selected_fact = current.index_select(&self.pre_fact, 2)?;
            (selected_action - selected_fact)?.relu()?
        };

        let (add, change) = self.effect_masses(&current, &action)?;
        let delete = (&change - &add)?;

        let transition = [
            (&add - &next)?.relu()?,
            ((&next + &delete)? - 1.0)?.relu()?,
            (((&current - &delete)? - &next)?).relu()?,
            (((&next - &current)? - &add)?).relu()?,
        ];

        let goal = {
            let terminal = state.narrow(1, self.horizon, 1)?.contiguous()?;
            let selected = terminal.index_select(&self.goal_fact, 2)?;
            (selected.ones_like()? - selected)?
        };

        Ok(Forward {
            action,
            action_log_probability,
            state,
            add,
            delete,
            precondition,
            transition,
            goal,
        })
    }
}
