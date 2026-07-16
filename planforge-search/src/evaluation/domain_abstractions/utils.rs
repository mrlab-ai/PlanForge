use anyhow::{Context, Result, anyhow};
use std::collections::{BTreeSet, HashSet};
use std::fmt::Write as _;

use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::numeric_task::{AbstractNumericTask, ExplicitFact};
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::int_packer::IntDoublePacker;
use tracing::debug;

use super::cegar::flaw_search::Flaw;
use super::comparison_expression::Interval;
use super::domain_abstraction::ComparisonAxiomIndex;
use super::domain_abstraction::NumericPartitions;
use super::domain_abstraction_factory::{
    AbstractDistanceTable, DomainAbstractionFactory, WildcardPlanResult,
};
use crate::evaluation::domain_abstractions::abstract_operator_generator::DomainMapping;
use crate::evaluation::domain_abstractions::cegar::flaw_search::SplitDirection;
use crate::evaluation::domain_abstractions::cegar::flaw_search::progression::{
    get_progression_numeric_deviation_flaws, get_progression_precondition_flaws,
};
use crate::evaluation::domain_abstractions::cegar::flaw_search::state::progress;

pub(crate) fn compute_abstraction_size_u128(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
) -> Option<u128> {
    let mut size: u128 = 1;
    for &d in domain_sizes.iter() {
        let du = u128::try_from(d).ok()?;
        if du == 0 {
            return Some(0);
        }
        size = size.checked_mul(du)?;
    }
    for &p in numeric_domain_sizes.iter() {
        let pu = u128::try_from(p).ok()?;
        if pu == 0 {
            return Some(0);
        }
        size = size.checked_mul(pu)?;
    }
    Some(size)
}

#[allow(unused)]
pub(crate) fn identity_domain_mapping_and_sizes(
    task: &dyn AbstractNumericTask,
) -> Result<(DomainMapping, Vec<usize>)> {
    let num_vars = task.get_num_variables();
    let derived_prop: HashSet<usize> = task
        .comparison_axioms()
        .iter()
        .map(|ax| ax.get_affected_var_id())
        .collect();

    let mut domain_mapping: DomainMapping = Vec::with_capacity(num_vars);
    let mut domain_sizes: Vec<usize> = Vec::with_capacity(num_vars);
    for var_id in 0..num_vars {
        if derived_prop.contains(&(var_id)) {
            domain_mapping.push(vec![0, 1, 2]);
            domain_sizes.push(3);
        } else {
            let size = task
                .get_variable_domain_size(var_id)
                .map_err(|e| anyhow!(e.to_string()))
                .with_context(|| format!("failed to get domain size for variable {var_id}"))?;
            domain_mapping.push((0..size).collect());
            domain_sizes.push(size);
        }
    }

    Ok((domain_mapping, domain_sizes))
}

pub(crate) fn debug_print_abstraction_stats(
    iteration: usize,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
) {
    let prop_vars = domain_sizes.len();
    let num_vars = numeric_domain_sizes.len();
    let refined_props = domain_sizes.iter().filter(|&&s| s > 1).count();
    let refined_nums = numeric_domain_sizes.iter().filter(|&&s| s > 1).count();
    let size = compute_abstraction_size_u128(domain_sizes, numeric_domain_sizes)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "<overflow>".to_string());

    let prop_max = domain_sizes.iter().copied().max().unwrap_or(0);
    let num_max = numeric_domain_sizes.iter().copied().max().unwrap_or(0);

    debug!(
        "[CEGAR] iteration {iteration}: abstract_states={size} (prop_vars={prop_vars}, num_vars={num_vars}, refined_prop={refined_props}, refined_num={refined_nums}, max_prop_size={prop_max}, max_num_parts={num_max})"
    );
}

pub(crate) fn debug_print_refinement_summary(
    before: Option<u128>,
    after: Option<u128>,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    refined: bool,
) {
    let before_s = before
        .map(|v| v.to_string())
        .unwrap_or_else(|| "<overflow>".to_string());
    let after_s = after
        .map(|v| v.to_string())
        .unwrap_or_else(|| "<overflow>".to_string());
    debug!("[Refine] refined={refined} abstract_states: {before_s} -> {after_s}");

    let mut refined_props: Vec<(usize, usize)> = domain_sizes
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| (s > 1).then_some((i, s)))
        .collect();
    refined_props.sort_by_key(|(i, _)| *i);
    let refined_nums: Vec<(usize, usize)> = numeric_domain_sizes
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| (s > 1).then_some((i, s)))
        .collect();

    if !refined_props.is_empty() {
        let preview = 30usize;
        let mut line = String::new();
        let _ = write!(
            &mut line,
            "[Refine] propositional splits: {} vars",
            refined_props.len()
        );
        for (i, s) in refined_props.iter().take(preview) {
            let _ = write!(&mut line, " v{i}=>{s}");
        }
        if refined_props.len() > preview {
            let _ = write!(&mut line, " ...");
        }
        debug!("{line}");
    }
    if !refined_nums.is_empty() {
        let preview = 30usize;
        let mut line = String::new();
        let _ = write!(
            &mut line,
            "[Refine] numeric splits: {} vars",
            refined_nums.len()
        );
        for (i, s) in refined_nums.iter().take(preview) {
            let _ = write!(&mut line, " n{i}=>{s}");
        }
        if refined_nums.len() > preview {
            let _ = write!(&mut line, " ...");
        }
        debug!("{line}");
    }
}

pub(crate) fn debug_print_flaws(flaws: &[Flaw]) {
    debug!("[Flaws] count={}", flaws.len());
    let max = 200usize;
    let shown = flaws.len().min(max);
    for (i, flaw) in flaws.iter().take(shown).enumerate() {
        match flaw {
            Flaw::Propositional(pf) => {
                debug!(
                    "  {i}: PropFlaw fact=(var={}, val={}) deps={}",
                    pf.fact.var(),
                    pf.fact.value(),
                    pf.dependent_numeric_flaws.len()
                );
                for (j, nf) in pf.dependent_numeric_flaws.iter().enumerate() {
                    debug!(
                        "      - dep[{j}]: NumericFlaw var={} value={} include_in_lower={}",
                        nf.numeric_var_id, nf.value, nf.include_in_lower
                    );
                }
            }
            Flaw::Numeric(nf) => {
                debug!(
                    "  {i}: NumericFlaw var={} value={} include_in_lower={}",
                    nf.numeric_var_id, nf.value, nf.include_in_lower
                );
            }
        }
    }
    if flaws.len() > max {
        debug!("[Flaws] (truncated: showing {shown} of {})", flaws.len());
    }
}

pub(crate) fn fmt_interval(iv: Interval) -> String {
    let l = if iv.lower_closed { '[' } else { '(' };
    let r = if iv.upper_closed { ']' } else { ')' };
    let lo = fmt_f64_compact(iv.lower);
    let hi = fmt_f64_compact(iv.upper);
    format!("{l}{lo}, {hi}{r}")
}

pub(crate) fn fmt_f64_compact(v: f64) -> String {
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v.is_sign_negative() {
            "-inf".to_string()
        } else {
            "inf".to_string()
        };
    }
    let mut s = format!("{v}");
    let is_scientific = s.contains('e') || s.contains('E');
    if !is_scientific && let Some(dot) = s.find('.') {
        let (head, tail) = s.split_at(dot + 1);
        let trimmed_tail = tail.trim_end_matches('0');
        s = if trimmed_tail.is_empty() {
            head.trim_end_matches('.').to_string()
        } else {
            format!("{head}{trimmed_tail}")
        };
    }
    if s == "-0" { "0".to_string() } else { s }
}

#[inline]
fn interval_contains_value_tolerant(iv: &Interval, value: f64) -> bool {
    if value.is_nan() || iv.is_empty() {
        return false;
    }

    // Parity-over-quality: exact match at partition boundaries, matching
    // C++ numeric-FD's `get_partition_index`. Tolerant comparison drifted
    // boundary-aligned values into the wrong partition relative to C++.
    let lower_ok = if iv.lower == f64::NEG_INFINITY {
        true
    } else if iv.lower_closed {
        value >= iv.lower
    } else {
        value > iv.lower
    };

    let upper_ok = if iv.upper == f64::INFINITY {
        true
    } else if iv.upper_closed {
        value <= iv.upper
    } else {
        value < iv.upper
    };

    lower_ok && upper_ok
}

pub(crate) fn partition_for_value(partitions: &[Interval], value: f64) -> Option<usize> {
    if partitions.len() <= 8 {
        return partitions
            .iter()
            .position(|iv| interval_contains_value_tolerant(iv, value));
    }

    let mut low = 0;
    let mut high = partitions.len();
    while low < high {
        let mid = low + (high - low) / 2;
        let iv = &partitions[mid];
        let below_lower = if iv.lower.is_finite() {
            let tolerance = float_tolerance::tolerance(value, iv.lower);
            value < iv.lower - tolerance
                || (value - iv.lower).abs() <= tolerance && !iv.lower_closed
        } else {
            false
        };
        if below_lower {
            high = mid;
            continue;
        }

        let above_upper = if iv.upper.is_finite() {
            let tolerance = float_tolerance::tolerance(value, iv.upper);
            value > iv.upper + tolerance
                || (value - iv.upper).abs() <= tolerance && !iv.upper_closed
        } else {
            false
        };
        if above_upper {
            low = mid + 1;
            continue;
        }

        let mut first = mid;
        while first > 0 && interval_contains_value_tolerant(&partitions[first - 1], value) {
            first -= 1;
        }
        return Some(first);
    }
    None
}

/// O(1) descriptor for a partition layout that is contiguous, uniform-width,
/// and surrounded by at most one unbounded interval on each side.
///
/// Examples that fit (sailing's typical CEGAR layout):
///   `(-∞, 0.5)  [0.5, 1.0)  [1.0, 1.5)  …  [N-0.5, N)  [N, +∞)`
///
/// When this pattern holds we can skip `partition_for_value`'s binary search
/// and compute the partition index directly. With ~200 partitions per
/// numeric var, the saving is ~log₂(200) tolerant float comparisons per
/// hash, per state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct EquispacedPartitioning {
    /// Lower bound of the first fully-finite partition.
    base: f64,
    /// Width of one finite partition.
    step: f64,
    /// Number of fully-finite partitions covering `[base, base+step*finite_count)`.
    finite_count: usize,
    /// Index of the first finite partition (0 if no lower unbounded tail, 1 otherwise).
    finite_offset: usize,
    /// Total number of partitions in the underlying slice.
    total: usize,
}

impl EquispacedPartitioning {
    pub(crate) fn detect(partitions: &[Interval]) -> Option<Self> {
        if partitions.len() < 2 {
            return None;
        }

        let mut finite_offset = 0;
        if !partitions[0].lower.is_finite() {
            finite_offset = 1;
        }

        let mut finite_end = partitions.len();
        if !partitions[finite_end - 1].upper.is_finite() {
            finite_end -= 1;
        }

        // Anything non-finite must be exactly one trailing or leading partition.
        if finite_end <= finite_offset {
            return None;
        }
        for iv in &partitions[finite_offset..finite_end] {
            if !iv.lower.is_finite() || !iv.upper.is_finite() {
                return None;
            }
        }

        let first = &partitions[finite_offset];
        let step = first.upper - first.lower;
        if !(step.is_finite() && step > 0.0) {
            return None;
        }
        let base = first.lower;
        let lower_closed = first.lower_closed;
        let upper_closed = first.upper_closed;

        let step_tol = step.abs() * 1e-9 + 1e-12;
        let mut expected_lower = base;
        let mut count = 0;
        for iv in &partitions[finite_offset..finite_end] {
            if iv.lower_closed != lower_closed || iv.upper_closed != upper_closed {
                return None;
            }
            if (iv.lower - expected_lower).abs() > step_tol {
                return None;
            }
            if (iv.upper - iv.lower - step).abs() > step_tol {
                return None;
            }
            count += 1;
            expected_lower = iv.lower + step;
        }
        if count < 2 {
            return None;
        }

        // Unbounded-lower partition (if present) must abut the finite region's
        // base; same for an unbounded-upper partition.
        if finite_offset == 1 {
            let head = &partitions[0];
            if head.lower != f64::NEG_INFINITY || (head.upper - base).abs() > step_tol {
                return None;
            }
        }
        if finite_end < partitions.len() {
            let tail = &partitions[finite_end];
            let last_upper = base + step * count as f64;
            if tail.upper != f64::INFINITY || (tail.lower - last_upper).abs() > step_tol {
                return None;
            }
        }

        Some(Self {
            base,
            step,
            finite_count: count,
            finite_offset,
            total: partitions.len(),
        })
    }

    /// O(1) partition lookup. Returns `None` when the value falls outside the
    /// covered range (which can only happen when there is no unbounded tail
    /// on the relevant side, e.g., a constant-pinned numeric var that drifted
    /// out of range — that is a real error, the same one `partition_for_value`
    /// reports by returning `None`).
    ///
    /// Currently unused: the cast-based body lookup does not respect
    /// per-interval closed/open boundary flags, so values that land exactly
    /// on a partition boundary can disagree with the tolerant
    /// `partition_for_value` and produce a different abstract hash than
    /// CEGAR's `compute_initial_state_hash_determined`. See the note on
    /// `NumericPartitions::equispaced` for context.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn lookup(&self, value: f64) -> Option<usize> {
        if !value.is_finite() {
            return None;
        }

        // Lower unbounded tail.
        if value < self.base {
            return if self.finite_offset == 1 {
                Some(0)
            } else {
                // Value is below the first finite partition with no head tail.
                // Tolerate values that round to `base` (matches the legacy
                // tolerant binary search).
                let tol = float_tolerance::tolerance(value, self.base);
                ((self.base - value) <= tol).then_some(0)
            };
        }

        let last_upper = self.base + self.step * self.finite_count as f64;
        let upper_tail_present = self.finite_offset + self.finite_count < self.total;
        if value >= last_upper {
            let tol = float_tolerance::tolerance(value, last_upper);
            if !upper_tail_present {
                return ((value - last_upper) <= tol).then_some(self.total - 1);
            }
            // If value is *exactly* at the boundary, the lower-closed
            // convention puts it in the finite partition; otherwise the tail.
            if (value - last_upper).abs() <= tol {
                return Some(self.finite_offset + self.finite_count - 1);
            }
            return Some(self.finite_offset + self.finite_count);
        }

        let raw = (value - self.base) / self.step;
        let mut idx = raw as usize;
        // Defensive clamp against tiny rounding edge cases at the right edge.
        if idx >= self.finite_count {
            idx = self.finite_count - 1;
        }
        Some(self.finite_offset + idx)
    }
}

#[allow(unused)]
pub(crate) fn partitions_for_interval(partitions: &[Interval], value: &Interval) -> Vec<usize> {
    partitions
        .iter()
        .enumerate()
        .filter_map(|(i, iv)| if iv.intersects(value) { Some(i) } else { None })
        .collect()
}

pub(crate) fn make_prop_state_packer(task: &dyn AbstractNumericTask) -> IntDoublePacker {
    let mut domain_sizes: Vec<u64> = Vec::with_capacity(task.variables().len());
    for var in task.variables().iter() {
        domain_sizes.push(var.domain_size() as u64);
    }
    IntDoublePacker::new(&domain_sizes)
}

pub(crate) fn set_initial_prop_values(
    task: &dyn AbstractNumericTask,
    packer: &IntDoublePacker,
    buffer: &mut [u64],
) {
    let init = task.get_initial_propositional_state_values();
    for (var_id, &val) in init.iter().enumerate() {
        packer.set(buffer, var_id, val as u64);
    }
}

pub(crate) fn get_initial_state(
    task: &dyn AbstractNumericTask,
    state_packer: &IntDoublePacker,
    axiom_evaluator: &AxiomEvaluator,
) -> Result<(Vec<u64>, Vec<f64>)> {
    let mut buffer = vec![0u64; state_packer.num_bins()];
    set_initial_prop_values(task, state_packer, &mut buffer);
    let mut numeric_state: Vec<f64> = task.get_initial_numeric_state_values().to_vec();

    axiom_evaluator
        .evaluate_arithmetic_axioms(&mut numeric_state)
        .map_err(|e| {
            anyhow::anyhow!("failed to evaluate arithmetic axioms for initial state: {e:?}")
        })?;
    axiom_evaluator
        .evaluate(&mut buffer, &mut numeric_state)
        .map_err(|e| anyhow::anyhow!("failed to evaluate axioms for initial state: {e:?}"))?;

    Ok((buffer, numeric_state))
}

pub(crate) fn fact_is_hold(fact: &ExplicitFact, packer: &IntDoublePacker, buffer: &[u64]) -> bool {
    let current = packer.get(buffer, fact.var()) as usize;
    current == fact.value()
}

pub(crate) fn debug_print_wildcard_plan(
    task: &dyn AbstractNumericTask,
    plan: &WildcardPlanResult,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    partitions: &NumericPartitions,
) {
    let steps = plan.wildcard_plan.len();
    debug!("[Abstract Plan] steps={steps}");

    let max_steps = 200usize;
    let shown_steps = steps.min(max_steps);
    if steps > max_steps {
        debug!("[Abstract Plan] (truncated to first {shown_steps} steps)");
    }

    if let Some(prop0) = plan.abstract_prop_states.first() {
        debug!(
            "  s0 props: {}",
            fmt_nontrivial_props(prop0, domain_sizes, 100)
        );
    }
    if let Some(num0) = plan.abstract_numeric_states.first() {
        debug!(
            "  s0 nums:  {}",
            fmt_nontrivial_nums(num0, numeric_domain_sizes, partitions, 100)
        );
    }

    let ops = task.get_operators();
    let mut representative: Vec<String> = Vec::with_capacity(shown_steps);

    for i in 0..shown_steps {
        let choices = &plan.wildcard_plan[i];
        let choice_count = choices.len();
        let rep = choices
            .first()
            .and_then(|&id| ops.get(id).map(|op| op.name().to_string()))
            .unwrap_or_else(|| "<none>".to_string());
        representative.push(rep);

        let mut line = String::new();
        let _ = write!(&mut line, "  step {i}: options={choice_count}");
        let preview = 10usize;
        for &op_id in choices.iter().take(preview) {
            let name = ops.get(op_id).map(|op| op.name()).unwrap_or("<bad-op-id>");
            let _ = write!(&mut line, " [{op_id}:{name}]");
        }
        if choice_count > preview {
            let _ = write!(&mut line, " ...");
        }
        debug!("{line}");

        if i + 1 < plan.abstract_prop_states.len() {
            let prev = &plan.abstract_prop_states[i];
            let cur = &plan.abstract_prop_states[i + 1];
            let delta = fmt_delta_i32(prev, cur, 50);
            if !delta.is_empty() {
                debug!("    props Δ: {delta}");
            }
        }
        if i + 1 < plan.abstract_numeric_states.len() {
            let prev = &plan.abstract_numeric_states[i];
            let cur = &plan.abstract_numeric_states[i + 1];
            let delta = fmt_delta_numeric_partitions(prev, cur, partitions, 50);
            if !delta.is_empty() {
                debug!("    nums  Δ: {delta}");
            }
        }
    }

    debug!("[Plan] {}", representative.join(" -> "));
    debug_print_concrete_trace(task, plan, partitions, shown_steps);
}

fn debug_print_concrete_trace(
    task: &dyn AbstractNumericTask,
    plan: &WildcardPlanResult,
    partitions: &NumericPartitions,
    shown_steps: usize,
) {
    let state_packer = std::sync::Arc::new(make_prop_state_packer(task));
    let axiom_evaluator = AxiomEvaluator::new(std::sync::Arc::new(task), state_packer.clone());

    let mut buffer = vec![0u64; state_packer.num_bins() as usize];
    set_initial_prop_values(task, &state_packer, &mut buffer);
    let mut numeric_state: Vec<f64> = task.get_initial_numeric_state_values().to_vec();

    let _ = axiom_evaluator.evaluate_arithmetic_axioms(&mut numeric_state);
    let _ = axiom_evaluator.evaluate(&mut buffer, &mut numeric_state);

    let (prop_scope, num_scope) = trace_variable_scope(task, plan, shown_steps);
    debug!(
        "[Concrete Trace] scope: props={} nums={}",
        prop_scope.len(),
        num_scope.len()
    );
    debug!(
        "  s0 props: {}",
        fmt_concrete_props(task, &state_packer, &buffer, &prop_scope, 200)
    );
    debug!(
        "  s0 nums:  {}",
        fmt_concrete_nums(&numeric_state, &num_scope, partitions, 200)
    );

    let comparison_index = ComparisonAxiomIndex::from_task(task).ok();
    let max_tries_per_step = 30usize;
    for step in 0..shown_steps {
        if step + 1 >= plan.abstract_numeric_states.len() {
            break;
        }
        let expected_abs_numeric_succ = &plan.abstract_numeric_states[step + 1];
        let choices = plan
            .wildcard_plan
            .get(step)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        let mut chosen: Option<(usize, Vec<u64>, Vec<f64>)> = None;
        let mut tries = 0usize;
        for &op_id in choices.iter() {
            if tries >= max_tries_per_step {
                debug!("  step {step}: ... (tried first {max_tries_per_step} options)");
                break;
            }
            let Some(op) = task.get_operators().get(op_id) else {
                continue;
            };
            tries += 1;

            let applicable = if let Some(idx) = comparison_index.as_ref() {
                // Debug-trace only; we use Forward direction which does not
                // consult `deltas`, so an empty map is fine here.
                let deltas: std::collections::HashMap<usize, Vec<f64>> =
                    std::collections::HashMap::new();
                get_progression_precondition_flaws(
                    task,
                    &deltas,
                    partitions,
                    idx,
                    op,
                    &state_packer,
                    &buffer,
                    &numeric_state,
                    step,
                    SplitDirection::Forward,
                )
                .is_empty()
            } else {
                op.preconditions()
                    .iter()
                    .all(|pre| fact_is_hold(pre, &state_packer, &buffer))
            };
            if !applicable {
                continue;
            }

            let mut cand_buffer = buffer.clone();
            let mut cand_numeric = numeric_state.clone();
            progress(
                op,
                &axiom_evaluator,
                &state_packer,
                &mut cand_buffer,
                &mut cand_numeric,
            )
            .expect("Error applying operator");

            let deviation_flaws = get_progression_numeric_deviation_flaws(
                op,
                &numeric_state,
                &cand_numeric,
                expected_abs_numeric_succ,
                partitions,
                step,
                SplitDirection::Forward,
            );

            if deviation_flaws.is_empty() {
                debug!("  step {step}: choose [{op_id}:{}]", op.name());
                chosen = Some((op_id, cand_buffer, cand_numeric));
                break;
            } else {
                debug!(
                    "  step {step}: try    [{op_id}:{}] (reject: numeric deviation)",
                    op.name()
                );
                debug!(
                    "    s{}' props: {}",
                    step + 1,
                    fmt_concrete_props(task, &state_packer, &cand_buffer, &prop_scope, 80)
                );
                debug!(
                    "    s{}' nums:  {}",
                    step + 1,
                    fmt_concrete_nums(&cand_numeric, &num_scope, partitions, 80)
                );
            }
        }

        let Some((_op_id, next_buffer, next_numeric)) = chosen else {
            debug!("  step {step}: no applicable concrete operator found for wildcard options");
            break;
        };
        buffer = next_buffer;
        numeric_state = next_numeric;

        debug!(
            "  s{} props: {}",
            step + 1,
            fmt_concrete_props(task, &state_packer, &buffer, &prop_scope, 200)
        );
        debug!(
            "  s{} nums:  {}",
            step + 1,
            fmt_concrete_nums(&numeric_state, &num_scope, partitions, 200)
        );
    }
}

fn trace_variable_scope(
    task: &dyn AbstractNumericTask,
    plan: &WildcardPlanResult,
    shown_steps: usize,
) -> (Vec<usize>, Vec<usize>) {
    let ops = task.get_operators();
    let mut prop_vars: BTreeSet<usize> = BTreeSet::new();
    let mut num_vars: BTreeSet<usize> = BTreeSet::new();

    for choices in plan.wildcard_plan.iter().take(shown_steps) {
        for &op_id in choices.iter() {
            let Some(op) = ops.get(op_id) else {
                continue;
            };
            for pre in op.preconditions().iter() {
                prop_vars.insert(pre.var());
            }
            for eff in op.effects().iter() {
                prop_vars.insert(eff.var_id());
                for c in eff.conditions().iter() {
                    prop_vars.insert(c.var());
                }
            }
            for neff in op.assignment_effects().iter() {
                num_vars.insert(neff.var_id());
                num_vars.insert(neff.affected_var_id());
                for c in neff.conditions().iter() {
                    prop_vars.insert(c.var());
                }
            }
        }
    }

    (
        prop_vars.into_iter().collect(),
        num_vars.into_iter().collect(),
    )
}

fn fmt_concrete_props(
    task: &dyn AbstractNumericTask,
    packer: &IntDoublePacker,
    buffer: &[u64],
    var_ids: &[usize],
    max_items: usize,
) -> String {
    let mut out = String::new();
    let mut shown = 0usize;
    for &var_id in var_ids.iter() {
        if shown >= max_items {
            let _ = write!(&mut out, " ...");
            break;
        }
        let dom = task
            .variables()
            .get(var_id)
            .map(|v| v.domain_size())
            .unwrap_or(0);
        if dom <= 1 {
            continue;
        }
        if shown > 0 {
            out.push(' ');
        }
        let val = packer.get(buffer, var_id);
        let _ = write!(&mut out, "v{var_id}={val}");
        shown += 1;
    }
    if out.is_empty() {
        "<empty>".to_string()
    } else {
        out
    }
}

fn fmt_concrete_nums(
    numeric_state: &[f64],
    var_ids: &[usize],
    partitions: &NumericPartitions,
    max_items: usize,
) -> String {
    let mut out = String::new();
    let mut shown = 0usize;
    for &num_id in var_ids.iter() {
        if shown >= max_items {
            let _ = write!(&mut out, " ...");
            break;
        }
        let Some(&v) = numeric_state.get(num_id) else {
            continue;
        };
        if shown > 0 {
            out.push(' ');
        }
        let mut part_s = String::new();
        if let Some(parts) = partitions.partitions(num_id)
            && let Some(pid) = partition_for_value(parts, v)
        {
            let iv_s = partitions
                .partition_interval(num_id, pid)
                .map(fmt_interval)
                .unwrap_or_else(|| "<missing-interval>".to_string());
            part_s = format!(" p{pid}:{iv_s}");
        }
        let _ = write!(&mut out, "n{num_id}={}{}", fmt_f64_compact(v), part_s);
        shown += 1;
    }
    if out.is_empty() {
        "<empty>".to_string()
    } else {
        out
    }
}

fn fmt_delta_i32(prev: &[usize], cur: &[usize], max_items: usize) -> String {
    let mut out = String::new();
    let len = prev.len().min(cur.len());
    let mut shown = 0usize;
    for i in 0..len {
        let a = prev[i];
        let b = cur[i];
        if a == b {
            continue;
        }
        if shown >= max_items {
            let _ = write!(&mut out, " ...");
            break;
        }
        if shown > 0 {
            out.push(' ');
        }
        let _ = write!(&mut out, "{i}:{a}->{b}");
        shown += 1;
    }
    out
}

fn fmt_nontrivial_props(values: &[usize], domain_sizes: &[usize], max_items: usize) -> String {
    let mut out = String::new();
    let mut shown = 0usize;
    let len = values.len().min(domain_sizes.len());
    for var_id in 0..len {
        if domain_sizes[var_id] <= 1 {
            continue;
        }
        if shown >= max_items {
            let _ = write!(&mut out, " ...");
            break;
        }
        if shown > 0 {
            out.push(' ');
        }
        let _ = write!(&mut out, "v{var_id}:{}", values[var_id]);
        shown += 1;
    }
    if out.is_empty() {
        "<no-nontrivial-vars>".to_string()
    } else {
        out
    }
}

fn fmt_nontrivial_nums(
    values: &[usize],
    numeric_domain_sizes: &[usize],
    partitions: &NumericPartitions,
    max_items: usize,
) -> String {
    let mut out = String::new();
    let mut shown = 0usize;
    let len = values.len().min(numeric_domain_sizes.len());
    for num_id in 0..len {
        if numeric_domain_sizes[num_id] <= 1 {
            continue;
        }
        if shown >= max_items {
            let _ = write!(&mut out, " ...");
            break;
        }
        if shown > 0 {
            out.push(' ');
        }
        let part = values[num_id];
        let iv_s = partitions
            .partition_interval(num_id, part)
            .map(fmt_interval)
            .unwrap_or_else(|| "<missing-interval>".to_string());
        let _ = write!(&mut out, "n{num_id}=p{part}:{iv_s}");
        shown += 1;
    }
    if out.is_empty() {
        "<no-nontrivial-vars>".to_string()
    } else {
        out
    }
}

fn fmt_delta_numeric_partitions(
    prev: &[usize],
    cur: &[usize],
    partitions: &NumericPartitions,
    max_items: usize,
) -> String {
    let mut out = String::new();
    let len = prev.len().min(cur.len());
    let mut shown = 0usize;
    for num_id in 0..len {
        let a = prev[num_id];
        let b = cur[num_id];
        if a == b {
            continue;
        }
        if shown >= max_items {
            let _ = write!(&mut out, " ...");
            break;
        }
        if shown > 0 {
            out.push(' ');
        }
        let a_s = partitions
            .partition_interval(num_id, a)
            .map(fmt_interval)
            .unwrap_or_else(|| "<missing-interval>".to_string());
        let b_s = partitions
            .partition_interval(num_id, b)
            .map(fmt_interval)
            .unwrap_or_else(|| "<missing-interval>".to_string());
        let _ = write!(&mut out, "n{num_id}:p{a}:{a_s}->p{b}:{b_s}");
        shown += 1;
    }
    out
}

#[allow(unused)]
pub(crate) fn debug_print_evaluate_state(
    prop_str: &str,
    num_str_vec: &[String],
    abs_prop_str: &[String],
    abs_num_str: &[String],
    dist: f64,
) {
    debug!("[Evaluate State]");
    debug!("  concrete props: {}", prop_str);
    debug!("  concrete nums:  {}", num_str_vec.join(" "));
    debug!("  abstract props: {}", abs_prop_str.join(" "));
    debug!("  abstract nums:  {}", abs_num_str.join(" "));
    debug!("  distance:       {}", dist);
}

pub(crate) fn dump_distances(
    factory: &DomainAbstractionFactory,
    task: &dyn AbstractNumericTask,
    table: &AbstractDistanceTable,
) {
    let num_states = table.distances.len();
    debug!("\n=== TABLE OF CORE VARIABLES FOR ALL {num_states} STATES ===\n");

    let num_prop_vars = factory.domain_sizes().len();
    if table.hash_multipliers.len() < num_prop_vars + table.numeric_domain_sizes.len() {
        debug!(
            "[dump_distances] invalid hash_multipliers len={} (expected >= {})",
            table.hash_multipliers.len(),
            num_prop_vars + table.numeric_domain_sizes.len()
        );
        return;
    }

    let mut is_axiom_var: Vec<bool> = vec![false; num_prop_vars];
    for ax in task.axioms().iter() {
        let v = ax.var_id();
        if v < is_axiom_var.len() {
            is_axiom_var[v] = true;
        }
    }

    let refined_numeric_vars: Vec<usize> = table
        .numeric_domain_sizes
        .iter()
        .enumerate()
        .filter_map(|(n, &parts)| (parts > 1).then_some(n))
        .collect();

    let non_axiom_vars: Vec<usize> = factory
        .domain_sizes()
        .iter()
        .enumerate()
        .filter_map(|(v, &dom)| {
            if dom > 1 && !is_axiom_var.get(v).copied().unwrap_or(false) {
                Some(v)
            } else {
                None
            }
        })
        .collect();

    if !refined_numeric_vars.is_empty() || !non_axiom_vars.is_empty() {
        debug!("=== ABSTRACT DOMAINS ===");
    }

    if !refined_numeric_vars.is_empty() {
        debug!("[NumericPartitions]");
        for &num_var_id in &refined_numeric_vars {
            let name = task
                .numeric_variables()
                .get(num_var_id)
                .map(|v| v.name())
                .unwrap_or("<unknown>");
            let parts = factory.partitions().partitions(num_var_id).unwrap_or(&[]);
            debug!("  n{num_var_id}({name}) parts={}", parts.len());
            for (pid, iv) in parts.iter().enumerate() {
                debug!("    p{pid}: {}", fmt_interval(*iv));
            }
        }
    }

    if std::env::var_os("DA_ENABLE_CORE_TABLE_DUMP").is_none() {
        return;
    }

    if !non_axiom_vars.is_empty() {
        debug!("[PropositionalDomains]");
        for &var_id in &non_axiom_vars {
            let abs_dom = factory.domain_sizes().get(var_id).copied().unwrap_or(0);
            let name = task.get_variable_name(var_id).unwrap_or("<unknown>");
            let mapping = factory.domain_mapping().get(var_id);
            debug!("  var{var_id}({name}) abs_dom={abs_dom}");

            let concrete_size = task.get_variable_domain_size(var_id).unwrap_or(0);
            let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); abs_dom];
            for concrete_val in 0..(concrete_size) {
                let abs_val = mapping
                    .and_then(|m| m.get(concrete_val))
                    .copied()
                    .unwrap_or(concrete_val);
                let Some(slot) = buckets.get_mut(abs_val) else {
                    continue;
                };
                slot.push(concrete_val);
            }

            for (abs_val, concretes) in buckets.iter().enumerate() {
                let mut line = format!("    abs{abs_val}: ");
                line.push('[');
                for (i, cv) in concretes.iter().enumerate() {
                    if i > 0 {
                        line.push_str(", ");
                    }
                    line.push_str(&cv.to_string());
                }
                line.push(']');

                let shown = concretes.len().min(3);
                if shown > 0 {
                    let mut names: Vec<&str> = Vec::new();
                    for cv in concretes.iter().take(shown) {
                        let fact = ExplicitFact::new(var_id, *cv);
                        let n = task.get_fact_name(&fact);
                        if !n.is_empty() {
                            names.push(n);
                        }
                    }
                    if !names.is_empty() {
                        line.push_str("  names=");
                        for (i, n) in names.iter().enumerate() {
                            if i > 0 {
                                line.push_str(" | ");
                            }
                            line.push_str(n);
                        }
                        if concretes.len() > shown {
                            line.push_str(" | ...");
                        }
                    }
                }
                debug!("{line}");
            }
        }
    }

    let mut num_headers: Vec<String> = Vec::with_capacity(refined_numeric_vars.len());
    let mut num_widths: Vec<usize> = Vec::with_capacity(refined_numeric_vars.len());
    let mut num_partition_texts: Vec<Vec<String>> = Vec::with_capacity(refined_numeric_vars.len());
    for &num_var_id in &refined_numeric_vars {
        let name = task
            .numeric_variables()
            .get(num_var_id)
            .map(|v| v.name())
            .unwrap_or("<unknown>");
        let header = format!("num{num_var_id}({name})");

        let parts = factory.partitions().partitions(num_var_id).unwrap_or(&[]);
        let mut texts: Vec<String> = Vec::with_capacity(parts.len());
        let mut max_part_len: usize = 0;
        for (pid, iv) in parts.iter().enumerate() {
            let s = format!("p{pid}:{}", fmt_interval(*iv));
            max_part_len = max_part_len.max(s.len());
            texts.push(s);
        }

        let width = header.len().max(6).max(max_part_len);
        num_headers.push(header);
        num_widths.push(width);
        num_partition_texts.push(texts);
    }

    let mut prop_headers: Vec<String> = Vec::with_capacity(non_axiom_vars.len());
    let mut prop_widths: Vec<usize> = Vec::with_capacity(non_axiom_vars.len());
    for &var_id in &non_axiom_vars {
        let name = task.get_variable_name(var_id).unwrap_or("<unknown>");
        let header = format!("var{var_id}({name})");
        let width = header.len().max(6);
        prop_headers.push(header);
        prop_widths.push(width);
    }

    let mut header_line = String::new();
    header_line.push_str("\nState | Flags | Distance | ");
    for (i, h) in num_headers.iter().enumerate() {
        header_line.push_str(&format!("{h:>width$} | ", width = num_widths[i]));
    }
    for (i, h) in prop_headers.iter().enumerate() {
        header_line.push_str(&format!("{h:>width$} | ", width = prop_widths[i]));
    }
    debug!("{header_line}");

    let mut sep = String::new();
    sep.push_str("------|-------|----------|");
    for &w in &num_widths {
        sep.push_str(&"-".repeat(w + 2));
        sep.push('|');
    }
    for &w in &prop_widths {
        sep.push_str(&"-".repeat(w + 2));
        sep.push('|');
    }
    debug!("{sep}");

    for state_hash in 0..(num_states) {
        let dist = table
            .distances
            .get(state_hash)
            .copied()
            .unwrap_or(f64::INFINITY);
        let is_init = state_hash == table.initial_state_hash;
        let is_goal = factory.is_goal_state(
            state_hash,
            &table.goal_facts,
            &table.numeric_domain_sizes,
            &table.hash_multipliers,
        );

        if !(dist.is_finite() || is_init || is_goal) {
            continue;
        }

        let flags = match (is_init, is_goal) {
            (true, true) => "IG",
            (true, false) => "I",
            (false, true) => "G",
            (false, false) => "",
        };

        let dist_cell = if dist.is_finite() {
            format!("{dist:>8.3}")
        } else {
            format!("{:>8}", "INF")
        };

        let mut line = String::new();
        line.push_str(&format!("{state_hash:>5} | {flags:>5} | {dist_cell} | "));

        for (i, &num_var_id) in refined_numeric_vars.iter().enumerate() {
            let abs_var_id = num_prop_vars + num_var_id;
            let mult = table.hash_multipliers[abs_var_id] as i64;
            let dom = table.numeric_domain_sizes[num_var_id] as i64;
            let part = ((state_hash as i64) / mult) % dom;
            let part_usize = usize::try_from(part).unwrap_or(0);
            let val = num_partition_texts
                .get(i)
                .and_then(|v| v.get(part_usize))
                .map(|s| s.as_str())
                .unwrap_or("<invalid>");
            line.push_str(&format!("{val:>width$} | ", width = num_widths[i]));
        }

        for (i, &var_id) in non_axiom_vars.iter().enumerate() {
            let mult = table.hash_multipliers[var_id] as i64;
            let dom = factory.domain_sizes()[var_id] as i64;
            let value = ((state_hash as i64) / mult) % dom;
            line.push_str(&format!(
                "{val:>width$} | ",
                val = value,
                width = prop_widths[i]
            ));
        }

        debug!("{line}");
    }

    debug!("");
}
