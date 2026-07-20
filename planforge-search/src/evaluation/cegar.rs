use std::fmt;
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::numeric_task::{AssignmentOperation, Operator};
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::int_packer::IntDoublePacker;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FlawKind {
    Progression,
    Regression,
    ExecuteEntirePlan,
    SequenceProgression,
    SequenceRegression,
    SequenceBidirectional,
    TargetCentered,
}

impl fmt::Display for FlawKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Progression => write!(f, "progression"),
            Self::Regression => write!(f, "regression"),
            Self::ExecuteEntirePlan => write!(f, "execute_entire_plan"),
            Self::SequenceProgression => write!(f, "sequence_progression"),
            Self::SequenceRegression => write!(f, "sequence_regression"),
            Self::SequenceBidirectional => write!(f, "sequence_bidirectional"),
            Self::TargetCentered => write!(f, "target_centered"),
        }
    }
}

impl crate::config::sealed::Sealed for FlawKind {}

impl crate::config::FromOptionValue for FlawKind {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "progression" => Ok(Self::Progression),
            "regression" => Ok(Self::Regression),
            "execute_entire_plan" => Ok(Self::ExecuteEntirePlan),
            "sequence_progression" => Ok(Self::SequenceProgression),
            "sequence_regression" => Ok(Self::SequenceRegression),
            "sequence_bidirectional" => Ok(Self::SequenceBidirectional),
            "target_centered" => Ok(Self::TargetCentered),
            other => Err(format!("invalid FlawKind `{other}`")),
        }
    }
}

pub fn progress_concrete_state(
    op: &Operator,
    axiom_evaluator: &AxiomEvaluator,
    packer: &IntDoublePacker,
    prop_state: &mut [u64],
    numeric_state: &mut [f64],
) -> Result<()> {
    for effect in op.effects() {
        if effect
            .conditions()
            .iter()
            .all(|condition| packer.get(prop_state, condition.var()) == condition.value() as u64)
        {
            packer.set(prop_state, effect.var_id(), effect.value() as u64);
        }
    }

    for effect in op.assignment_effects() {
        if effect.is_conditional()
            && !effect.conditions().iter().all(|condition| {
                packer.get(prop_state, condition.var()) == condition.value() as u64
            })
        {
            continue;
        }
        let source_var = effect.var_id();
        let affected_var = effect.affected_var_id();
        ensure!(
            source_var < numeric_state.len() && affected_var < numeric_state.len(),
            "operator {} numeric effect references vars ({source_var}, {affected_var}) outside {} numeric variables",
            op.name(),
            numeric_state.len()
        );
        numeric_state[affected_var] = float_tolerance::canonicalize(AssignmentOperation::apply(
            numeric_state[affected_var],
            effect.operation(),
            numeric_state[source_var],
        ));
    }

    axiom_evaluator
        .evaluate_arithmetic_axioms(numeric_state)
        .map_err(|error| anyhow::anyhow!("failed to evaluate arithmetic axioms: {error:?}"))?;
    axiom_evaluator
        .evaluate(prop_state, numeric_state)
        .map_err(|error| anyhow::anyhow!("failed to evaluate axioms: {error:?}"))?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CegarStopReason {
    ConcretePlan,
    SizeLimit,
    TimeLimit,
    MemoryLimit,
    IterationLimit,
    NoRefinableFlaw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CegarIterationResult {
    Continue,
    Stop(CegarStopReason),
}

#[derive(Debug, Clone, Copy)]
pub struct CegarRunResult {
    pub next_iteration: usize,
    pub stop_reason: CegarStopReason,
}

#[derive(Debug, Clone, Copy)]
pub struct CegarDriver {
    max_iterations: usize,
    max_time: Option<Duration>,
}

impl CegarDriver {
    pub fn new(max_iterations: usize, max_time: Option<Duration>) -> Self {
        assert!(max_iterations > 0, "CEGAR max_iterations must be > 0");
        Self {
            max_iterations,
            max_time,
        }
    }

    #[inline]
    pub fn run_from(
        self,
        start: Instant,
        run_iteration: impl FnMut(usize, Option<Instant>) -> Result<CegarIterationResult>,
    ) -> Result<CegarRunResult> {
        self.run_from_with_poll_phase::<false>(start, run_iteration)
    }

    /// Runs a backend whose pre-iteration resource poll historically used the
    /// number of completed refinements, starting at zero.
    #[inline]
    pub fn run_from_zero_based(
        self,
        start: Instant,
        run_iteration: impl FnMut(usize, Option<Instant>) -> Result<CegarIterationResult>,
    ) -> Result<CegarRunResult> {
        self.run_from_with_poll_phase::<true>(start, run_iteration)
    }

    #[inline]
    fn run_from_with_poll_phase<const POLL_BEFORE_FIRST: bool>(
        self,
        start: Instant,
        mut run_iteration: impl FnMut(usize, Option<Instant>) -> Result<CegarIterationResult>,
    ) -> Result<CegarRunResult> {
        let deadline = self.max_time.map(|max_time| start + max_time);
        let mut iteration = 1usize;

        while iteration <= self.max_iterations {
            let memory_poll_counter = if POLL_BEFORE_FIRST {
                iteration - 1
            } else {
                iteration
            };
            if memory_poll_counter.is_multiple_of(64)
                && !crate::resource_limits::poll_and_release_if_exceeded()
            {
                return Ok(CegarRunResult {
                    next_iteration: iteration,
                    stop_reason: CegarStopReason::MemoryLimit,
                });
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Ok(CegarRunResult {
                    next_iteration: iteration,
                    stop_reason: CegarStopReason::TimeLimit,
                });
            }

            match run_iteration(iteration, deadline)? {
                CegarIterationResult::Continue => iteration += 1,
                CegarIterationResult::Stop(stop_reason) => {
                    return Ok(CegarRunResult {
                        next_iteration: iteration,
                        stop_reason,
                    });
                }
            }
        }

        Ok(CegarRunResult {
            next_iteration: iteration,
            stop_reason: CegarStopReason::IterationLimit,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_counts_completed_iterations() {
        let mut seen = Vec::new();
        let result = CegarDriver::new(3, None)
            .run_from(Instant::now(), |iteration, deadline| {
                assert!(deadline.is_none());
                seen.push(iteration);
                Ok(CegarIterationResult::Continue)
            })
            .unwrap();

        assert_eq!(seen, vec![1, 2, 3]);
        assert_eq!(result.next_iteration, 4);
        assert_eq!(result.stop_reason, CegarStopReason::IterationLimit);
    }

    #[test]
    fn driver_propagates_backend_stop_reason() {
        let result = CegarDriver::new(10, None)
            .run_from(Instant::now(), |iteration, _| {
                Ok(if iteration == 2 {
                    CegarIterationResult::Stop(CegarStopReason::ConcretePlan)
                } else {
                    CegarIterationResult::Continue
                })
            })
            .unwrap();

        assert_eq!(result.next_iteration, 2);
        assert_eq!(result.stop_reason, CegarStopReason::ConcretePlan);
    }
}
