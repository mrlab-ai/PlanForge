pub(crate) mod fact_splits;
pub(crate) mod progression;
pub(crate) mod refine;
pub(crate) mod regression;
pub(crate) mod selection;
pub(crate) mod semantics;
pub(crate) mod split;
pub(crate) mod view;

use std::fmt;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Generates the option-value parsing and display of a CEGAR option enum
/// from its `(variant, name)` pairs.
macro_rules! option_enum {
    ($option:literal, $type:ident { $($variant:ident => $name:literal),+ $(,)? }) => {
        impl fmt::Display for $type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(match self {
                    $(Self::$variant => $name),+
                })
            }
        }

        impl crate::config::FromOptionValue for $type {
            fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
                match crate::config::atom(value)? {
                    $($name => Ok(Self::$variant),)+
                    other => Err(format!(
                        "invalid {} `{other}`; expected one of {}",
                        $option,
                        [$($name),+].join(", ")
                    )),
                }
            }
        }
    };
}

/// How CEGAR finds the flaws of an abstract plan.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FlawKind {
    /// By executing the plan from the refinement root.
    Progression,
    /// By regressing the goal through the plan (Pozo et al. 2024).
    Regression,
}

option_enum!("flaw_kind", FlawKind {
    Progression => "progression",
    Regression => "regression",
});

/// Which flaws of an abstract plan CEGAR collects.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FlawScope {
    /// The first flaw.
    First,
    /// Every flaw of the whole plan; split selection chooses among them.
    /// Progression continues past a flaw by applying the effects anyway,
    /// regression by ignoring the contradicted condition and regressing the
    /// effects anyway.
    WholePlan,
}

option_enum!("flaw_scope", FlawScope {
    First => "first",
    WholePlan => "whole_plan",
});

/// Where a split separates a flawed value from the values the abstract plan
/// wants: next to the concrete value, or at the bound of the wanted values.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SplitKind {
    #[default]
    Concrete,
    Wanted,
}

impl crate::config::FromOptionValue for Option<SplitKind> {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        SplitKind::from_option_value(value).map(Some)
    }
}

option_enum!("split_kind", SplitKind {
    Concrete => "concrete",
    Wanted => "wanted",
});

/// How CEGAR chooses among the candidate splits of a flaw.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SplitSelection {
    /// The split that grows the abstraction least (Kreft et al. 2023); ties
    /// by a seeded key.
    MinGrowth,
    /// A uniformly random split.
    Random,
}

option_enum!("split_selection", SplitSelection {
    MinGrowth => "min_growth",
    Random => "random",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CegarStopReason {
    ConcretePlan,
    SizeLimit,
    TimeLimit,
    MemoryLimit,
    IterationLimit,
    NoRefinableFlaw,
    /// The refinement root is an abstract dead end. Abstractions are
    /// admissible, so it is a concrete dead end as well.
    DeadEnd,
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
