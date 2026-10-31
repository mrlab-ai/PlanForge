//! The random walks of Haslum et al. (2007) that iPDB and the offline order
//! diversification of saturated cost partitioning sample states with.

use planforge_sas::utils::float_tolerance;
use rand::Rng;
use rand::rngs::SmallRng;

use crate::evaluation::evaluator::EvaluationError;

/// A random walk length from the estimate `initial_h` of the initial state.
pub(crate) fn random_walk_length(
    initial_h: f64,
    average_operator_cost: f64,
    rng: &mut SmallRng,
) -> Result<usize, EvaluationError> {
    if !initial_h.is_finite() || initial_h < 0.0 {
        return Err(EvaluationError::ComputationFailed(format!(
            "random-walk sampling requires a finite non-negative initial h, got {initial_h}"
        )));
    }
    if !average_operator_cost.is_finite() || average_operator_cost < 0.0 {
        return Err(EvaluationError::ComputationFailed(format!(
            "random-walk sampling requires a finite non-negative average operator cost, got {average_operator_cost}"
        )));
    }
    let trials_f64 = if initial_h <= float_tolerance::SEARCH_EPSILON
        || average_operator_cost <= float_tolerance::SEARCH_EPSILON
    {
        UNSCALED_WALK_TRIALS
    } else {
        TRIALS_PER_ESTIMATED_STEP * (initial_h / average_operator_cost).round()
    };
    if trials_f64 > usize::MAX as f64 {
        return Err(EvaluationError::ComputationFailed(format!(
            "random-walk trial count does not fit usize: {trials_f64}"
        )));
    }
    let trials = trials_f64 as usize;
    Ok((0..trials)
        .filter(|_| rng.gen_bool(STEP_PROBABILITY))
        .count())
}

/// The walk length is binomial, `trials` trials with success probability
/// [`STEP_PROBABILITY`], as in the random-walk sampling of Haslum et al.
/// (2007) that Seipp et al. (2020) diversify over.
const STEP_PROBABILITY: f64 = 0.5;

/// Trials per estimated step to the goal, `h / average operator cost`: the
/// expected walk length is twice the estimated goal distance.
const TRIALS_PER_ESTIMATED_STEP: f64 = 4.0;

/// Trials when the initial state's estimate or the average operator cost is
/// zero and gives no estimated goal distance.
const UNSCALED_WALK_TRIALS: f64 = 10.0;
