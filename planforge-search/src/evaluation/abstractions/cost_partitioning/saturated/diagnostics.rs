use super::*;
use tracing::{Level, enabled};

pub(super) fn log_label_table_summary(
    step: &str,
    abstraction_id: usize,
    distances: &[f64],
    saturated_costs: &[f64],
    abstract_state_ids: &[Option<usize>],
) {
    if !enabled!(Level::DEBUG) {
        return;
    }
    let (positive_count, total_positive) = positive_cost_stats(saturated_costs);
    let current_h = current_h_for_distances(abstraction_id, distances, abstract_state_ids);
    debug!(
        "scp: label {step} abstraction {abstraction_id}: current_h={current_h}, positive_saturated_labels={positive_count}, total_positive_saturated={total_positive:.6}"
    );
}

pub(super) fn log_transition_table_summary(
    step: &str,
    abstraction_id: usize,
    distances: &[f64],
    operator_costs: &[f64],
    abstract_state_ids: &[Option<usize>],
) {
    if !enabled!(Level::DEBUG) {
        return;
    }
    let (positive_count, total_positive) = positive_cost_stats(operator_costs);
    let current_h = current_h_for_distances(abstraction_id, distances, abstract_state_ids);
    debug!(
        "scp: interval {step} abstraction {abstraction_id}: current_h={current_h}, positive_saturated_abstract_ops={positive_count}, total_positive_saturated={total_positive:.6}"
    );
}

pub(super) fn log_abstract_source_set_summary(
    abstraction_id: usize,
    source_sets: &[AbstractOperatorSourceSets],
) {
    if !enabled!(Level::DEBUG) {
        return;
    }
    let stats = abstract_source_set_stats(source_sets);
    debug!(
        "scp: source sets of abstraction {abstraction_id}: labels={}, bounded_labels={}, bounded_numeric_dimensions={}",
        stats.total_labels, stats.bounded_labels, stats.bounded_numeric_dimensions,
    );
}

#[derive(Debug, Clone, Copy, Default)]
struct AbstractOperatorRegionStats {
    total_labels: usize,
    bounded_labels: usize,
    bounded_numeric_dimensions: usize,
}

fn abstract_source_set_stats(
    source_sets: &[AbstractOperatorSourceSets],
) -> AbstractOperatorRegionStats {
    let mut stats = AbstractOperatorRegionStats::default();
    for label in source_sets.iter().flat_map(|fp| fp.labels.iter()) {
        stats.total_labels = stats.total_labels.saturating_add(1);
        let bounded_dimensions = label
            .source
            .numeric
            .iter()
            .filter(|interval| interval.lower.is_finite() || interval.upper.is_finite())
            .count();
        if bounded_dimensions > 0 {
            stats.bounded_labels = stats.bounded_labels.saturating_add(1);
        }
        stats.bounded_numeric_dimensions = stats
            .bounded_numeric_dimensions
            .saturating_add(bounded_dimensions);
    }
    stats
}

pub(super) fn log_transition_residual_summary(remaining_costs: &RemainingCosts) {
    if !enabled!(Level::DEBUG) {
        return;
    }
    debug!(
        "scp: remaining costs now store {} source-set reductions",
        remaining_costs.num_reductions()
    );
}
