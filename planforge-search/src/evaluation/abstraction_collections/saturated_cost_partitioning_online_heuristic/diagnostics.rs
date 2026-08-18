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
        "scp_online: label {step} abstraction {abstraction_id}: current_h={current_h}, positive_saturated_labels={positive_count}, total_positive_saturated={total_positive:.6}"
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
        "scp_online: abstract-operator {step} abstraction {abstraction_id}: current_h={current_h}, positive_saturated_abstract_ops={positive_count}, total_positive_saturated={total_positive:.6}"
    );
}

pub(super) fn log_abstract_operator_region_summary(
    abstraction_id: usize,
    operator_regions: &[AbstractOperatorRegions],
) {
    if !enabled!(Level::DEBUG) {
        return;
    }
    let stats = abstract_operator_region_stats(operator_regions);
    debug!(
        "scp_online: abstract-operator regions abstraction {abstraction_id}: labels={}, bounded_labels={}, bounded_numeric_dimensions={}",
        stats.total_labels, stats.bounded_labels, stats.bounded_numeric_dimensions,
    );
}

pub(super) fn log_abstraction_candidate_report(
    mode: &str,
    state: &ScpOnlineState,
    components: &[AbstractionComponent<'_>],
    order: &[usize],
    abstract_state_ids: &[Option<usize>],
    scoring_function: ScoringFunction,
) {
    let inactive = order
        .iter()
        .filter(|&&abstraction_id| {
            state
                .h_values_by_abstraction
                .get(abstraction_id)
                .map(|distances| {
                    current_h_for_distances(abstraction_id, distances, abstract_state_ids)
                })
                .unwrap_or(0.0)
                <= 1e-9
        })
        .count();
    info!(
        "scp_online: {mode} abstraction candidate report, candidates={}, inactive_current_state={inactive}, showing_top={}",
        order.len(),
        order.len().min(25),
    );

    for (rank, &abstraction_id) in order.iter().take(25).enumerate() {
        let h = state
            .h_values_by_abstraction
            .get(abstraction_id)
            .map(|distances| current_h_for_distances(abstraction_id, distances, abstract_state_ids))
            .unwrap_or(0.0);
        let stolen = state
            .stolen_costs_by_abstraction
            .get(abstraction_id)
            .copied()
            .unwrap_or(0.0);
        let Some(component) = components.get(abstraction_id) else {
            info!(
                "scp_online: candidate rank={rank}, id={abstraction_id}, h={h}, stolen={stolen}, missing_component=true"
            );
            continue;
        };
        let Some(abstraction) = component.as_domain() else {
            info!(
                "scp_online: candidate rank={rank}, id={abstraction_id}, h={h}, stolen={stolen}, kind={}",
                component.kind()
            );
            continue;
        };
        let score = compute_score(h, stolen, scoring_function);
        let stats = abstract_operator_region_stats(&abstraction.abstract_operator_regions);
        let metadata = &abstraction.metadata;
        let seeds = truncate_for_log(&metadata.initial_seed_splits.join("|"), 220);
        info!(
            "scp_online: candidate rank={rank}, id={abstraction_id}, score={score:.6}, h={h}, stolen={stolen:.6}, states={}, abstract_ops={}, operator_region_labels={}, bounded_operator_region_labels={}, bounded_numeric_dimensions={}, iteration={:?}, flaw_kind={:?}, full_goal_task={:?}, seeds={seeds}",
            abstraction_state_count(abstraction),
            abstraction.abstract_operators.len(),
            stats.total_labels,
            stats.bounded_labels,
            stats.bounded_numeric_dimensions,
            metadata.collection_iteration,
            metadata.flaw_kind,
            metadata.full_goal_task,
        );
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct AbstractOperatorRegionStats {
    total_labels: usize,
    bounded_labels: usize,
    bounded_numeric_dimensions: usize,
}

impl AbstractOperatorRegionStats {
    pub(super) fn total_labels(&self) -> usize {
        self.total_labels
    }

    pub(super) fn bounded_labels(&self) -> usize {
        self.bounded_labels
    }
}

pub(super) fn abstract_operator_region_stats(
    operator_regions: &[AbstractOperatorRegions],
) -> AbstractOperatorRegionStats {
    let mut stats = AbstractOperatorRegionStats::default();
    for label in operator_regions.iter().flat_map(|fp| fp.labels.iter()) {
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

#[derive(Debug, Default)]
struct LabelOperatorRegionCounts {
    operator_regions: usize,
    bounded_operator_regions: usize,
    bounded_numeric_dimensions: usize,
}

pub(super) fn log_positive_label_operator_region_diagnostics(
    abstraction_id: usize,
    task: &dyn AbstractNumericTask,
    operator_regions: &[AbstractOperatorRegions],
    label_saturated_costs: &[f64],
) {
    if !enabled!(Level::INFO) {
        return;
    }
    let mut counts_by_label: HashMap<usize, LabelOperatorRegionCounts> = HashMap::new();
    for label in operator_regions
        .iter()
        .flat_map(|operator_region| operator_region.labels.iter())
    {
        let counts = counts_by_label.entry(label.concrete_op_id).or_default();
        counts.operator_regions += 1;
        let bounded_dimensions = label
            .source
            .numeric
            .iter()
            .filter(|interval| interval.lower.is_finite() || interval.upper.is_finite())
            .count();
        if bounded_dimensions > 0 {
            counts.bounded_operator_regions += 1;
        }
        counts.bounded_numeric_dimensions += bounded_dimensions;
    }

    let mut positive_labels = label_saturated_costs
        .iter()
        .enumerate()
        .filter_map(|(concrete_op_id, &cost)| {
            if cost.is_finite() && cost > 1e-9 {
                Some((concrete_op_id, cost))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    positive_labels.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    for (rank, (concrete_op_id, saturated_cost)) in positive_labels.into_iter().take(12).enumerate()
    {
        let counts = counts_by_label.get(&concrete_op_id);
        let (operator_region_count, bounded_operator_regions, bounded_numeric_dimensions) = counts
            .map(|counts| {
                (
                    counts.operator_regions,
                    counts.bounded_operator_regions,
                    counts.bounded_numeric_dimensions,
                )
            })
            .unwrap_or((0, 0, 0));
        let op = task.get_operators().get(concrete_op_id);
        let op_name = op.map(|op| op.name()).unwrap_or("<missing operator>");
        let numeric_effects = op.map(|op| op.assignment_effects().len()).unwrap_or(0);
        info!(
            "scp_online: abstract-operator label diagnostic detail abstraction {abstraction_id}: rank={rank}, label={concrete_op_id}, saturated={saturated_cost:.6}, numeric_effects={numeric_effects}, operator_regions={operator_region_count}, bounded_operator_regions={bounded_operator_regions}, bounded_numeric_dimensions={bounded_numeric_dimensions}, op={op_name}"
        );
    }
}

pub(super) fn abstraction_metadata_summary(abstraction: &DomainAbstraction) -> String {
    let metadata = &abstraction.metadata;
    format!(
        "iteration={:?},strategy={:?},flaw_kind={:?},full_goal_task={:?},seeds={}",
        metadata.collection_iteration,
        metadata.collection_strategy,
        metadata.flaw_kind,
        metadata.full_goal_task,
        metadata.initial_seed_splits.join("|"),
    )
}

fn truncate_for_log(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut truncated = value.chars().take(max_chars).collect::<String>();
    truncated.push_str("...");
    truncated
}

pub(super) fn log_transition_residual_summary(remaining_costs: &TransitionResidualCosts) {
    if !enabled!(Level::DEBUG) {
        return;
    }
    debug!(
        "scp_online: abstract-operator residuals now store {} region reductions",
        remaining_costs.num_reductions()
    );
}
