//! The costs a cost partitioning has left to distribute: one base cost per
//! operator, less what tables used of it uniformly or on source sets.

use std::cell::Cell;
use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use planforge_sas::numeric_task::OperatorIndex;
use planforge_sas::utils::float_tolerance;

use super::usage::{SourceSetUsage, TableUsageEnvelope};
use super::{GEOMETRIC_POLL_STRIDE, SCP_TABLE_CONSTRUCTION, SourceSetCostAllocation};
use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;
use crate::evaluation::abstractions::common::transition_system::LabelSourceSet;
use crate::resource_limits::Deadline;

#[derive(Debug)]
pub struct RemainingCosts {
    pub(super) operator_residuals: Vec<OperatorRemainingCost>,
}

#[derive(Debug)]
pub(super) struct OperatorRemainingCost {
    base_cost: f64,
    /// Regions where prior tables consumed the complete operator cost. These
    /// may overlap because pointwise max and sum are both equal to the base
    /// cost there; keeping the cover avoids an unnecessary geometric union.
    pub(super) full_usage: SourceSetUsage,
    /// Exact disjoint overlay for genuinely fractional regional allocations.
    pub(super) fractional_usage: SourceSetUsage,
    uniform_cost_cache: Cell<Option<f64>>,
}

impl OperatorRemainingCost {
    /// The most of the cost used anywhere on `region`.
    fn used_on(&self, region: &CartesianSet) -> f64 {
        if self.full_usage.overlaps(region) {
            self.base_cost
        } else {
            self.fractional_usage.max_over(region)
        }
    }

    fn invalidate_cache(&self) {
        self.uniform_cost_cache.set(None);
    }
}

impl RemainingCosts {
    pub fn from_operator_costs(costs: &[f64]) -> Self {
        let operator_residuals = costs
            .iter()
            .map(|&base_cost| OperatorRemainingCost {
                base_cost,
                full_usage: SourceSetUsage::default(),
                fractional_usage: SourceSetUsage::default(),
                uniform_cost_cache: Cell::new(None),
            })
            .collect();
        Self { operator_residuals }
    }

    pub fn num_reductions(&self) -> usize {
        self.operator_residuals
            .iter()
            .map(|residual| residual.full_usage.cells.len() + residual.fractional_usage.cells.len())
            .sum()
    }

    pub fn base_cost(&self, concrete_op_id: OperatorIndex) -> f64 {
        self.residual(concrete_op_id).base_cost
    }

    /// Panics on an unknown operator id: costs are indexed by concrete operator
    /// and every caller derives the id from the task the residuals were built
    /// from, so a miss is a broken invariant rather than missing input.
    #[inline]
    fn residual(&self, concrete_op_id: OperatorIndex) -> &OperatorRemainingCost {
        self.operator_residuals
            .get(concrete_op_id.index())
            .unwrap_or_else(|| {
                panic!(
                    "missing concrete operator {concrete_op_id}: operator residual count is {}",
                    self.operator_residuals.len()
                )
            })
    }

    /// What remains of every operator's cost everywhere, for tables that
    /// take one cost per operator.
    pub fn operator_costs_for_label_cp(&self) -> Vec<f64> {
        self.operator_residuals
            .iter()
            .map(|residual| {
                if !residual.base_cost.is_finite() {
                    return f64::INFINITY;
                }
                if let Some(cost) = residual.uniform_cost_cache.get() {
                    return cost;
                }
                let full_regional_reduction = if !residual.full_usage.is_empty() {
                    residual.base_cost
                } else {
                    0.0
                };
                let reduction = full_regional_reduction.max(
                    residual
                        .fractional_usage
                        .cells
                        .iter()
                        .map(|cell| cell.amount)
                        .fold(0.0, f64::max),
                );
                let cost = cost_left(residual.base_cost, reduction);
                residual.uniform_cost_cache.set(Some(cost));
                cost
            })
            .collect()
    }

    /// What remains of the cost of `source_set`'s operator everywhere on
    /// its source set.
    pub fn cost_for_source_set(&self, source_set: &LabelSourceSet) -> f64 {
        let residual = self.residual(source_set.concrete_op_id);
        if !residual.base_cost.is_finite() {
            return f64::INFINITY;
        }
        cost_left(residual.base_cost, residual.used_on(&source_set.source))
    }

    pub fn reduce_by_source_set_allocation_with_deadline(
        &mut self,
        allocation: &SourceSetCostAllocation,
        deadline: Deadline,
    ) -> Result<()> {
        let mut table_envelopes: HashMap<OperatorIndex, TableUsageEnvelope> = HashMap::new();
        for (entry_id, entry) in allocation.entries().iter().enumerate() {
            if entry_id.is_multiple_of(GEOMETRIC_POLL_STRIDE) {
                deadline.ensure(SCP_TABLE_CONSTRUCTION)?;
            }
            ensure!(
                entry.amount.is_finite() && entry.amount >= -float_tolerance::SEARCH_EPSILON,
                "regional allocation entry {entry_id} has invalid amount {}",
                entry.amount
            );
            if entry.amount <= float_tolerance::SEARCH_EPSILON {
                continue;
            }
            let concrete_op_id = entry.source_set.concrete_op_id;
            let residual = self
                .operator_residuals
                .get(concrete_op_id.index())
                .with_context(|| {
                    format!(
                        "regional allocation references missing concrete operator {concrete_op_id}"
                    )
                })?;
            ensure!(
                residual.base_cost.is_finite()
                    && residual.base_cost > float_tolerance::SEARCH_EPSILON,
                "regional allocation requires a positive finite base cost for operator {concrete_op_id}"
            );
            ensure!(
                entry.amount <= residual.base_cost + float_tolerance::SEARCH_EPSILON,
                "regional allocation {} exceeds base cost {} for operator {concrete_op_id}",
                entry.amount,
                residual.base_cost
            );
            table_envelopes
                .entry(concrete_op_id)
                .or_default()
                .maximize(
                    &entry.source_set.source,
                    entry.amount,
                    residual.base_cost,
                    deadline,
                )?;
        }

        for (concrete_op_id, envelope) in table_envelopes {
            deadline.ensure(SCP_TABLE_CONSTRUCTION)?;
            let residual = self
                .operator_residuals
                .get_mut(concrete_op_id.index())
                .expect("validated concrete operator region must exist");
            match envelope {
                TableUsageEnvelope::Full(regions) => {
                    for region in &regions {
                        let already_used = residual.used_on(region);
                        ensure!(
                            already_used <= float_tolerance::SEARCH_EPSILON,
                            "full regional allocation overlaps prior usage for operator {concrete_op_id}: used={already_used}, base={}",
                            residual.base_cost
                        );
                    }
                    residual
                        .full_usage
                        .append_full_cover(regions, residual.base_cost);
                }
                TableUsageEnvelope::Fractional(envelope) => {
                    for cell in envelope.cells {
                        let already_used = residual.used_on(&cell.region);
                        ensure!(
                            already_used + cell.amount
                                <= residual.base_cost + float_tolerance::SEARCH_EPSILON,
                            "regional residual cost underflow for operator {concrete_op_id}: used={already_used}, allocation={}, base={}",
                            cell.amount,
                            residual.base_cost
                        );
                        residual.fractional_usage.add_with_deadline(
                            &cell.region,
                            cell.amount,
                            deadline,
                        )?;
                    }
                }
            }
            residual.invalidate_cache();
        }
        Ok(())
    }

    pub fn reduce_operator_costs_uniform(&mut self, saturated_costs: &[f64]) -> Result<()> {
        ensure!(
            self.operator_residuals.len() == saturated_costs.len(),
            "operator cost vector length mismatch: buckets={}, saturated={}",
            self.operator_residuals.len(),
            saturated_costs.len()
        );
        for (op_id, saturated) in saturated_costs.iter().copied().enumerate() {
            ensure!(
                saturated.is_finite() && saturated >= -float_tolerance::SEARCH_EPSILON,
                "operator {op_id} has the invalid saturated cost {saturated}"
            );
            if saturated <= float_tolerance::SEARCH_EPSILON {
                continue;
            }
            self.operator_residuals[op_id].base_cost =
                subtract_cost(self.operator_residuals[op_id].base_cost, saturated).with_context(
                    || format!("uniform residual reduction underflow for operator {op_id}"),
                )?;
            self.operator_residuals[op_id].invalidate_cache();
        }
        Ok(())
    }
}

/// What is left of `cost` when `used` of it is used, which never exceeds it
/// beyond round-off; the round-off is snapped to zero as [`subtract_cost`]
/// does.
fn cost_left(cost: f64, used: f64) -> f64 {
    let left = cost - used;
    assert!(
        left >= -float_tolerance::SEARCH_EPSILON,
        "{used} of the operator cost {cost} is used, more than it is"
    );
    left.max(0.0)
}

fn subtract_cost(cost: f64, saturated: f64) -> Result<f64> {
    ensure!(cost.is_finite(), "residual cost must be finite, got {cost}");
    ensure!(
        saturated.is_finite(),
        "saturated cost must be finite, got {saturated}"
    );
    let reduced = cost - saturated;
    if reduced < 0.0 && reduced > -float_tolerance::SEARCH_EPSILON {
        Ok(0.0)
    } else {
        ensure!(
            reduced >= 0.0,
            "residual cost underflow: {cost} - {saturated} = {reduced}"
        );
        Ok(reduced)
    }
}
