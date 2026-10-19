//! Debug output of domain-abstraction CEGAR, written at `debug` level.

use planforge_sas::numeric_task::AbstractNumericTask;
use planforge_sas::numeric_task::VariableIndex;
use planforge_sas::utils::interval::Interval;
use tracing::debug;

use super::abstraction::compute_abstraction_size_u128;
use super::factory::DomainAbstractionFactory;
use crate::evaluation::abstractions::common::transition_system::AbstractDistanceTable;

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
        if v.index() < is_axiom_var.len() {
            is_axiom_var[v.index()] = true;
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
            let parts = factory
                .partitions()
                .partitions(VariableIndex::from_usize(num_var_id))
                .unwrap_or(&[]);
            debug!("  n{num_var_id}({name}) parts={}", parts.len());
            for (pid, iv) in parts.iter().enumerate() {
                debug!("    p{pid}: {}", fmt_interval(*iv));
            }
        }
    }
}

pub(crate) fn fmt_interval(iv: Interval) -> String {
    let l = if iv.lower_closed { '[' } else { '(' };
    let r = if iv.upper_closed { ']' } else { ')' };
    let lo = fmt_f64_compact(iv.lower.value());
    let hi = fmt_f64_compact(iv.upper.value());
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
