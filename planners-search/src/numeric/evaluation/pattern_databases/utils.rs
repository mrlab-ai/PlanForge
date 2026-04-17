use log::{debug, info};
use planners_sas::numeric::numeric_task::AbstractNumericTask;

use super::pattern_database::PatternDatabase;
use super::projected_task::{Pattern, ProjectedTask};

pub(crate) fn print_projection_summary(
    base: &dyn AbstractNumericTask,
    pattern: &Pattern,
    projected_task: &ProjectedTask<'_>,
) {
    info!("=== GREEDY NUMERIC PDB ===");
    info!(
        "  propositional vars: base={} pattern={} projected={}",
        base.variables().len(),
        pattern.regular.len(),
        projected_task.variables().len()
    );
    info!(
        "  numeric vars:       base={} pattern={} projected={}",
        base.numeric_variables().len(),
        pattern.numeric.len(),
        projected_task.numeric_variables().len()
    );
    info!(
        "  goals/operators:    goals={} operators={}",
        projected_task.get_num_goals(),
        projected_task.get_operators().len()
    );
}

#[allow(unused)]
pub(crate) fn dump_distance_table(pdb: &PatternDatabase<'_>) {
    let goal_states: Vec<usize> = pdb
        .states
        .iter()
        .enumerate()
        .filter_map(|(state_id, state)| {
            pdb.is_goal_state(pdb.state_propositional_values(state))
                .then_some(state_id)
        })
        .collect();
    let dead_end_count = pdb
        .distances
        .iter()
        .filter(|distance| !distance.is_finite())
        .count();

    debug!("=== GREEDY NUMERIC PDB DISTANCES ===");
    debug!("  initial state:      0");
    debug!("  reachable states:   {}", pdb.states.len());
    debug!(
        "  goal states:        {} {:?}",
        goal_states.len(),
        goal_states
    );
    debug!("  truncated:          {}", pdb.truncated);
    debug!("  frontier states:    {:?}", pdb.frontier_states);
    debug!("  dead ends:          {}", dead_end_count);
    debug!(
        "  min operator cost:  {}",
        fmt_distance(pdb.min_operator_cost)
    );

    let pattern_regular_projected_ids = pdb.task.pattern_regular_projected_ids();
    let prop_headers: Vec<String> = pattern_regular_projected_ids
        .iter()
        .map(|&var_id| {
            let name = pdb.task.get_variable_name(var_id).unwrap_or("<unknown>");
            format!("p{var_id}({name})")
        })
        .collect();
    let pattern_numeric_projected_ids = pdb.task.pattern_numeric_projected_ids();
    let num_headers: Vec<String> = pattern_numeric_projected_ids
        .iter()
        .map(|&var_id| format!("n{var_id}({})", pdb.task.numeric_variables()[var_id].name()))
        .collect();

    let prop_widths: Vec<usize> = prop_headers
        .iter()
        .zip(pattern_regular_projected_ids.iter())
        .map(|(header, &var_id)| {
            let value_width = pdb
                .states
                .iter()
                .map(|state| {
                    pdb.state_propositional_values(state)[var_id]
                        .to_string()
                        .len()
                })
                .max()
                .unwrap_or(1);
            header.len().max(value_width)
        })
        .collect();
    let num_widths: Vec<usize> = num_headers
        .iter()
        .zip(pattern_numeric_projected_ids.iter())
        .map(|(header, &var_id)| {
            let value_width = pdb
                .states
                .iter()
                .map(|state| fmt_numeric(pdb.state_numeric_values(state)[var_id]).len())
                .max()
                .unwrap_or(3);
            header.len().max(value_width)
        })
        .collect();

    let mut header_line = String::from("\nState | Flags | Distance | ");
    for (header, width) in prop_headers.iter().zip(prop_widths.iter()) {
        header_line.push_str(&format!("{header:>width$} | ", width = *width));
    }
    for (header, width) in num_headers.iter().zip(num_widths.iter()) {
        header_line.push_str(&format!("{header:>width$} | ", width = *width));
    }
    debug!("{header_line}");

    let mut separator = String::from("------|-------|----------|");
    for width in prop_widths.iter().chain(num_widths.iter()) {
        separator.push_str(&"-".repeat(*width + 2));
        separator.push('|');
    }
    debug!("{separator}");

    for (state_id, state) in pdb.states.iter().enumerate() {
        if state_id > 200 {
            debug!("... (truncated)");
            break;
        }
        let is_init = state_id == 0;
        let is_goal = pdb.is_goal_state(pdb.state_propositional_values(state));
        let is_dead_end = !pdb.distances[state_id].is_finite();

        let mut flags = String::new();
        if is_init {
            flags.push('I');
        }
        if is_goal {
            flags.push('G');
        }
        if is_dead_end {
            flags.push('D');
        }

        let mut line = format!(
            "{state_id:>5} | {flags:>5} | {:>8} | ",
            fmt_distance(pdb.distances[state_id])
        );

        for (&projected_var_id, width) in
            pattern_regular_projected_ids.iter().zip(prop_widths.iter())
        {
            line.push_str(&format!(
                "{:>width$} | ",
                pdb.state_propositional_values(state)[projected_var_id],
                width = *width
            ));
        }
        for (&projected_numeric_id, width) in
            pattern_numeric_projected_ids.iter().zip(num_widths.iter())
        {
            line.push_str(&format!(
                "{:>width$} | ",
                fmt_numeric(pdb.state_numeric_values(state)[projected_numeric_id]),
                width = *width
            ));
        }

        debug!("{line}");
    }
}

fn fmt_distance(distance: f64) -> String {
    if distance.is_finite() {
        format!("{distance:.3}")
    } else {
        "INF".to_string()
    }
}

fn fmt_numeric(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if !value.is_finite() {
        if value.is_sign_positive() {
            "INF".to_string()
        } else {
            "-INF".to_string()
        }
    } else if (value.fract()).abs() < 1e-12 {
        format!("{value:.1}")
    } else {
        format!("{value:.3}")
    }
}
