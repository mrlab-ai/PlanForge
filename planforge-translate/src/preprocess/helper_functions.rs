use std::io::Write;

use tracing::debug;

use super::axiom::{AxiomFunctionalComparison, AxiomNumericComputation, AxiomRelational};
use super::fact::ExplicitFact;
use super::mutex_group::MutexGroup;
use super::operator::Operator;
use super::state::State;
use super::variable::{ExplicitVariable, NumericVariable};
use super::{GlobalConstraint, Metric, PRE_FILE_VERSION};

/// Writes the reordered task in the SAS+ format the search reads. Every
/// variable reference is remapped through the level the causal graph gave it;
/// `orig_vars` and `orig_numeric_vars` are indexed by the old id and only
/// carry that mapping.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn to_sas_writer<W: Write>(
    orig_vars: &[ExplicitVariable],
    orig_numeric_vars: &[NumericVariable],
    ordered_vars: &[ExplicitVariable],
    ordered_numeric_vars: &[NumericVariable],
    metric: &Metric,
    mutexes: &Vec<MutexGroup>,
    initial_state: &State,
    goals: &Vec<ExplicitFact>,
    operators: &Vec<Operator>,
    axioms_rel: &Vec<AxiomRelational>,
    axioms_func_ass: &Vec<AxiomNumericComputation>,
    axioms_func_comp: &Vec<AxiomFunctionalComparison>,
    constraint: &Option<GlobalConstraint>,
    outfile: &mut W,
) {
    writeln!(outfile, "begin_version").unwrap();
    writeln!(outfile, "{}", PRE_FILE_VERSION).unwrap();
    writeln!(outfile, "end_version").unwrap();

    writeln!(outfile, "begin_metric").unwrap();
    writeln!(
        outfile,
        "{} {}",
        metric.optimization_criterion, metric.index
    )
    .unwrap();
    writeln!(outfile, "end_metric").unwrap();

    let num_vars = ordered_vars.len();
    writeln!(outfile, "{}", num_vars).unwrap();
    debug!("Variables in output are: ");
    for var in ordered_vars {
        var.to_sas(outfile);
        debug!("{} {{", var.get_name());
        for i in 0..var.get_range() {
            debug!("{}, ", var.get_fact_name(i));
        }
        debug!("}}");
        debug!("Initial value = {}", initial_state.get(var.index));
    }

    debug!("Numeric Variables in output are: ");
    writeln!(outfile, "{}", ordered_numeric_vars.len()).unwrap();
    writeln!(outfile, "begin_numeric_variables").unwrap();
    for numeric_var in ordered_numeric_vars {
        numeric_var.dump();
        numeric_var.to_sas(outfile);
    }
    writeln!(outfile, "end_numeric_variables").unwrap();

    writeln!(outfile, "{}", mutexes.len()).unwrap();
    for mutex in mutexes {
        mutex.to_sas(outfile, orig_vars);
    }

    writeln!(outfile, "begin_state").unwrap();
    for var in ordered_vars {
        writeln!(outfile, "{}", initial_state.get(var.index)).unwrap();
    }
    writeln!(outfile, "end_state").unwrap();
    writeln!(outfile, "begin_numeric_state").unwrap();
    for numvar in ordered_numeric_vars {
        writeln!(outfile, "{}", initial_state.get_nv(numvar.index)).unwrap();
    }
    writeln!(outfile, "end_numeric_state").unwrap();

    let mut ordered_goal_values: Vec<i32> = vec![-1; num_vars];
    for goal in goals {
        let var_index = orig_vars[goal.var].get_level();
        if var_index > -1 {
            ordered_goal_values[var_index as usize] = goal.value as i32;
        }
    }
    writeln!(outfile, "begin_goal").unwrap();
    writeln!(outfile, "{}", goals.len()).unwrap();
    for i in 0..num_vars {
        if ordered_goal_values[i] != -1 {
            writeln!(outfile, "{} {}", i, ordered_goal_values[i]).unwrap();
        }
    }
    writeln!(outfile, "end_goal").unwrap();

    writeln!(outfile, "{}", operators.len()).unwrap();
    for op in operators {
        op.to_sas(outfile, orig_vars, orig_numeric_vars);
    }

    writeln!(outfile, "{}", axioms_rel.len()).unwrap();
    for ax in axioms_rel {
        ax.to_sas(outfile, orig_vars);
    }

    writeln!(outfile, "{}", axioms_func_comp.len()).unwrap();
    writeln!(outfile, "begin_comparison_axioms").unwrap();
    for ax in axioms_func_comp {
        ax.to_sas(outfile, orig_vars, orig_numeric_vars);
    }
    writeln!(outfile, "end_comparison_axioms").unwrap();

    writeln!(outfile, "{}", axioms_func_ass.len()).unwrap();
    writeln!(outfile, "begin_numeric_axioms").unwrap();
    for ax in axioms_func_ass {
        ax.to_sas(outfile, orig_numeric_vars);
    }
    writeln!(outfile, "end_numeric_axioms").unwrap();

    if let Some(gc) = constraint {
        writeln!(outfile, "begin_global_constraint").unwrap();
        writeln!(outfile, "{} {}", orig_vars[gc.var].get_level(), gc.value).unwrap();
        writeln!(outfile, "end_global_constraint").unwrap();
    }

    writeln!(outfile, "begin_SG").unwrap();
}
