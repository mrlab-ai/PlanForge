use crate::translate::sas::SASTask;
use std::fs::File;
use std::io::Write;

pub fn write_sas(task: &SASTask, path: &std::path::Path) -> anyhow::Result<()> {
    let mut f = File::create(path)?;
    // follow the Python translator textual layout for compatibility
    writeln!(f, "begin_version")?;
    writeln!(f, "4")?; // match python/translate/sas_tasks.SAS_FILE_VERSION
    writeln!(f, "end_version")?;
    // Metric: try to find a numeric variable named 'total-cost()' or 'cost()'; if none, use -1
    writeln!(f, "begin_metric")?;
    let mut metric_idx: isize = -1;
    // prefer total-cost(), else cost()
    for (i, nv) in task.numeric_variables.iter().enumerate() {
        if nv.name == "total-cost()" { metric_idx = i as isize; break; }
        if metric_idx == -1 && nv.name == "cost()" { metric_idx = i as isize; }
    }
    // Python prints '< idx' where idx is -1 for unit-cost, else the numeric var index
    writeln!(f, "< {}", metric_idx)?;
    writeln!(f, "end_metric")?;
    // Emit full begin_variable blocks for each variable. We don't yet compute axiom layers here so use -1.
    writeln!(f, "{}", task.variables.len())?;
    for (i, v) in task.variables.iter().enumerate() {
        writeln!(f, "begin_variable")?;
        writeln!(f, "var{}", i)?;
        writeln!(f, "-1")?; // axiom layer placeholder
        writeln!(f, "{}", v.value_names.len())?;
        for name in &v.value_names {
            writeln!(f, "{}", name)?;
        }
        writeln!(f, "end_variable")?;
    }
    // Numeric variables section (minimal, parseable by numeric_parser)
    writeln!(f, "{}", task.numeric_variables.len())?;
    writeln!(f, "begin_numeric_variables")?;
    for nv in task.numeric_variables.iter() {
        // nv.ntype should be one of D/C/R/I; axiom_layer may be -1
        writeln!(f, "{} {} PNE {}", nv.ntype, nv.axiom_layer, nv.name)?;
    }
    writeln!(f, "end_numeric_variables")?;

    // Mutex groups
    writeln!(f, "{}", task.mutex_groups.len())?;
    for mg in &task.mutex_groups {
        writeln!(f, "begin_mutex_group")?;
        // each mutex group line is a (var, val) pair; we encode as 2-column rows count
        writeln!(f, "{}", mg.len())?;
        for (v, val) in mg { writeln!(f, "{} {}", v, val)?; }
        writeln!(f, "end_mutex_group")?;
    }
    // comparison axioms block
    writeln!(f, "{}", task.comparison_axioms.len())?;
    writeln!(f, "begin_comparison_axioms")?;
    for cax in &task.comparison_axioms {
        // format: <effect> <comp> <part0> <part1>
        writeln!(f, "{} {} {} {}", cax.effect, cax.comp, cax.parts[0], cax.parts[1])?;
    }
    writeln!(f, "end_comparison_axioms")?;

    // numeric axioms: emit in textual block format
    writeln!(f, "{}", task.numeric_axioms.len())?;
    writeln!(f, "begin_numeric_axioms")?;
    for ax in task.numeric_axioms.iter() {
        match ax {
            crate::translate::sas::NumericAxiom::VarConst(effect, opstr, val) => {
                writeln!(f, "{} {} {} {}", effect, opstr, effect, val)?;
            }
            crate::translate::sas::NumericAxiom::VarVar(effect, opstr, other) => {
                writeln!(f, "{} {} {} {}", effect, opstr, effect, other)?;
            }
        }
    }
    writeln!(f, "end_numeric_axioms")?;
    // Initial propositional and numeric states
    writeln!(f, "begin_state")?;
    for v in &task.init { writeln!(f, "{}", v)?; }
    writeln!(f, "end_state")?;
    writeln!(f, "begin_numeric_state")?;
    for v in &task.numeric_init { writeln!(f, "{}.0", v)?; }
    writeln!(f, "end_numeric_state")?;

    // Goal
    writeln!(f, "begin_goal")?;
    writeln!(f, "{}", task.goal.len())?;
    for (v, val) in &task.goal { writeln!(f, "{} {}", v, val)?; }
    writeln!(f, "end_goal")?;

    writeln!(f, "# operators: {}", task.operators.len())?;
    for (oi, op) in task.operators.iter().enumerate() {
        writeln!(f, "begin_operator")?;
        writeln!(f, "name {} {}", oi, op.name)?;
    writeln!(f, "prevails {}", op.prevails.len())?;
    for (v, val) in &op.prevails { writeln!(f, "{} {}", v, val)?; }
    writeln!(f, "num_preconds {}", op.numeric_preconds.len())?;
    for idx in &op.numeric_preconds { writeln!(f, "pax {}", idx)?; }
    writeln!(f, "effects {}", op.effects.len())?;
    for (v, pre, post) in &op.effects { writeln!(f, "{} {} {}", v, pre.map(|p| p as i64).unwrap_or(-1), post)?; }
    writeln!(f, "num_effects {}", op.numeric_effects.len())?;
    for (ni, delta) in &op.numeric_effects { writeln!(f, "{} {}", ni, delta)?; }
        writeln!(f, "end_operator")?;
    }
    Ok(())
}
