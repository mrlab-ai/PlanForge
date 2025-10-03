use crate::translate::sas::SASTask;
use std::fs::File;
use std::io::Write;

pub fn write_sas(task: &SASTask, path: &std::path::Path) -> anyhow::Result<()> {
    let mut f = File::create(path)?;
    // follow the Python translator textual layout for compatibility
    writeln!(f, "begin_version")?;
    writeln!(f, "4")?; // match python/translate/sas_tasks.SAS_FILE_VERSION
    writeln!(f, "end_version")?;
    writeln!(f, "begin_metric")?;
    writeln!(f, "0")?; // metric placeholder
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
    writeln!(f, "# numeric variables: {}", task.numeric_variables.len())?;
    for (i, nv) in task.numeric_variables.iter().enumerate() {
        writeln!(f, "num {} {} {}", i, nv.name, nv.initial.map(|v| v.to_string()).unwrap_or_else(|| "none".to_string()))?;
    }
    writeln!(f, "# numeric init: {}", task.numeric_init.len())?;
    for (i, v) in task.numeric_init.iter().enumerate() {
        writeln!(f, "numinit {} {}", i, v)?;
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
