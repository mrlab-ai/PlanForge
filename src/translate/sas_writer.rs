use crate::translate::sas::SASTask;
use std::fs::File;
use std::io::Write;

pub fn write_sas(task: &SASTask, path: &std::path::Path) -> anyhow::Result<()> {
    let mut f = File::create(path)?;
    // follow the Python translator textual layout for compatibility
    writeln!(f, "begin_version")?;
    writeln!(f, "4")?; // match python/translate/sas_tasks.SAS_FILE_VERSION
    writeln!(f, "end_version")?;
    // Metric: prefer canonical metric info when available, otherwise fall back to
    // scanning numeric variables for total-cost()/cost().
    writeln!(f, "begin_metric")?;
    let metric = if let Some(m) = &task.canonical_metric {
        m.clone()
    } else {
        let mut idx: isize = -1;
        for (i, nv) in task.numeric_variables.iter().enumerate() {
            if nv.name == "total-cost()" {
                idx = i as isize;
                break;
            }
            if idx == -1 && nv.name == "cost()" {
                idx = i as isize;
            }
        }
        ("<".to_string(), idx)
    };
    writeln!(f, "{} {}", metric.0, metric.1)?;
    writeln!(f, "end_metric")?;
    // Emit full begin_variable blocks
    if !task.canonical_variables.is_empty() {
        writeln!(f, "{}", task.canonical_variables.len())?;
        for var in &task.canonical_variables {
            writeln!(f, "begin_variable")?;
            writeln!(f, "{}", var.name)?;
            writeln!(f, "{}", var.axiom_layer)?;
            writeln!(f, "{}", var.values.len())?;
            for name in &var.values {
                writeln!(f, "{}", name)?;
            }
            writeln!(f, "end_variable")?;
        }
    } else {
        writeln!(f, "{}", task.variables.len())?;
        for (i, v) in task.variables.iter().enumerate() {
            writeln!(f, "begin_variable")?;
            writeln!(f, "var{}", i)?;
            writeln!(f, "-1")?;
            writeln!(f, "{}", v.value_names.len())?;
            for name in &v.value_names {
                writeln!(f, "{}", name)?;
            }
            writeln!(f, "end_variable")?;
        }
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
        for (v, val) in mg {
            writeln!(f, "{} {}", v, val)?;
        }
        writeln!(f, "end_mutex_group")?;
    }

    // Initial propositional state
    writeln!(f, "begin_state")?;
    for v in &task.init {
        writeln!(f, "{}", v)?;
    }
    writeln!(f, "end_state")?;

    // Initial numeric state
    writeln!(f, "begin_numeric_state")?;
    for v in &task.numeric_init {
        writeln!(f, "{}.0", v)?;
    }
    writeln!(f, "end_numeric_state")?;

    // Goal
    writeln!(f, "begin_goal")?;
    writeln!(f, "{}", task.goal.len())?;
    for (v, val) in &task.goal {
        writeln!(f, "{} {}", v, val)?;
    }
    writeln!(f, "end_goal")?;

    // Operators
    if !task.canonical_operators.is_empty() {
        writeln!(f, "{}", task.canonical_operators.len())?;
        for op in &task.canonical_operators {
            writeln!(f, "begin_operator")?;
            writeln!(f, "{}", op.name)?;
            writeln!(f, "{}", op.prevail.len())?;
            for (v, val) in &op.prevail {
                writeln!(f, "{} {}", v, val)?;
            }
            writeln!(f, "{}", op.pre_post.len())?;
            for eff in &op.pre_post {
                write!(f, "{} ", eff.condition.len())?;
                for (cv, cval) in &eff.condition {
                    write!(f, "{} {} ", cv, cval)?;
                }
                let pre = eff.pre.map(|p| p as i32).unwrap_or(-1);
                writeln!(f, "{} {} {}", eff.var, pre, eff.post)?;
            }
            writeln!(f, "{}", op.assign_effects.len())?;
            for assign in &op.assign_effects {
                write!(f, "{} ", assign.condition.len())?;
                for (cv, cval) in &assign.condition {
                    write!(f, "{} {} ", cv, cval)?;
                }
                let rhs = match &assign.rhs {
                    crate::translate::sas::CanonicalAssignRhs::Variable(idx) => idx.to_string(),
                    crate::translate::sas::CanonicalAssignRhs::Constant(val) => val.to_string(),
                };
                writeln!(f, "{} {} {}", assign.target, assign.op, rhs)?;
            }
            writeln!(f, "{}", op.cost)?;
            writeln!(f, "end_operator")?;
        }
    } else {
        writeln!(f, "# operators: {}", task.operators.len())?;
        for (oi, op) in task.operators.iter().enumerate() {
            writeln!(f, "begin_operator")?;
            writeln!(f, "name {} {}", oi, op.name)?;
            writeln!(f, "prevails {}", op.prevails.len())?;
            for (v, val) in &op.prevails {
                writeln!(f, "{} {}", v, val)?;
            }
            writeln!(f, "effects {}", op.effects.len())?;
            for (v, pre, post, cond) in &op.effects {
                write!(f, "{} ", cond.len())?;
                for (cv, cval) in cond {
                    write!(f, "{} {} ", cv, cval)?;
                }
                writeln!(f, "{} {} {}", v, pre, post)?;
            }
            writeln!(f, "num_effects {}", op.numeric_effects.len())?;
            for (nvar, assign_op, rhs_var, cond) in &op.numeric_effects {
                write!(f, "{} ", cond.len())?;
                for (cv, cval) in cond {
                    write!(f, "{} {} ", cv, cval)?;
                }
                writeln!(f, "{} {} {}", nvar, assign_op, rhs_var)?;
            }
            writeln!(f, "{}", op.cost)?;
            writeln!(f, "end_operator")?;
        }
    }

    // Propositional axioms (rules)
    writeln!(f, "{}", task.axioms.len())?;
    for ax in &task.axioms {
        writeln!(f, "begin_rule")?;
        writeln!(f, "{}", ax.condition.len())?;
        for (var, val) in &ax.condition {
            writeln!(f, "{} {}", var, val)?;
        }
        let (eff_var, eff_val) = ax.effect;
        writeln!(f, "{} {} {}", eff_var, 1 - eff_val, eff_val)?;
        writeln!(f, "end_rule")?;
    }

    // comparison axioms block
    writeln!(f, "{}", task.comparison_axioms.len())?;
    writeln!(f, "begin_comparison_axioms")?;
    for cax in &task.comparison_axioms {
        // format: <effect_var> <comp> <part0> <part1>
        writeln!(
            f,
            "{} {} {} {}",
            cax.effect_var, cax.comp, cax.parts[0], cax.parts[1]
        )?;
    }
    writeln!(f, "end_comparison_axioms")?;

    // numeric axioms: emit in textual block format
    writeln!(f, "{}", task.numeric_axioms.len())?;
    writeln!(f, "begin_numeric_axioms")?;
    for ax in task.numeric_axioms.iter() {
        // format: <effect> <op> <part0> [part1]
        let parts_str = ax
            .parts
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(f, "{} {} {}", ax.effect, ax.op, parts_str)?;
    }
    writeln!(f, "end_numeric_axioms")?;

    // Global constraint
    writeln!(f, "begin_global_constraint")?;
    if let Some(gc) = &task.global_constraint {
        writeln!(f, "{} {}", gc.0, gc.1)?;
    } else {
        writeln!(f, "0 -1")?;
    }
    writeln!(f, "end_global_constraint")?;

    Ok(())
}
