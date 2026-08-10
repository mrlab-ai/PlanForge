//! Writing the reordered task as the SAS+ file the search reads.
//!
//! Every variable reference in the task is still phrased in the id the variable
//! had before the reordering, and is remapped to its level on the way out.

use std::io::Write;

use crate::preprocess::{NO_LAYER, NO_LEVEL, PreprocessedTask, ReorderedTask};
use crate::sas_tasks::{SAS_FILE_VERSION, SasFact, assignment_operator};

/// The level a variable reference maps to. A reference to a pruned variable
/// would silently change the task, so it is an error rather than something to
/// leave out.
fn placed(level: i32, what: &str) -> i32 {
    assert_ne!(level, NO_LEVEL, "{what} was pruned");
    level
}

pub fn write_sas<W: Write>(reordered: &ReorderedTask, out: &mut W) {
    let ReorderedTask {
        task:
            PreprocessedTask {
                sas,
                metric,
                vars,
                numeric_vars,
            },
        prop_order,
        numeric_order,
    } = reordered;

    let var_level = |var: usize| placed(vars[var].level(), "variable");
    let numeric_level = |var: usize| placed(numeric_vars[var].level(), "numeric variable");
    let write_conditions = |out: &mut W, conditions: &[SasFact]| {
        write!(out, "{}", conditions.len()).unwrap();
        for &(var, value) in conditions {
            write!(out, " {} {}", var_level(var), value).unwrap();
        }
    };

    writeln!(out, "begin_version").unwrap();
    writeln!(out, "{SAS_FILE_VERSION}").unwrap();
    writeln!(out, "end_version").unwrap();

    writeln!(out, "begin_metric").unwrap();
    writeln!(out, "{} {}", metric.optimization_criterion, metric.index).unwrap();
    writeln!(out, "end_metric").unwrap();

    writeln!(out, "{}", prop_order.len()).unwrap();
    for &var in prop_order {
        let facts = &sas.variables.value_names[var];
        writeln!(out, "begin_variable").unwrap();
        writeln!(out, "var{var}").unwrap();
        writeln!(out, "{}", sas.variables.axiom_layers[var]).unwrap();
        writeln!(out, "{}", facts.len()).unwrap();
        for fact in facts {
            writeln!(out, "{fact}").unwrap();
        }
        writeln!(out, "end_variable").unwrap();
    }

    writeln!(out, "{}", numeric_order.len()).unwrap();
    writeln!(out, "begin_numeric_variables").unwrap();
    for &var in numeric_order {
        let state = numeric_vars[var];
        assert!(state.is_necessary());
        let layer = sas.numeric_variables.axiom_layers[var];
        assert!(layer >= NO_LAYER);
        writeln!(
            out,
            "{} {} {}",
            state.ntype().as_sas(),
            layer,
            sas.numeric_variables.variable_names[var]
        )
        .unwrap();
    }
    writeln!(out, "end_numeric_variables").unwrap();

    writeln!(out, "{}", sas.mutexes.len()).unwrap();
    for mutex in &sas.mutexes {
        writeln!(out, "begin_mutex_group").unwrap();
        writeln!(out, "{}", mutex.facts.len()).unwrap();
        for &(var, value) in &mutex.facts {
            writeln!(out, "{} {}", var_level(var), value).unwrap();
        }
        writeln!(out, "end_mutex_group").unwrap();
    }

    writeln!(out, "begin_state").unwrap();
    for &var in prop_order {
        writeln!(out, "{}", sas.init.values[var]).unwrap();
    }
    writeln!(out, "end_state").unwrap();
    writeln!(out, "begin_numeric_state").unwrap();
    for &var in numeric_order {
        writeln!(out, "{}", sas.init.num_values[var]).unwrap();
    }
    writeln!(out, "end_numeric_state").unwrap();

    // The goal is written in the new variable order. A goal variable is
    // necessary by definition, so the reordering always gave it a level; a goal
    // that lost its variable would silently weaken the task.
    let mut goal_values: Vec<Option<usize>> = vec![None; prop_order.len()];
    for &(var, value) in &sas.goal.pairs {
        goal_values[var_level(var) as usize] = Some(value);
    }
    writeln!(out, "begin_goal").unwrap();
    writeln!(out, "{}", sas.goal.pairs.len()).unwrap();
    for (var, value) in goal_values.iter().enumerate() {
        if let Some(value) = value {
            writeln!(out, "{var} {value}").unwrap();
        }
    }
    writeln!(out, "end_goal").unwrap();

    writeln!(out, "{}", sas.operators.len()).unwrap();
    for op in &sas.operators {
        writeln!(out, "begin_operator").unwrap();
        writeln!(out, "{}", op.output_name()).unwrap();

        writeln!(out, "{}", op.prevail.len()).unwrap();
        for &(var, value) in &op.prevail {
            writeln!(out, "{} {}", var_level(var), value).unwrap();
        }

        writeln!(out, "{}", op.pre_post.len()).unwrap();
        for (var, pre, post, conditions) in &op.pre_post {
            write_conditions(out, conditions);
            writeln!(out, " {} {} {}", var_level(*var), pre, post).unwrap();
        }

        writeln!(out, "{}", op.assign_effects.len()).unwrap();
        for (var, operator, operand, conditions) in &op.assign_effects {
            // An assignment effect's conditions are propositional facts,
            // exactly like a `pre_post` effect's, so they map through the
            // propositional levels and not the numeric ones.
            write_conditions(out, conditions);
            writeln!(
                out,
                " {} {} {}",
                numeric_level(*var),
                assignment_operator(operator),
                numeric_level(*operand)
            )
            .unwrap();
        }

        writeln!(out, "{}", op.cost).unwrap();
        writeln!(out, "end_operator").unwrap();
    }

    writeln!(out, "{}", sas.axioms.len()).unwrap();
    for axiom in &sas.axioms {
        let (effect_var, effect_value) = axiom.effect;
        writeln!(out, "begin_rule").unwrap();
        writeln!(out, "{}", axiom.condition.len()).unwrap();
        for &(var, value) in &axiom.condition {
            writeln!(out, "{} {}", var_level(var), value).unwrap();
        }
        // A derived variable is binary and defaults to the value the axiom does
        // not derive, so the rule names the value it overwrites.
        writeln!(
            out,
            "{} {} {}",
            var_level(effect_var),
            1 - effect_value,
            effect_value
        )
        .unwrap();
        writeln!(out, "end_rule").unwrap();
    }

    writeln!(out, "{}", sas.comp_axioms.len()).unwrap();
    writeln!(out, "begin_comparison_axioms").unwrap();
    for axiom in &sas.comp_axioms {
        writeln!(
            out,
            "{} {} {} {}",
            var_level(axiom.effect),
            axiom.comp,
            numeric_level(axiom.parts[0]),
            numeric_level(axiom.parts[1])
        )
        .unwrap();
    }
    writeln!(out, "end_comparison_axioms").unwrap();

    writeln!(out, "{}", sas.numeric_axioms.len()).unwrap();
    writeln!(out, "begin_numeric_axioms").unwrap();
    for axiom in &sas.numeric_axioms {
        writeln!(
            out,
            "{} {} {} {}",
            numeric_level(axiom.effect),
            assignment_operator(&axiom.op),
            numeric_level(axiom.parts[0]),
            numeric_level(axiom.parts[1])
        )
        .unwrap();
    }
    writeln!(out, "end_numeric_axioms").unwrap();

    let (constraint_var, constraint_value) = sas.global_constraint;
    writeln!(out, "begin_global_constraint").unwrap();
    writeln!(out, "{} {}", var_level(constraint_var), constraint_value).unwrap();
    writeln!(out, "end_global_constraint").unwrap();

    writeln!(out, "begin_SG").unwrap();
}
