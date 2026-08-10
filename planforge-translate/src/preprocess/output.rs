//! Writing the reordered task as the SAS+ file the search reads.
//!
//! Every variable reference in the task is still phrased in the id the variable
//! had before the reordering, and is remapped to its level on the way out.

use std::io::{self, Write};

use crate::preprocess::{NO_LAYER, NO_LEVEL, PreprocessedTask, ReorderedTask};
use crate::sas_tasks::{SAS_FILE_VERSION, SasFact, assignment_operator};

/// The two shapes the SAS+ file is built out of, as methods on whatever the
/// task is written to.
///
/// A section is either a `begin_x` / `end_x` pair around a body, or a count
/// followed by exactly that many records. Both were spelled out at each of the
/// fifteen places below, and neither mistake is caught by anything: a `begin_`
/// without its `end_`, or a count that disagrees with what follows, produces a
/// file the search misparses rather than rejects. Passing the body as a closure
/// makes both impossible to write.
///
/// Blanket-implemented for every [`Write`], so every call below is a direct one.
trait SasSink: Write {
    /// `body` between `begin_{tag}` and `end_{tag}`.
    fn block(&mut self, tag: &str, body: impl FnOnce(&mut Self) -> io::Result<()>) -> io::Result<()>
    where
        Self: Sized,
    {
        writeln!(self, "begin_{tag}")?;
        body(self)?;
        writeln!(self, "end_{tag}")
    }

    /// How many `items` there are, then `record` for each of them.
    fn counted<T>(
        &mut self,
        items: &[T],
        mut record: impl FnMut(&mut Self, &T) -> io::Result<()>,
    ) -> io::Result<()>
    where
        Self: Sized,
    {
        writeln!(self, "{}", items.len())?;
        items.iter().try_for_each(|item| record(self, item))
    }
}

impl<W: Write> SasSink for W {}

/// The level a variable reference maps to. A reference to a pruned variable
/// would silently change the task, so it is an error rather than something to
/// leave out.
fn placed(level: i32, what: &str) -> i32 {
    assert_ne!(level, NO_LEVEL, "{what} was pruned");
    level
}

pub fn write_sas<W: Write>(reordered: &ReorderedTask, out: &mut W) -> io::Result<()> {
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
    // A fact list inside an effect is written on the effect's own line, so each
    // pair is preceded by its separator rather than followed by a newline.
    let inline_conditions = |out: &mut W, conditions: &[SasFact]| -> io::Result<()> {
        write!(out, "{}", conditions.len())?;
        conditions
            .iter()
            .try_for_each(|&(var, value)| write!(out, " {} {}", var_level(var), value))
    };
    let fact_lines = |out: &mut W, facts: &[SasFact]| -> io::Result<()> {
        out.counted(facts, |out, &(var, value)| {
            writeln!(out, "{} {}", var_level(var), value)
        })
    };

    out.block("version", |out| writeln!(out, "{SAS_FILE_VERSION}"))?;
    out.block("metric", |out| {
        writeln!(out, "{} {}", metric.optimization_criterion, metric.index)
    })?;

    out.counted(prop_order, |out, &var| {
        out.block("variable", |out| {
            let facts = &sas.variables.value_names[var];
            writeln!(out, "var{var}")?;
            writeln!(out, "{}", sas.variables.axiom_layers[var])?;
            writeln!(out, "{}", facts.len())?;
            facts.iter().try_for_each(|fact| writeln!(out, "{fact}"))
        })
    })?;

    writeln!(out, "{}", numeric_order.len())?;
    out.block("numeric_variables", |out| {
        numeric_order.iter().try_for_each(|&var| {
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
        })
    })?;

    out.counted(&sas.mutexes, |out, mutex| {
        out.block("mutex_group", |out| fact_lines(out, &mutex.facts))
    })?;

    out.block("state", |out| {
        prop_order
            .iter()
            .try_for_each(|&var| writeln!(out, "{}", sas.init.values[var]))
    })?;
    out.block("numeric_state", |out| {
        numeric_order
            .iter()
            .try_for_each(|&var| writeln!(out, "{}", sas.init.num_values[var]))
    })?;

    // The goal is written in the new variable order. A goal variable is
    // necessary by definition, so the reordering always gave it a level; a goal
    // that lost its variable would silently weaken the task.
    let mut goal_values: Vec<Option<usize>> = vec![None; prop_order.len()];
    for &(var, value) in &sas.goal.pairs {
        goal_values[var_level(var) as usize] = Some(value);
    }
    out.block("goal", |out| {
        writeln!(out, "{}", sas.goal.pairs.len())?;
        goal_values
            .iter()
            .enumerate()
            .filter_map(|(var, value)| value.map(|value| (var, value)))
            .try_for_each(|(var, value)| writeln!(out, "{var} {value}"))
    })?;

    out.counted(&sas.operators, |out, op| {
        out.block("operator", |out| {
            writeln!(out, "{}", op.output_name())?;
            fact_lines(out, &op.prevail)?;

            out.counted(&op.pre_post, |out, (var, pre, post, conditions)| {
                inline_conditions(out, conditions)?;
                writeln!(out, " {} {} {}", var_level(*var), pre, post)
            })?;

            // An assignment effect's conditions are propositional facts, exactly
            // like a `pre_post` effect's, so they map through the propositional
            // levels and not the numeric ones.
            out.counted(
                &op.assign_effects,
                |out, (var, operator, operand, conditions)| {
                    inline_conditions(out, conditions)?;
                    writeln!(
                        out,
                        " {} {} {}",
                        numeric_level(*var),
                        assignment_operator(operator),
                        numeric_level(*operand)
                    )
                },
            )?;

            writeln!(out, "{}", op.cost)
        })
    })?;

    out.counted(&sas.axioms, |out, axiom| {
        out.block("rule", |out| {
            let (effect_var, effect_value) = axiom.effect;
            fact_lines(out, &axiom.condition)?;
            // A derived variable is binary and defaults to the value the axiom
            // does not derive, so the rule names the value it overwrites.
            writeln!(
                out,
                "{} {} {}",
                var_level(effect_var),
                1 - effect_value,
                effect_value
            )
        })
    })?;

    writeln!(out, "{}", sas.comp_axioms.len())?;
    out.block("comparison_axioms", |out| {
        sas.comp_axioms.iter().try_for_each(|axiom| {
            writeln!(
                out,
                "{} {} {} {}",
                var_level(axiom.effect),
                axiom.comp,
                numeric_level(axiom.parts[0]),
                numeric_level(axiom.parts[1])
            )
        })
    })?;

    writeln!(out, "{}", sas.numeric_axioms.len())?;
    out.block("numeric_axioms", |out| {
        sas.numeric_axioms.iter().try_for_each(|axiom| {
            writeln!(
                out,
                "{} {} {} {}",
                numeric_level(axiom.effect),
                assignment_operator(&axiom.op),
                numeric_level(axiom.parts[0]),
                numeric_level(axiom.parts[1])
            )
        })
    })?;

    let (constraint_var, constraint_value) = sas.global_constraint;
    out.block("global_constraint", |out| {
        writeln!(out, "{} {}", var_level(constraint_var), constraint_value)
    })?;

    // The successor generator the search builds itself starts here, so this
    // marker has no `end_` of its own.
    writeln!(out, "begin_SG")
}
