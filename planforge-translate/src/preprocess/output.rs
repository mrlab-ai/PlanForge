//! Writing the reordered task as the SAS+ file the search reads.
//!
//! Every variable reference in the task is still phrased in the id the variable
//! had before the reordering, and is remapped to its level on the way out.

use std::io::{self, Write};

use planforge_sas::sas_format::{SAS_FILE_VERSION, operator_cost_from_sas};
use tracing::info;

use crate::preprocess::{PreprocessedTask, ReorderedTask};
use crate::sas_tasks::{SasFact, assignment_operator};

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
    ///
    /// The two markers are written as bytes rather than formatted: a task with
    /// fifty thousand operators opens a hundred thousand blocks, and going
    /// through `core::fmt` to concatenate two known strings costs more than the
    /// copy does.
    fn block(&mut self, tag: &str, body: impl FnOnce(&mut Self) -> io::Result<()>) -> io::Result<()>
    where
        Self: Sized,
    {
        self.marker(b"begin_", tag)?;
        body(self)?;
        self.marker(b"end_", tag)
    }

    fn marker(&mut self, prefix: &[u8], tag: &str) -> io::Result<()> {
        self.write_all(prefix)?;
        self.write_all(tag.as_bytes())?;
        self.write_all(b"\n")
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

pub fn write_sas<W: Write>(reordered: &ReorderedTask, out: &mut W) -> io::Result<()> {
    let ReorderedTask {
        task: PreprocessedTask { sas, metric, .. },
        prop_order,
        numeric_order,
    } = reordered;

    let var_level = |var: usize| reordered.prop_level(var);
    let numeric_level = |var: usize| reordered.numeric_level(var);
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

    info!("Writing output...");

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
            let (numeric_type, layer, name) = reordered.numeric_variable(var);
            writeln!(out, "{} {layer} {name}", numeric_type.as_sas())
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

    // The goal's facts already name their new variables, so they are written
    // rather than mapped a second time.
    out.block("goal", |out| {
        out.counted(&reordered.ordered_goal(), |out, &(var, value)| {
            writeln!(out, "{var} {value}")
        })
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

            writeln!(out, "{}", operator_cost_from_sas(op.cost))
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
    writeln!(out, "begin_SG")?;
    info!("done");
    Ok(())
}
