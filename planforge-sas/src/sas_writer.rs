//! Writing a task as SAS+ text.
//!
//! The mirror image of [`crate::numeric_parser`], over the same
//! [`SasTaskParts`] it reads: the file is the interoperability surface, so there
//! is exactly one writer of it, and it sits beside the reader so that a change
//! to either has the other in view.
//!
//! What the sections *mean* is [`crate::sas_format`]'s; this module knows only
//! the syntax.

use std::io::{self, Write};

use crate::numeric_task::ExplicitFact;
use crate::sas_format::{SasTaskParts, optional_value_to_sas};

/// The two shapes the SAS+ file is built out of, as methods on whatever the
/// task is written to.
///
/// A section is either a `begin_x` / `end_x` pair around a body, or a count
/// followed by exactly that many records. Both used to be spelled out at each of
/// the fifteen places below, and neither mistake is caught by anything: a
/// `begin_` without its `end_`, or a count that disagrees with what follows,
/// produces a file the reader misparses rather than rejects. Passing the body as
/// a closure makes both impossible to write.
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

    /// One value per line, between the section's markers.
    fn value_lines(&mut self, tag: &str, values: &[impl std::fmt::Display]) -> io::Result<()>
    where
        Self: Sized,
    {
        self.block(tag, |out| {
            values.iter().try_for_each(|value| writeln!(out, "{value}"))
        })
    }

    /// A fact list as a section of its own: the count, then one fact per line.
    fn fact_lines(&mut self, facts: &[ExplicitFact]) -> io::Result<()>
    where
        Self: Sized,
    {
        self.counted(facts, |out, fact| {
            writeln!(out, "{} {}", fact.var(), fact.value())
        })
    }

    /// A fact list inside an effect, which shares the effect's line: each fact
    /// is preceded by its separator rather than followed by a newline.
    fn inline_facts(&mut self, facts: &[ExplicitFact]) -> io::Result<()> {
        write!(self, "{}", facts.len())?;
        facts
            .iter()
            .try_for_each(|fact| write!(self, " {} {}", fact.var(), fact.value()))
    }
}

impl<W: Write> SasSink for W {}

/// Write `parts` as the SAS+ text another planner reads.
pub fn write_sas<W: Write>(parts: &SasTaskParts, out: &mut W) -> io::Result<()> {
    out.block("version", |out| writeln!(out, "{}", parts.version))?;
    out.block("metric", |out| {
        let (direction, index) = parts.metric.as_sas();
        writeln!(out, "{direction} {index}")
    })?;

    out.counted(&parts.variables, |out, variable| {
        out.block("variable", |out| {
            writeln!(out, "{}", variable.name)?;
            writeln!(out, "{}", optional_value_to_sas(variable.axiom_layer))?;
            writeln!(out, "{}", variable.domain_size)?;
            variable
                .fact_names
                .iter()
                .try_for_each(|fact| writeln!(out, "{fact}"))
        })
    })?;

    writeln!(out, "{}", parts.numeric_variables.len())?;
    out.block("numeric_variables", |out| {
        parts.numeric_variables.iter().try_for_each(|variable| {
            writeln!(
                out,
                "{} {} {}",
                variable.get_type().as_sas(),
                optional_value_to_sas(variable.axiom_layer()),
                variable.name()
            )
        })
    })?;

    out.counted(&parts.mutexes, |out, mutex| {
        out.block("mutex_group", |out| out.fact_lines(mutex))
    })?;

    out.value_lines("state", &parts.state)?;
    out.value_lines("numeric_state", &parts.numeric_state)?;
    out.block("goal", |out| out.fact_lines(&parts.goals))?;

    out.counted(&parts.operators, |out, operator| {
        out.block("operator", |out| {
            writeln!(out, "{}", operator.name)?;
            out.fact_lines(&operator.prevail)?;
            out.counted(&operator.effects, |out, effect| {
                out.inline_facts(effect.conditions())?;
                writeln!(
                    out,
                    " {} {} {}",
                    effect.var_id(),
                    optional_value_to_sas(effect.precondition_value()),
                    effect.value()
                )
            })?;
            // An assignment effect's conditions are propositional facts, exactly
            // like a propositional effect's; only the effect itself names
            // numeric variables.
            out.counted(&operator.assignment_effects, |out, effect| {
                out.inline_facts(effect.conditions())?;
                writeln!(
                    out,
                    " {} {} {}",
                    effect.affected_var_id(),
                    effect.operation().as_sas(),
                    effect.var_id()
                )
            })?;
            writeln!(out, "{}", operator.cost)
        })
    })?;

    out.counted(&parts.axioms, |out, axiom| {
        out.block("rule", |out| {
            out.fact_lines(axiom.conditions())?;
            writeln!(
                out,
                "{} {} {}",
                axiom.var_id(),
                axiom.precondition_value(),
                axiom.effect_value()
            )
        })
    })?;

    writeln!(out, "{}", parts.comparison_axioms.len())?;
    out.block("comparison_axioms", |out| {
        parts.comparison_axioms.iter().try_for_each(|axiom| {
            writeln!(
                out,
                "{} {} {} {}",
                axiom.get_affected_var_id(),
                axiom.get_operator().as_sas(),
                axiom.get_left_var_id(),
                axiom.get_right_var_id()
            )
        })
    })?;

    writeln!(out, "{}", parts.assignment_axioms.len())?;
    out.block("numeric_axioms", |out| {
        parts.assignment_axioms.iter().try_for_each(|axiom| {
            writeln!(
                out,
                "{} {} {} {}",
                axiom.get_affected_var_id(),
                axiom.get_operator().as_sas(),
                axiom.get_left_var_id(),
                axiom.get_right_var_id()
            )
        })
    })?;

    out.block("global_constraint", |out| {
        writeln!(
            out,
            "{} {}",
            parts.global_constraint.var(),
            parts.global_constraint.value()
        )
    })?;

    // The successor generator the search builds itself starts here, so this
    // marker has no `end_` of its own.
    writeln!(out, "begin_SG")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric_parser::parse_sas_parts;

    /// The writer is the reader's inverse, byte for byte.
    ///
    /// That is what makes the file an interoperability surface rather than a
    /// dump: a task read from a file and written back out is the same file, so
    /// the two halves cannot drift apart on a field one of them spells
    /// differently.
    fn assert_round_trips(sas_text: &str, what: &str) {
        let (rest, parts) =
            parse_sas_parts(sas_text).unwrap_or_else(|error| panic!("{what} parses: {error}"));
        assert_eq!(rest, "", "{what} was not read to its end");

        let mut written: Vec<u8> = Vec::new();
        write_sas(&parts, &mut written).expect("writing into a `Vec` cannot fail");
        let written = String::from_utf8(written).expect("the writer emits UTF-8");
        assert_eq!(written, sas_text, "{what} changed on the way back out");
    }

    #[test]
    fn the_checked_in_sas_tasks_survive_a_round_trip() {
        let assets =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/assets/numeric_sas");
        for fixture in ["example2.sas", "example5.sas"] {
            let path = assets.join(fixture);
            let sas_text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
            assert_round_trips(&sas_text, fixture);
        }
    }
}
