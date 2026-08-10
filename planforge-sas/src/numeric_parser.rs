//! The text syntax of the SAS+ format.
//!
//! What the sections *mean* lives in [`crate::sas_format`], so that the
//! translator can build the same task without going through text at all.

use crate::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use crate::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, Metric, NumericRootTask,
    NumericType, NumericVariable,
};
use crate::sas_format::{
    SasOperator, SasTaskParts, SasVariable, axiom_layer_from_sas, effect_precondition_from_sas,
};
use nom::Parser;
use nom::bytes::complete::take_while1;
use nom::combinator::map_opt;
use nom::multi::{count, length_count, many0};
use nom::number::complete::double;
use nom::{
    IResult,
    branch::alt,
    bytes::complete::tag,
    character::complete::{
        alphanumeric1, char, digit1, i32, line_ending, not_line_ending, space1, u32, u64, usize,
    },
    combinator::map_res,
    sequence::{delimited, pair, separated_pair, terminated},
};
use std::vec;

/// A section the format states as a count on a line of its own and then exactly
/// that many records: the reader's counterpart to the writer's `counted`.
fn counted<'a, T>(
    input: &'a str,
    record: impl FnMut(&'a str) -> IResult<&'a str, T>,
) -> IResult<&'a str, Vec<T>> {
    length_count(terminated(u32, line_ending), record).parse(input)
}

/// One fact on a line of its own, as the goal, a mutex group, an axiom body and
/// an operator's prevail conditions all state them.
fn parse_fact_line(input: &str) -> IResult<&str, ExplicitFact> {
    let (input, (var, value)) = terminated(
        separated_pair(parse_integer, space1, parse_integer),
        line_ending,
    )
    .parse(input)?;
    Ok((
        input,
        ExplicitFact::propositional(var as usize, value as usize),
    ))
}

/// One token of an operator table, as the format writes it: a run of the
/// characters the tables are spelled with, which the table then has to accept.
fn parse_operator_token(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| matches!(c, '<' | '>' | '=' | '!' | '+' | '-' | '*' | '/'))(input)
}

fn parse_version(input: &str) -> IResult<&str, u32> {
    let (input, _) = tag("begin_version")(input)?;
    let (input, _) = line_ending(input)?;
    let (input, version) = u32(input)?;
    let (input, _) = line_ending(input)?;
    let (input, _) = tag("end_version")(input)?;
    let (input, _) = line_ending(input)?;
    Ok((input, version))
}

fn parse_metric(input: &str) -> IResult<&str, Metric> {
    let (input, _) = tag("begin_metric")(input)?;
    let (input, _) = line_ending(input)?;
    let (input, direction) = alt((char('<'), char('>'))).parse(input)?;
    let (input, _) = space1(input)?;
    let (input, index) = usize(input)?;
    let (input, _) = line_ending(input)?;
    let (input, _) = tag("end_metric")(input)?;
    let (input, _) = line_ending(input)?;

    let metric = Metric::from_sas(direction, index)
        .expect("the direction was parsed as one of the two the format spells");
    Ok((input, metric))
}

fn parse_variable(input: &str) -> IResult<&str, SasVariable> {
    let (input, _) = tag("begin_variable")(input)?;
    let (input, _) = line_ending(input)?;
    let (input, variable_name) = alphanumeric1(input)?;
    let (input, _) = line_ending(input)?;
    let (input, axiom_layer) = i32(input)?;
    let (input, _) = line_ending(input)?;
    let (input, domain_size) = usize(input)?;
    let (input, _) = line_ending(input)?;

    let (input, fact_names) =
        count(terminated(not_line_ending, line_ending), domain_size).parse(input)?;
    let fact_names = fact_names.into_iter().map(str::to_owned).collect();
    let (input, _) = tag("end_variable")(input)?;
    let (input, _) = line_ending(input)?;
    let variable = SasVariable {
        domain_size,
        name: variable_name.to_string(),
        fact_names,
        axiom_layer: axiom_layer_from_sas(axiom_layer),
    };
    Ok((input, variable))
}

fn parse_all_variables(input: &str) -> IResult<&str, Vec<SasVariable>> {
    counted(input, parse_variable)
}

fn parse_numeric_type(input: &str) -> IResult<&str, NumericType> {
    map_opt(alphanumeric1, NumericType::from_sas).parse(input)
}

fn parse_assignment_operation(input: &str) -> IResult<&str, AssignmentOperation> {
    map_opt(parse_operator_token, AssignmentOperation::from_sas).parse(input)
}

fn parse_name(input: &str) -> IResult<&str, String> {
    // `take_while1`` takes all characters until a newline or end of input.
    let (input, name) = take_while1(|c: char| c != '\n')(input)?;
    Ok((input, name.trim().to_string()))
}

fn parse_numeric_variable(input: &str) -> IResult<&str, NumericVariable> {
    let (input, numeric_type) = parse_numeric_type(input)?;
    let (input, _) = space1(input)?;
    let (input, layer) = i32(input)?;
    let (input, _) = space1(input)?;
    let (input, variable_name) = parse_name(input)?;
    let (input, _) = line_ending(input)?;
    let var = NumericVariable::new(variable_name, numeric_type, axiom_layer_from_sas(layer));
    Ok((input, var))
}

/// The numeric variables, whose count the format states *before* the section's
/// markers rather than inside them.
fn parse_all_numeric_variables(input: &str) -> IResult<&str, Vec<NumericVariable>> {
    let (input, num_numeric_variables) = terminated(u32, line_ending).parse(input)?;
    delimited(
        pair(tag("begin_numeric_variables"), line_ending),
        count(parse_numeric_variable, num_numeric_variables as usize),
        pair(tag("end_numeric_variables"), line_ending),
    )
    .parse(input)
}

fn parse_integer(input: &str) -> IResult<&str, u32> {
    map_res(digit1, str::parse::<u32>).parse(input)
}

fn parse_mutex_group(input: &str) -> IResult<&str, Vec<ExplicitFact>> {
    delimited(
        pair(tag("begin_mutex_group"), line_ending),
        |input| counted(input, parse_fact_line),
        pair(tag("end_mutex_group"), line_ending),
    )
    .parse(input)
}

fn parse_mutexes(input: &str) -> IResult<&str, Vec<Vec<ExplicitFact>>> {
    counted(input, parse_mutex_group)
}

/// The initial state, one value per line. The format states no count for it --
/// the section holds one line per variable -- so the values are read until one
/// does not parse, which is where the closing marker stands.
fn parse_state(input: &str) -> IResult<&str, Vec<usize>> {
    delimited(
        pair(tag("begin_state"), line_ending),
        many0(terminated(usize, line_ending)),
        pair(tag("end_state"), line_ending),
    )
    .parse(input)
}

fn parse_numeric_state(input: &str) -> IResult<&str, Vec<f64>> {
    delimited(
        pair(tag("begin_numeric_state"), line_ending),
        many0(terminated(double, line_ending)),
        pair(tag("end_numeric_state"), line_ending),
    )
    .parse(input)
}

fn parse_goal(input: &str) -> IResult<&str, Vec<ExplicitFact>> {
    delimited(
        pair(tag("begin_goal"), line_ending),
        |input| counted(input, parse_fact_line),
        pair(tag("end_goal"), line_ending),
    )
    .parse(input)
}

fn parse_operator(input: &str) -> IResult<&str, SasOperator> {
    let (input, _) = tag("begin_operator")(input)?;
    let (input, _) = line_ending(input)?;
    let (input, name) = not_line_ending(input)?;
    let (input, _) = line_ending(input)?;
    let (input, prevail) = counted(input, parse_fact_line)?;

    let (input, num_effects) = u32(input)?;
    let (input, _) = line_ending(input)?;

    let mut input = input;
    let mut effects = vec![];
    for _ in 0..num_effects {
        let (loop_input, num_conditions) = u32(input)?;
        let (loop_input, _) = tag(" ")(loop_input)?;
        let mut effect_conditions = vec![];
        let mut loop_input = loop_input;
        for _ in 0..num_conditions {
            let mut parser = separated_pair(parse_integer, space1, parse_integer);
            let (loop_input2, condition) = parser.parse(loop_input)?;
            let condition = ExplicitFact::propositional(condition.0 as usize, condition.1 as usize);
            effect_conditions.push(condition);
            let (loop_input2, _) = space1(loop_input2)?;
            loop_input = loop_input2;
        }

        let (loop_input, effect_var_id) = usize(loop_input)?;
        let (loop_input, _) = space1(loop_input)?;
        let (loop_input, precondition_field) = i32(loop_input)?;
        let (loop_input, _) = space1(loop_input)?;
        let (loop_input, effect_value) = usize(loop_input)?;

        let effect = Effect::new(
            effect_conditions,
            effect_var_id,
            effect_precondition_from_sas(precondition_field),
            effect_value,
        );
        effects.push(effect);
        let (loop_input, _) = line_ending(loop_input)?;
        input = loop_input;
    }

    let mut assignment_effects = vec![];
    let (input, num_assignment_effects) = u32(input)?;
    let (mut input, _) = line_ending(input)?;
    for _ in 0..num_assignment_effects {
        let (loop_input, cond_count) = u32(input)?;
        let is_conditional_effect = cond_count > 0;
        let mut conditions = vec![];
        let (mut loop_input, _) = space1(loop_input)?;
        for _ in 0..cond_count {
            // Thread the remaining input through the loop, exactly as the
            // propositional effect loop above does. Reading from `input` here
            // would re-parse the condition count as the first condition's
            // variable and leave `effect_var_id` pointing at a condition.
            let (rest, var_id) = usize(loop_input)?;
            let (rest, _) = space1(rest)?;
            let (rest, value) = usize(rest)?;
            let (rest, _) = space1(rest)?;
            conditions.push(ExplicitFact::propositional(var_id, value));
            loop_input = rest;
        }
        let (loop_input, effect_var_id) = usize(loop_input)?;
        let (loop_input, _) = space1(loop_input)?;
        let (loop_input, operation) = parse_assignment_operation(loop_input)?;
        let (loop_input, _) = space1(loop_input)?;
        let (loop_input, effect_value) = usize(loop_input)?;
        let (loop_input, _) = line_ending(loop_input)?;
        let assignment_effect = AssignmentEffect::new(
            effect_var_id,
            operation,
            effect_value,
            is_conditional_effect,
            conditions,
        );
        assignment_effects.push(assignment_effect);
        input = loop_input;
    }
    let (input, cost) = u64(input)?;
    let (input, _) = line_ending(input)?;
    let (input, _) = tag("end_operator")(input)?;
    let (input, _) = line_ending(input)?;

    let operator = SasOperator {
        name: name.to_string(),
        prevail,
        effects,
        assignment_effects,
        cost,
    };

    Ok((input, operator))
}

fn parse_operators(input: &str) -> IResult<&str, Vec<SasOperator>> {
    counted(input, parse_operator)
}

fn parse_axiom(input: &str) -> IResult<&str, PropositionalAxiom> {
    let (input, _) = tag("begin_rule")(input)?;
    let (input, _) = line_ending(input)?;

    let (input, conditions) = counted(input, parse_fact_line)?;
    let (input, var_id) = usize(input)?;
    let (input, _) = tag(" ")(input)?;
    let (input, precondition_value) = usize(input)?;
    let (input, _) = tag(" ")(input)?;
    let (input, effect_value) = usize(input)?;
    let (input, _) = line_ending(input)?;
    let (input, _) = tag("end_rule")(input)?;
    let (input, _) = line_ending(input)?;
    let axiom = PropositionalAxiom::new(conditions, var_id, precondition_value, effect_value);

    Ok((input, axiom))
}

fn parse_axioms(input: &str) -> IResult<&str, Vec<PropositionalAxiom>> {
    counted(input, parse_axiom)
}

fn parse_comparison_operator(input: &str) -> IResult<&str, ComparisonOperator> {
    map_opt(parse_operator_token, ComparisonOperator::from_sas).parse(input)
}

fn parse_comparison_axiom(input: &str) -> IResult<&str, ComparisonAxiom> {
    let (input, affected_var_id) = usize(input)?;
    let (input, _) = space1(input)?;
    let (input, comparison_operator) = parse_comparison_operator(input)?;
    let (input, _) = space1(input)?;
    let (input, left_hand_side) = usize(input)?;
    let (input, _) = space1(input)?;
    let (input, right_hand_side) = usize(input)?;
    let (input, _) = line_ending(input)?;
    Ok((
        input,
        ComparisonAxiom::new(
            affected_var_id,
            left_hand_side,
            right_hand_side,
            comparison_operator,
        ),
    ))
}

fn parse_comparison_axioms(input: &str) -> IResult<&str, Vec<ComparisonAxiom>> {
    let (input, num_comparison_axioms) = terminated(u32, line_ending).parse(input)?;
    delimited(
        pair(tag("begin_comparison_axioms"), line_ending),
        count(parse_comparison_axiom, num_comparison_axioms as usize),
        pair(tag("end_comparison_axioms"), line_ending),
    )
    .parse(input)
}

fn parse_cal_operator(input: &str) -> IResult<&str, CalOperator> {
    map_opt(parse_operator_token, CalOperator::from_sas).parse(input)
}

fn parse_assignment_axiom(input: &str) -> IResult<&str, AssignmentAxiom> {
    let (input, affected_var_id) = usize(input)?;
    let (input, _) = space1(input)?;
    let (input, cal_operator) = parse_cal_operator(input)?;
    let (input, _) = space1(input)?;
    let (input, left_hand_side) = usize(input)?;
    let (input, _) = space1(input)?;
    let (input, right_hand_side) = usize(input)?;
    let (input, _) = line_ending(input)?;
    Ok((
        input,
        AssignmentAxiom::new(
            affected_var_id,
            cal_operator,
            left_hand_side,
            right_hand_side,
        ),
    ))
}

fn parse_assignment_axioms(input: &str) -> IResult<&str, Vec<AssignmentAxiom>> {
    let (input, num_numeric_axioms) = terminated(u32, line_ending).parse(input)?;
    delimited(
        pair(tag("begin_numeric_axioms"), line_ending),
        count(parse_assignment_axiom, num_numeric_axioms as usize),
        pair(tag("end_numeric_axioms"), line_ending),
    )
    .parse(input)
}

fn parse_global_constraint(input: &str) -> IResult<&str, ExplicitFact> {
    let (input, _) = tag("begin_global_constraint")(input)?;
    let (input, _) = line_ending(input)?;
    let (input, constraint_var_id) = usize(input)?;
    let (input, _) = space1(input)?;
    let (input, constraning_value) = usize(input)?;
    let (input, _) = line_ending(input)?;
    let (input, _) = tag("end_global_constraint")(input)?;
    let (input, _) = line_ending(input)?;
    let constraint = ExplicitFact::propositional(constraint_var_id, constraning_value);
    Ok((input, constraint))
}

/// The whole file, section by section, in the shape both ways out of the format
/// go through.
///
/// Split out from [`parse_numeric_sas_output`] so that the writer can be held to
/// being this function's inverse; see `sas_writer`'s round-trip test.
pub(crate) fn parse_sas_parts(input: &str) -> IResult<&str, SasTaskParts> {
    let (input, version) = parse_version(input)?;
    let (input, metric) = parse_metric(input)?;
    let (input, variables) = parse_all_variables(input)?;
    let (input, numeric_variables) = parse_all_numeric_variables(input)?;
    let (input, mutexes) = parse_mutexes(input)?;
    let (input, state) = parse_state(input)?;
    let (input, numeric_state) = parse_numeric_state(input)?;
    let (input, goals) = parse_goal(input)?;
    let (input, operators) = parse_operators(input)?;
    let (input, axioms) = parse_axioms(input)?;
    let (input, comparison_axioms) = parse_comparison_axioms(input)?;
    let (input, assignment_axioms) = parse_assignment_axioms(input)?;
    let (input, global_constraint) = parse_global_constraint(input)?;
    let (input, _) = tag("begin_SG")(input)?;
    let (input, _) = line_ending(input)?;

    let parts = SasTaskParts {
        version,
        metric,
        variables,
        numeric_variables,
        mutexes,
        state,
        numeric_state,
        goals,
        operators,
        axioms,
        comparison_axioms,
        assignment_axioms,
        global_constraint,
    };

    Ok((input, parts))
}

pub fn parse_numeric_sas_output(input: &str) -> IResult<&str, NumericRootTask> {
    let (input, parts) = parse_sas_parts(input)?;
    Ok((input, NumericRootTask::from_sas_parts(parts)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric_conditions::ConditionValue;
    use crate::numeric_task::AbstractNumericTask;

    /// One assignment effect guarded by a single condition: `if var5 == 1 then
    /// var3 += var2`. Threading the input incorrectly through the condition
    /// loop silently yields `var1 += var5` here, which is why this pins every
    /// field rather than just the condition.
    #[test]
    fn conditional_assignment_effect_parses_all_fields() {
        let input = "begin_operator\nmove\n0\n0\n1\n1 5 1 3 + 2\n7\nend_operator\n";

        let (rest, operator) = parse_operator(input).expect("operator parses");

        assert_eq!(rest, "");
        assert_eq!(operator.name, "move");
        assert_eq!(operator.cost, 7);

        let effects = &operator.assignment_effects;
        assert_eq!(effects.len(), 1);
        let effect = &effects[0];
        assert!(effect.is_conditional());
        assert_eq!(
            effect.conditions(),
            &vec![ExplicitFact::propositional(5, 1)]
        );
        assert_eq!(effect.affected_var_id(), 3);
        assert_eq!(effect.operation(), &AssignmentOperation::Plus);
        assert_eq!(effect.var_id(), 2);
    }

    /// Two conditions, to catch an off-by-one in how the loop advances.
    #[test]
    fn multi_condition_assignment_effect_parses_all_conditions() {
        let input = "begin_operator\nmove\n0\n0\n1\n2 5 1 6 0 3 + 2\n7\nend_operator\n";

        let (_, operator) = parse_operator(input).expect("operator parses");

        let effect = &operator.assignment_effects[0];
        assert_eq!(
            effect.conditions(),
            &vec![
                ExplicitFact::propositional(5, 1),
                ExplicitFact::propositional(6, 0)
            ]
        );
        assert_eq!(effect.affected_var_id(), 3);
        assert_eq!(effect.var_id(), 2);
    }

    /// The unconditional case must keep working unchanged.
    #[test]
    fn unconditional_assignment_effect_parses() {
        let input = "begin_operator\nmove\n0\n0\n1\n0 3 + 2\n7\nend_operator\n";

        let (_, operator) = parse_operator(input).expect("operator parses");

        let effect = &operator.assignment_effects[0];
        assert!(!effect.is_conditional());
        assert!(effect.conditions().is_empty());
        assert_eq!(effect.affected_var_id(), 3);
        assert_eq!(effect.var_id(), 2);
    }

    /// A task whose numeric conditions are *interleaved* with its genuine
    /// propositional variables, which is how a SAS file writes them: `cond_ge`
    /// on var1 and `cond_lt` on var3, with `a`, `b`, `c` and the derived `d`
    /// around them. Every place a propositional variable id can hide is used
    /// exactly once, so a site that reads an id from the wrong place shows up
    /// here.
    ///
    /// Numerically: `x = 5`, `three = 3`, so `cond_ge` (`x >= three`) holds and
    /// `cond_lt` (`x < three`) does not. Both condition variables are written in
    /// the legacy three-valued form, `<none of those>` and all, so that building
    /// a task out of this file exercises the narrowing to
    /// [`ConditionValue::DOMAIN_SIZE`].
    const INTERLEAVED_CONDITIONS_SAS: &str = "\
begin_version
4
end_version
begin_metric
< 2
end_metric
6
begin_variable
var0
-1
2
Atom a()
NegatedAtom a()
end_variable
begin_variable
var1
0
3
Atom cond_ge()
NegatedAtom cond_ge()
<none of those>
end_variable
begin_variable
var2
-1
2
Atom b()
NegatedAtom b()
end_variable
begin_variable
var3
0
3
Atom cond_lt()
NegatedAtom cond_lt()
<none of those>
end_variable
begin_variable
var4
-1
2
Atom c()
NegatedAtom c()
end_variable
begin_variable
var5
1
2
Atom d()
NegatedAtom d()
end_variable
3
begin_numeric_variables
R -1 x
C -1 three
I -1 total_cost
end_numeric_variables
1
begin_mutex_group
2
1 0
3 0
end_mutex_group
begin_state
0
2
0
2
0
1
end_state
begin_numeric_state
5
3
0
end_numeric_state
begin_goal
2
0 1
5 0
end_goal
2
begin_operator
raise_x
1
1 0
1
0 4 0 1
1
0 0 + 1
1
end_operator
begin_operator
guarded
0
1
1 3 0 2 -1 1
1
1 1 0 0 + 1
1
end_operator
1
begin_rule
1
1 0
5 1 0
end_rule
2
begin_comparison_axioms
1 >= 0 1
3 < 0 1
end_comparison_axioms
0
begin_numeric_axioms
end_numeric_axioms
begin_global_constraint
5 0
end_global_constraint
begin_SG
";

    /// A parsed task numbers its propositional variables exactly as the file
    /// does, conditions interleaved and all, and every site that names a
    /// variable reads the id the file wrote.
    ///
    /// Only *some* of these sites are covered by a plan cost: nothing in the
    /// search reads a mutex group, so a mutex group parsed against the wrong
    /// ids would leave every benchmark's plan intact and silently mislead the
    /// potential heuristic, which is the one consumer of `are_facts_mutex`.
    #[test]
    fn parsing_a_sas_task_keeps_the_file_s_variable_order() {
        let (rest, task) =
            parse_numeric_sas_output(INTERLEAVED_CONDITIONS_SAS).expect("the fixture parses");
        assert_eq!(rest, "");

        let names: Vec<&str> = (0..task.get_num_variables())
            .map(|var_id| task.get_variable_name(var_id).expect("variable in range"))
            .collect();
        assert_eq!(names, ["var0", "var1", "var2", "var3", "var4", "var5"]);
        let conditions = task.numeric_conditions();
        assert_eq!(conditions.len(), 2);
        assert_eq!(
            conditions
                .iter()
                .map(|condition| condition.prop_var_id())
                .collect::<Vec<_>>(),
            [1, 3]
        );
        for var_id in 0..task.variables().len() {
            assert_eq!(
                conditions.is_condition_var(var_id),
                var_id == 1 || var_id == 3,
                "variable {var_id} is taken for the wrong kind"
            );
        }

        // Each variable keeps its own metadata, except that the file's third
        // condition value is narrowed away: `var1` carries a comparison, so it
        // is two-valued and its `<none of those>` default collapses onto
        // `False`. `var5` is the derived one.
        assert_eq!(
            task.get_variable_domain_size(1),
            Ok(ConditionValue::DOMAIN_SIZE)
        );
        assert_eq!(
            task.get_variable_default_axiom_value(1),
            Ok(ConditionValue::False.as_usize())
        );
        assert_eq!(task.get_variable_axiom_layer(5), Ok(Some(1)));
        assert_eq!(task.get_variable_name(5), Ok("var5"));

        // The initial state is the file's, closed under the axioms: `cond_ge`
        // holds, `cond_lt` does not, and `d` is proven by the rule that reads
        // `cond_ge`.
        assert_eq!(
            task.get_initial_propositional_state_values(),
            // a=0, cond_ge=true, b=0, cond_lt=false, c=0, d=true
            [0, 0, 0, 1, 0, 0]
        );

        // Goals, mutex groups and the global constraint.
        let goals: Vec<ExplicitFact> = (0..task.get_num_goals())
            .map(|goal_id| *task.get_goal_fact(goal_id))
            .collect();
        assert_eq!(
            goals,
            [
                ExplicitFact::propositional(0, 1),
                ExplicitFact::propositional(5, 0),
            ]
        );
        // Read through `are_facts_mutex`, the only consumer there is: the group
        // is `{cond_ge = true, cond_lt = true}`.
        assert!(task.are_facts_mutex(
            &ExplicitFact::condition(1, 0),
            &ExplicitFact::condition(3, 0)
        ));
        assert!(!task.are_facts_mutex(
            &ExplicitFact::propositional(0, 0),
            &ExplicitFact::propositional(2, 0)
        ));
        assert_eq!(task.global_constraint(), &ExplicitFact::propositional(5, 0));

        // Operator preconditions and effects, including the effect condition and
        // the guard of an assignment effect. `raise_x` gains a precondition on
        // its effect variable from the `0` precondition value in `0 4 0 1`.
        let raise_x = &task.get_operators()[0];
        assert_eq!(raise_x.name(), "raise_x");
        assert_eq!(
            raise_x.preconditions(),
            &vec![
                ExplicitFact::condition(1, 0),
                ExplicitFact::propositional(4, 0),
            ]
        );
        assert_eq!(raise_x.effects()[0].var_id(), 4);
        assert_eq!(raise_x.assignment_effects()[0].affected_var_id(), 0);

        let guarded = &task.get_operators()[1];
        assert_eq!(
            guarded.effects()[0].conditions(),
            &vec![ExplicitFact::condition(3, 0)]
        );
        assert_eq!(guarded.effects()[0].var_id(), 2);
        assert_eq!(
            guarded.assignment_effects()[0].conditions(),
            &vec![ExplicitFact::condition(1, 0)]
        );

        // The propositional axiom's head and its condition.
        let rule = &task.axioms()[0];
        assert_eq!(rule.var_id(), 5);
        assert_eq!(rule.conditions(), &vec![ExplicitFact::condition(1, 0)]);

        // The comparison axioms name the variables they write.
        let heads: Vec<usize> = task
            .comparison_axioms()
            .iter()
            .map(|axiom| axiom.get_affected_var_id())
            .collect();
        assert_eq!(heads, [1, 3]);
    }
}
