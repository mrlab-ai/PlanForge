//! The text syntax of the SAS+ format.
//!
//! What the sections *mean* lives in [`crate::sas_format`], so that the
//! translator can build the same task without going through text at all.
//!
#[cfg(test)]
mod tests;

use crate::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use crate::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitValueIndex, Metric,
    NumericRootTask, NumericType, NumericValue, NumericVariable, OperatorCost, VariableIndex,
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

    let metric = Metric::from_sas(
        direction,
        if index > 0 {
            Some(VariableIndex::from_usize(index))
        } else {
            None
        },
    )
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
fn parse_state(input: &str) -> IResult<&str, Vec<ExplicitValueIndex>> {
    delimited(
        pair(tag("begin_state"), line_ending),
        many0(terminated(usize, line_ending)),
        pair(tag("end_state"), line_ending),
    )
    .parse(input)
    .map(|(i, v)| {
        (
            i,
            v.into_iter().map(ExplicitValueIndex::from_usize).collect(),
        )
    })
}

fn parse_numeric_state(input: &str) -> IResult<&str, Vec<NumericValue>> {
    delimited(
        pair(tag("begin_numeric_state"), line_ending),
        many0(terminated(double, line_ending)),
        pair(tag("end_numeric_state"), line_ending),
    )
    .parse(input)
    .map(|(i, v)| (i, v.into_iter().map(NumericValue::new).collect()))
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

        let (loop_input, effect_var_id) = u32(loop_input)?;
        let (loop_input, _) = space1(loop_input)?;
        let (loop_input, precondition_field) = i32(loop_input)?;
        let (loop_input, _) = space1(loop_input)?;
        let (loop_input, effect_value) = u32(loop_input)?;

        let effect = Effect::new(
            effect_conditions,
            VariableIndex::new(effect_var_id),
            effect_precondition_from_sas(precondition_field),
            ExplicitValueIndex::new(effect_value),
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
        let (loop_input, effect_var_id) = u32(loop_input)?;
        let (loop_input, _) = space1(loop_input)?;
        let (loop_input, operation) = parse_assignment_operation(loop_input)?;
        let (loop_input, _) = space1(loop_input)?;
        let (loop_input, effect_value) = u32(loop_input)?;
        let (loop_input, _) = line_ending(loop_input)?;
        let assignment_effect = AssignmentEffect::new(
            VariableIndex::new(effect_var_id),
            operation,
            VariableIndex::new(effect_value),
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
        cost: OperatorCost::new(cost),
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
    let (input, var_id) = u32(input)?;
    let (input, _) = tag(" ")(input)?;
    let (input, precondition_value) = u32(input)?;
    let (input, _) = tag(" ")(input)?;
    let (input, effect_value) = u32(input)?;
    let (input, _) = line_ending(input)?;
    let (input, _) = tag("end_rule")(input)?;
    let (input, _) = line_ending(input)?;
    let axiom = PropositionalAxiom::new(
        conditions,
        VariableIndex::new(var_id),
        ExplicitValueIndex::new(precondition_value),
        ExplicitValueIndex::new(effect_value),
    );

    Ok((input, axiom))
}

fn parse_axioms(input: &str) -> IResult<&str, Vec<PropositionalAxiom>> {
    counted(input, parse_axiom)
}

fn parse_comparison_operator(input: &str) -> IResult<&str, ComparisonOperator> {
    map_opt(parse_operator_token, ComparisonOperator::from_sas).parse(input)
}

fn parse_comparison_axiom(input: &str) -> IResult<&str, ComparisonAxiom> {
    let (input, affected_var_id) = u32(input)?;
    let (input, _) = space1(input)?;
    let (input, comparison_operator) = parse_comparison_operator(input)?;
    let (input, _) = space1(input)?;
    let (input, left_hand_side) = u32(input)?;
    let (input, _) = space1(input)?;
    let (input, right_hand_side) = u32(input)?;
    let (input, _) = line_ending(input)?;
    Ok((
        input,
        ComparisonAxiom::new(
            VariableIndex::new(affected_var_id),
            VariableIndex::new(left_hand_side),
            VariableIndex::new(right_hand_side),
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
    let (input, affected_var_id) = u32(input)?;
    let (input, _) = space1(input)?;
    let (input, cal_operator) = parse_cal_operator(input)?;
    let (input, _) = space1(input)?;
    let (input, left_hand_side) = u32(input)?;
    let (input, _) = space1(input)?;
    let (input, right_hand_side) = u32(input)?;
    let (input, _) = line_ending(input)?;
    Ok((
        input,
        AssignmentAxiom::new(
            VariableIndex::new(affected_var_id),
            cal_operator,
            VariableIndex::new(left_hand_side),
            VariableIndex::new(right_hand_side),
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
