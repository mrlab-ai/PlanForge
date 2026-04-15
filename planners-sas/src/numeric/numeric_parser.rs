use crate::numeric::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use crate::numeric::numeric_task::{
    AssignmentEffect, AssignmentOperation, Metric, NumericType, NumericVariable,
};
use crate::numeric::numeric_task::{
    Effect, ExplicitFact, ExplicitVariable, NumericRootTask, Operator,
};
use nom::Parser;
use nom::bytes::complete::take_while1;
use nom::combinator::{map, opt, recognize};
use nom::number::complete::double;
use nom::{
    IResult,
    branch::alt,
    bytes::complete::tag,
    character::complete::{
        alphanumeric1, char, digit1, i32, line_ending, not_line_ending, space1, u32, u64, usize,
    },
    combinator::map_res,
    sequence::separated_pair,
};
use std::vec;

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
    let (input, min_or_max) = alt((char('<'), char('>'))).parse(input)?;
    let is_min = min_or_max == '<';
    let (input, _) = space1(input)?;
    let (input, metric) = usize(input)?;
    let (input, _) = line_ending(input)?;
    let (input, _) = tag("end_metric")(input)?;
    let (input, _) = line_ending(input)?;

    let metric = Metric::new(is_min, if metric > 0 { Some(metric) } else { None });
    Ok((input, metric))
}

fn parse_variable(input: &str) -> IResult<&str, ExplicitVariable> {
    let (input, _) = tag("begin_variable")(input)?;
    let (input, _) = line_ending(input)?;
    let (input, variable_name) = alphanumeric1(input)?;
    let (input, _) = line_ending(input)?;
    let (input, axiom_layer) = i32(input)?;
    let (input, _) = line_ending(input)?;
    let (input, domain_size) = usize(input)?;
    let (input, _) = line_ending(input)?;

    let mut fact_names = Vec::with_capacity(domain_size);
    let mut input = input;
    for _ in 0..domain_size {
        let (loop_input, fact_name) = not_line_ending(input)?;
        fact_names.push(fact_name.to_string());
        let (loop_input, _) = line_ending(loop_input)?;
        input = loop_input;
    }
    let (input, _) = tag("end_variable")(input)?;
    let (input, _) = line_ending(input)?;
    let var = ExplicitVariable::new(
        domain_size,
        variable_name.to_string(),
        fact_names,
        if axiom_layer >= 0 {
            Some(axiom_layer as usize)
        } else {
            None
        },
        0,
    );
    Ok((input, var))
}

fn parse_all_variables(input: &str) -> IResult<&str, Vec<ExplicitVariable>> {
    let (input, num_variables) = u32(input)?;
    let (input, _) = line_ending(input)?;
    let mut variables = Vec::new();
    let mut input = input;
    for _ in 0..num_variables {
        let (loop_input, var) = parse_variable(input)?;
        variables.push(var);
        input = loop_input;
    }
    Ok((input, variables))
}

fn parse_line_type(input: &str) -> IResult<&str, NumericType> {
    alt((
        map(tag("C"), |_| NumericType::Constant),
        map(tag("D"), |_| NumericType::Derived),
        map(tag("I"), |_| NumericType::Cost),
        map(tag("R"), |_| NumericType::Regular),
    ))
    .parse(input)
}

fn parse_plus_or_minus(input: &str) -> IResult<&str, AssignmentOperation> {
    alt((
        map(tag("="), |_| AssignmentOperation::Assign),
        map(tag("+"), |_| AssignmentOperation::Plus),
        map(tag("-"), |_| AssignmentOperation::Minus),
        map(tag("*"), |_| AssignmentOperation::Times),
        map(tag("/"), |_| AssignmentOperation::Divide),
    ))
    .parse(input)
}

fn parse_layer(input: &str) -> IResult<&str, i32> {
    map(
        // We recognize an optional minus sign followed by one or more digits.
        recognize(nom::sequence::pair(opt(char('-')), digit1)),
        |s: &str| s.parse::<i32>().unwrap(),
    )
    .parse(input)
}

fn parse_name(input: &str) -> IResult<&str, String> {
    // `take_while1`` takes all characters until a newline or end of input.
    let (input, name) = take_while1(|c: char| c != '\n')(input)?;
    Ok((input, name.trim().to_string()))
}

fn parse_numeric_variable(input: &str) -> IResult<&str, NumericVariable> {
    let (input, numeric_type) = parse_line_type(input)?;
    let (input, _) = space1(input)?;
    let (input, layer) = parse_layer(input)?;
    let (input, _) = space1(input)?;
    let (input, variable_name) = parse_name(input)?;
    let (input, _) = line_ending(input)?;
    let var = NumericVariable::new(
        variable_name.to_string(),
        numeric_type,
        if layer >= 0 {
            Some(layer as usize)
        } else {
            None
        },
    );
    Ok((input, var))
}

fn parse_all_numeric_variables(input: &str) -> IResult<&str, Vec<NumericVariable>> {
    let (input, num_numeric_variables) = u32(input)?;
    let (input, _) = line_ending(input)?;
    let (input, _) = tag("begin_numeric_variables")(input)?;
    let (input, _) = line_ending(input)?;
    let mut numeric_variables = Vec::new();
    let mut input = input;
    for _ in 0..num_numeric_variables {
        let (loop_input, var) = parse_numeric_variable(input)?;
        numeric_variables.push(var);
        input = loop_input;
    }
    let (input, _) = tag("end_numeric_variables")(input)?;
    let (input, _) = line_ending(input)?;
    Ok((input, numeric_variables))
}

fn parse_integer(input: &str) -> IResult<&str, u32> {
    map_res(digit1, str::parse::<u32>).parse(input)
}

fn parse_mutex_group(input: &str) -> IResult<&str, Vec<ExplicitFact>> {
    let (input, _) = tag("begin_mutex_group")(input)?;
    let (input, _) = line_ending(input)?;

    let (input, num_facts) = u32(input)?;
    let (input, _) = line_ending(input)?;
    let mut input = input;

    let mut mutex_group = Vec::with_capacity(num_facts as usize);
    for _ in 0..num_facts {
        let mut parser = separated_pair(parse_integer, tag(" "), parse_integer);
        let (new_input, fact) = parser.parse(input)?;
        let fact = ExplicitFact::new(fact.0 as usize, fact.1 as usize);

        mutex_group.push(fact);
        let (new_input, _) = line_ending(new_input)?;
        input = new_input;
    }

    let (input, _) = tag("end_mutex_group")(input)?;
    let (input, _) = line_ending(input)?;

    Ok((input, mutex_group))
}

fn parse_mutexes(input: &str) -> IResult<&str, Vec<Vec<ExplicitFact>>> {
    let (input, num_mutexes) = usize(input)?;
    let (input, _) = line_ending(input)?;
    let mut input = input;

    let mut mutexes = Vec::with_capacity(num_mutexes);

    for _ in 0..num_mutexes {
        let (new_input, mutex_group) = parse_mutex_group(input)?;
        mutexes.push(mutex_group);
        input = new_input;
    }
    Ok((input, mutexes))
}

fn parse_state(input: &str) -> IResult<&str, Vec<usize>> {
    let (input, _) = tag("begin_state")(input)?;
    let (input, _) = line_ending(input)?;
    let mut input = input;
    let mut states = vec![];
    loop {
        let (loop_input, state) = not_line_ending(input)?;
        if state == "end_state" {
            input = loop_input;
            break;
        }
        let (_, state) = usize(state)?;
        states.push(state);
        let (loop_input, _) = line_ending(loop_input)?;
        input = loop_input;
    }
    let (input, _) = line_ending(input)?;
    Ok((input, states))
}

fn parse_numeric_state(input: &str) -> IResult<&str, Vec<f64>> {
    let (input, _) = tag("begin_numeric_state")(input)?;
    let (input, _) = line_ending(input)?;
    let mut input = input;
    let mut states = vec![];
    loop {
        let (loop_input, state) = not_line_ending(input)?;
        if state == "end_numeric_state" {
            input = loop_input;
            break;
        }
        let (_, state) = double(state)?;
        states.push(state);
        let (loop_input, _) = line_ending(loop_input)?;
        input = loop_input;
    }
    let (input, _) = line_ending(input)?;
    Ok((input, states))
}

fn parse_goal(input: &str) -> IResult<&str, Vec<ExplicitFact>> {
    let (input, _) = tag("begin_goal")(input)?;
    let (input, _) = line_ending(input)?;
    let (input, num_goals) = u32(input)?;
    let (input, _) = line_ending(input)?;

    let mut input = input;
    let mut goals = vec![];
    for _ in 0..num_goals {
        let mut parser = separated_pair(parse_integer, tag(" "), parse_integer);
        let (loop_input, goal) = parser.parse(input)?;
        let goal = ExplicitFact::new(goal.0 as usize, goal.1 as usize);
        goals.push(goal);
        let (loop_input, _) = line_ending(loop_input)?;
        input = loop_input;
    }

    let (input, _) = tag("end_goal")(input)?;
    let (input, _) = line_ending(input)?;
    Ok((input, goals))
}

fn parse_operator(input: &str) -> IResult<&str, Operator> {
    let (input, _) = tag("begin_operator")(input)?;
    let (input, _) = line_ending(input)?;
    let (input, name) = not_line_ending(input)?;
    let (input, _) = line_ending(input)?;
    let (input, num_prevail_cond) = u32(input)?;
    let (input, _) = line_ending(input)?;
    let mut preconditions = vec![];
    let mut input = input;
    for _ in 0..num_prevail_cond {
        let mut parser = separated_pair(parse_integer, space1, parse_integer);
        let (loop_input, prevail_cond) = parser.parse(input)?;
        let prevail_cond = ExplicitFact::new(prevail_cond.0 as usize, prevail_cond.1 as usize);
        preconditions.push(prevail_cond);
        let (loop_input, _) = line_ending(loop_input)?;
        input = loop_input;
    }

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
            let condition = ExplicitFact::new(condition.0 as usize, condition.1 as usize);
            effect_conditions.push(condition);
            let (loop_input2, _) = space1(loop_input2)?;
            loop_input = loop_input2;
        }

        let (loop_input, effect_var_id) = usize(loop_input)?;
        let (loop_input, _) = space1(loop_input)?;
        let (loop_input, precondition_value) = i32(loop_input)?; // NOTE: -1 if there is no precondition.
        let (loop_input, _) = space1(loop_input)?;
        let (loop_input, effect_value) = usize(loop_input)?;

        if precondition_value != -1 {
            let precondition = ExplicitFact::new(effect_var_id, precondition_value as usize);
            preconditions.push(precondition);
        }

        let effect = Effect::new(
            effect_conditions,
            effect_var_id,
            if precondition_value >= 0 {
                Some(precondition_value as usize)
            } else {
                None
            },
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
        let (loop_input, _) = space1(loop_input)?;
        for _ in 0..cond_count {
            let (loop_input, var_id) = usize(input)?;
            let (loop_input, _) = space1(loop_input)?;
            let (loop_input, value) = usize(loop_input)?;
            let (loop_input, _) = space1(loop_input)?;
            input = loop_input;
            let condition = ExplicitFact::new(var_id, value);
            conditions.push(condition);
        }
        let (loop_input, effect_var_id) = usize(loop_input)?;
        let (loop_input, _) = space1(loop_input)?;
        let (loop_input, operation) = parse_plus_or_minus(loop_input)?;
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

    let operator = Operator::new(
        name.to_string(),
        preconditions,
        effects,
        assignment_effects,
        cost,
    );

    Ok((input, operator))
}

fn parse_operators(input: &str) -> IResult<&str, Vec<Operator>> {
    let (input, num_operators) = u32(input)?;
    let (input, _) = line_ending(input)?;
    let mut input = input;
    let mut operators = vec![];
    for _ in 0..num_operators {
        let (loop_input, operator) = parse_operator(input)?;
        operators.push(operator);
        input = loop_input;
    }
    Ok((input, operators))
}

fn parse_axiom(input: &str) -> IResult<&str, PropositionalAxiom> {
    let (input, _) = tag("begin_rule")(input)?;
    let (input, _) = line_ending(input)?;

    let (input, num_conditions) = u32(input)?;
    let (input, _) = line_ending(input)?;

    let mut input = input;
    let mut conditions = vec![];
    for _ in 0..num_conditions {
        let mut parser = separated_pair(parse_integer, tag(" "), parse_integer);
        let (loop_input, condition) = parser.parse(input)?;
        let condition = ExplicitFact::new(condition.0 as usize, condition.1 as usize);
        conditions.push(condition);
        let (loop_input, _) = line_ending(loop_input)?;
        input = loop_input;
    }
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
    let (input, num_axioms) = u32(input)?;
    let (input, _) = line_ending(input)?;
    let mut input = input;
    let mut axioms = vec![];
    for _ in 0..num_axioms {
        let (loop_input, axiom) = parse_axiom(input)?;
        axioms.push(axiom);
        input = loop_input;
    }
    Ok((input, axioms))
}

fn parse_comparison_operator(input: &str) -> IResult<&str, ComparisonOperator> {
    alt((
        map(tag("<="), |_| ComparisonOperator::LessThanOrEqual),
        map(tag(">="), |_| ComparisonOperator::GreaterThanOrEqual),
        map(tag("!="), |_| ComparisonOperator::UnEqual),
        map(tag(">"), |_| ComparisonOperator::GreaterThan),
        map(tag("<"), |_| ComparisonOperator::LessThan),
        map(tag("="), |_| ComparisonOperator::Equal),
    ))
    .parse(input)
}

fn parse_comparison_axiom(input: &str) -> IResult<&str, ComparisonAxiom> {
    // This function is a placeholder for parsing comparison axioms.
    // Currently, it returns an empty Axiom as no comparison axioms are defined.
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
    // This function is a placeholder for parsing comparison axioms.
    // Currently, it returns an empty vector as no comparison axioms are defined.
    let (input, num_comparison_axioms) = u32(input)?;
    let (input, _) = line_ending(input)?;
    let (input, _) = tag("begin_comparison_axioms")(input)?;
    let (mut input, _) = line_ending(input)?;

    let mut comparison_axioms = vec![];
    for _ in 0..num_comparison_axioms {
        let (loop_input, comparison_axiom) = parse_comparison_axiom(input)?;
        comparison_axioms.push(comparison_axiom);
        input = loop_input;
    }
    let (input, _) = tag("end_comparison_axioms")(input)?;
    let (input, _) = line_ending(input)?;
    Ok((input, comparison_axioms))
}

fn parse_cal_operator(input: &str) -> IResult<&str, CalOperator> {
    alt((
        map(tag("+"), |_| CalOperator::Sum),
        map(tag("-"), |_| CalOperator::Difference),
        map(tag("*"), |_| CalOperator::Product),
        map(tag("/"), |_| CalOperator::Division),
    ))
    .parse(input)
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
    // This function is a placeholder for parsing numeric axioms.
    // Currently, it returns an empty vector as no numeric axioms are defined.
    let (input, num_numeric_axioms) = u32(input)?;
    let (input, _) = line_ending(input)?;
    let (input, _) = tag("begin_numeric_axioms")(input)?;
    let (mut input, _) = line_ending(input)?;

    let mut assignment_axioms = vec![];
    for _ in 0..num_numeric_axioms {
        let (loop_input, axiom) = parse_assignment_axiom(input)?;
        assignment_axioms.push(axiom);
        input = loop_input;
    }
    let (input, _) = tag("end_numeric_axioms")(input)?;
    let (input, _) = line_ending(input)?;
    Ok((input, assignment_axioms))
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
    let constraint = ExplicitFact {
        var: constraint_var_id,
        value: constraning_value,
    };
    Ok((input, constraint))
}

pub fn parse_numeric_sas_output(input: &str) -> IResult<&str, NumericRootTask> {
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

    let output = NumericRootTask::new(
        version,
        metric,
        variables,
        numeric_variables,
        goals,
        mutexes,
        state,
        numeric_state,
        operators,
        axioms,
        comparison_axioms,
        assignment_axioms,
        global_constraint,
    );

    Ok((input, output))
}
