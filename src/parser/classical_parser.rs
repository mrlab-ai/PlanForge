use crate::search::classical::classical_task::{
    Axiom, Effect, ExplicitVariable, Fact, Operator, RootTask,
};
use nom::{
    bytes::complete::tag,
    character::complete::{alphanumeric1, digit1, i32, line_ending, not_line_ending, u32},
    combinator::map_res,
    sequence::separated_pair,
    IResult,
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

fn parse_metric(input: &str) -> IResult<&str, bool> {
    let (input, _) = tag("begin_metric")(input)?;
    let (input, _) = line_ending(input)?;
    let (input, metric) = u32(input)?;
    let (input, _) = line_ending(input)?;
    let (input, _) = tag("end_metric")(input)?;
    let (input, _) = line_ending(input)?;
    let metric = metric != 0;
    Ok((input, metric))
}

fn parse_variable(input: &str) -> IResult<&str, ExplicitVariable> {
    let (input, _) = tag("begin_variable")(input)?;
    let (input, _) = line_ending(input)?;
    let (input, variable_name) = alphanumeric1(input)?;
    let (input, _) = line_ending(input)?;
    let (input, ws) = i32(input)?;
    let (input, _) = line_ending(input)?;
    let (input, domain_size) = u32(input)?;
    let (input, _) = line_ending(input)?;

    let mut fact_names = Vec::with_capacity(domain_size as usize);
    let mut input = input;
    for _ in 0..domain_size {
        let (loop_input, fact_name) = not_line_ending(input)?;
        fact_names.push(fact_name.to_string());
        let (loop_input, _) = line_ending(loop_input)?;
        input = loop_input;
    }
    let (input, _) = tag("end_variable")(input)?;
    let (input, _) = line_ending(input)?;
    let var = ExplicitVariable::new(domain_size, variable_name.to_string(), fact_names, ws, 0);
    Ok((input, var))
}

fn parse_all_variables(input: &str) -> IResult<&str, Vec<ExplicitVariable>> {
    let (input, num_variables) = u32(input)?;
    let (input, _) = line_ending(input)?;
    println!("Number of variables: {}", num_variables);
    let mut variables = Vec::new();
    let mut input = input;
    for _ in 0..num_variables {
        let (loop_input, var) = parse_variable(input)?;
        variables.push(var);
        input = loop_input;
    }
    Ok((input, variables))
}

fn parse_integer(input: &str) -> IResult<&str, u32> {
    map_res(digit1, str::parse::<u32>)(input)
}

fn parse_mutex_group(input: &str) -> IResult<&str, Vec<Fact>> {
    let (input, _) = tag("begin_mutex_group")(input)?;
    let (input, _) = line_ending(input)?;

    let (input, num_facts) = u32(input)?;
    let (input, _) = line_ending(input)?;
    let mut input = input;

    let mut mutex_group = Vec::with_capacity(num_facts as usize);
    for _ in 0..num_facts {
        let mut parser = separated_pair(parse_integer, tag(" "), parse_integer);
        let (new_input, fact) = parser(input)?;
        let fact = Fact::new(fact.0, fact.1);

        mutex_group.push(fact);
        let (new_input, _) = line_ending(new_input)?;
        input = new_input;
    }

    let (input, _) = tag("end_mutex_group")(input)?;
    let (input, _) = line_ending(input)?;

    Ok((input, mutex_group))
}

fn parse_mutexes(input: &str) -> IResult<&str, Vec<Vec<Fact>>> {
    let (input, num_mutexes) = u32(input)?;
    let (input, _) = line_ending(input)?;
    let mut input = input;

    let mut mutexes = Vec::with_capacity(num_mutexes as usize);

    for _ in 0..num_mutexes {
        let (new_input, mutex_group) = parse_mutex_group(input)?;
        mutexes.push(mutex_group);
        input = new_input;
    }
    Ok((input, mutexes))
}

fn parse_state(input: &str) -> IResult<&str, Vec<i32>> {
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
        let (_, state) = i32(state)?;
        states.push(state);
        let (loop_input, _) = line_ending(loop_input)?;
        input = loop_input;
    }
    let (input, _) = line_ending(input)?;
    Ok((input, states))
}

fn parse_goal(input: &str) -> IResult<&str, Vec<Fact>> {
    let (input, _) = tag("begin_goal")(input)?;
    let (input, _) = line_ending(input)?;
    let (input, num_goals) = u32(input)?;
    let (input, _) = line_ending(input)?;

    let mut input = input;
    let mut goals = vec![];
    for _ in 0..num_goals {
        let mut parser = separated_pair(parse_integer, tag(" "), parse_integer);
        let (loop_input, goal) = parser(input)?;
        let goal = Fact::new(goal.0, goal.1);
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
    let mut prevail_conditions = vec![];
    let mut input = input;
    for _ in 0..num_prevail_cond {
        let mut parser = separated_pair(parse_integer, tag(" "), parse_integer);
        let (loop_input, prevail_cond) = parser(input)?;

        let prevail_cond = Fact::new(prevail_cond.0, prevail_cond.1);

        prevail_conditions.push(prevail_cond);
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
            let mut parser = separated_pair(parse_integer, tag(" "), parse_integer);
            let (loop_input2, condition) = parser(loop_input)?;
            let condition = Fact::new(condition.0, condition.1);
            effect_conditions.push(condition);
            let (loop_input2, _) = tag(" ")(loop_input2)?;
            loop_input = loop_input2;
        }
        let (loop_input, effect_var_id) = u32(loop_input)?;
        let (loop_input, _) = tag(" ")(loop_input)?;
        let (loop_input, precondition_value) = i32(loop_input)?; // NOTE: -1 if there is no precondition
        let (loop_input, _) = tag(" ")(loop_input)?;
        let (loop_input, effect_value) = u32(loop_input)?;

        let effect = Effect::new(
            effect_conditions,
            effect_var_id,
            precondition_value,
            effect_value,
        );
        effects.push(effect);
        let (loop_input, _) = line_ending(loop_input)?;
        input = loop_input;
    }
    let (input, cost) = u32(input)?;
    let (input, _) = line_ending(input)?;
    let (input, _) = tag("end_operator")(input)?;
    let (input, _) = line_ending(input)?;

    let operator = Operator::new(name.to_string(), effects, cost);

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

fn parse_axiom(input: &str) -> IResult<&str, Axiom> {
    let (input, _) = tag("begin_rule")(input)?;
    let (input, _) = line_ending(input)?;

    let (input, num_conditions) = u32(input)?;
    let (input, _) = line_ending(input)?;

    let mut input = input;
    let mut conditions = vec![];
    for _ in 0..num_conditions {
        let mut parser = separated_pair(parse_integer, tag(" "), parse_integer);
        let (loop_input, condition) = parser(input)?;
        let condition = Fact::new(condition.0, condition.1);
        conditions.push(condition);
        let (loop_input, _) = line_ending(loop_input)?;
        input = loop_input;
    }
    let (input, var_id) = u32(input)?;
    let (input, _) = tag(" ")(input)?;
    let (input, precondition_value) = u32(input)?;
    let (input, _) = tag(" ")(input)?;
    let (input, effect_value) = u32(input)?;
    let (input, _) = line_ending(input)?;
    let (input, _) = tag("end_rule")(input)?;
    let (input, _) = line_ending(input)?;
    let axiom = Axiom::new(conditions, var_id, precondition_value, effect_value);

    Ok((input, axiom))
}

fn parse_axioms(input: &str) -> IResult<&str, Vec<Axiom>> {
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

pub fn parse_sas_output(input: &str) -> IResult<&str, RootTask> {
    let (input, version) = parse_version(input)?;
    println!("Parsed version: {}", version);
    let (input, metric) = parse_metric(input)?;
    println!("Parsed metric: {}", metric);

    let (input, variables) = parse_all_variables(input)?;

    let (input, mutexes) = parse_mutexes(input)?;
    let (input, states) = parse_state(input)?;
    println!("Parsed states: {:?}", states);

    let (input, goals) = parse_goal(input)?;
    println!("Parsed goals: {:?}", goals);

    let (input, operators) = parse_operators(input)?;

    let (input, axioms) = parse_axioms(input)?;

    let output = RootTask::new(
        version, metric, variables, goals, mutexes, states, operators, axioms,
    );

    Ok((input, output))
}
