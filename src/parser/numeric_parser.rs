use crate::search::numeric_task::{
    AssignmentAxiom,
    AssignmentEffect,
    CalOperator,
    ComparisonAxiom,
    ComparisonOperator,
    NumericType,
    NumericVariable,
    PlusMinus,
};
use crate::search::numeric_task::{
    Axiom,
    Effect,
    ExplicitVariable,
    Fact,
    NumericRootTask,
    Operator,
};
use nom::bytes::complete::take_while1;
use nom::combinator::{ map, opt, recognize };
use nom::number::complete::double;
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{
        alphanumeric1,
        char,
        digit1,
        i32,
        line_ending,
        not_line_ending,
        space1,
        u32,
    },
    combinator::map_res,
    sequence::separated_pair,
    IResult,
};
use std::{ process::exit, vec };

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
    let (input, min_or_max) = alt((char('<'), char('>')))(input)?;
    let (input, _) = space1(input)?;
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

fn parse_line_type(input: &str) -> IResult<&str, NumericType> {
    alt((
        map(tag("C"), |_| NumericType::Constant),
        map(tag("D"), |_| NumericType::Derived),
        map(tag("I"), |_| NumericType::Implicit),
        map(tag("R"), |_| NumericType::Root),
    ))(input)
}

fn parse_plus_or_minus(input: &str) -> IResult<&str, PlusMinus> {
    alt((map(tag("+"), |_| PlusMinus::Plus), map(tag("-"), |_| PlusMinus::Minus)))(input)
}

fn parse_layer(input: &str) -> IResult<&str, i32> {
    map(
        // We recognize an optional minus sign followed by one or more digits.
        recognize(nom::sequence::pair(opt(char('-')), digit1)),
        |s: &str| s.parse::<i32>().unwrap()
    )(input)
}

fn parse_name(input: &str) -> IResult<&str, String> {
    // take_while1 takes all characters until a newline or end of input
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
    let var = NumericVariable::new(variable_name.to_string(), numeric_type, layer);
    Ok((input, var))
}

fn parse_all_numeric_variables(input: &str) -> IResult<&str, Vec<NumericVariable>> {
    let (input, num_numeric_variables) = u32(input)?;
    println!("Number of variables: {}", num_numeric_variables);
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
    println!("Number of effects: {}", num_effects);
    for _ in 0..num_effects {
        let (loop_input, num_conditions) = u32(input)?;
        let (loop_input, _) = tag(" ")(loop_input)?;
        let mut effect_conditions = vec![];
        println!("Number of conditions: {}", num_conditions);
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
            effect_value
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
        let (loop_input, _) = space1(loop_input)?;
        //TODO: Add conditional counts. For now we ignore them.
        if cond_count > 0 {
            panic!("Conditional effects are not supported yet.");
        }
        let (loop_input, effect_var_id) = u32(loop_input)?;

        let (loop_input, _) = space1(loop_input)?;

        let (loop_input, plus_or_minus) = parse_plus_or_minus(loop_input)?;

        let (loop_input, _) = space1(loop_input)?;

        let (loop_input, effect_value) = u32(loop_input)?;
        let (loop_input, _) = line_ending(loop_input)?;
        let assignment_effect = AssignmentEffect::new(effect_var_id, plus_or_minus, effect_value);
        assignment_effects.push(assignment_effect);
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

fn parse_comparison_operator(input: &str) -> IResult<&str, ComparisonOperator> {
    alt((
        map(tag(">="), |_| ComparisonOperator::GreaterThanOrEqual),
        map(tag("<="), |_| ComparisonOperator::LessThanOrEqual),
        map(tag("!="), |_| ComparisonOperator::UnEqual),
        map(tag(">"), |_| ComparisonOperator::GreaterThan),
        map(tag("<"), |_| ComparisonOperator::LessThan),
        map(tag("="), |_| ComparisonOperator::Equal),
    ))(input)
}

fn parse_comparison_axiom(input: &str) -> IResult<&str, ComparisonAxiom> {
    // This function is a placeholder for parsing comparison axioms.
    // Currently, it returns an empty Axiom as no comparison axioms are defined.
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
        ComparisonAxiom::new(affected_var_id, comparison_operator, left_hand_side, right_hand_side),
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
    ))(input)
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
        AssignmentAxiom::new(affected_var_id, cal_operator, left_hand_side, right_hand_side),
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

pub fn parse_numeric_sas_output(input: &str) -> IResult<&str, NumericRootTask> {
    let (input, version) = parse_version(input)?;
    println!("Parsed version: {}", version);
    let (input, metric) = parse_metric(input)?;
    println!("Parsed metric: {}", metric);

    let (input, variables) = parse_all_variables(input)?;
    let (input, numeric_variables) = parse_all_numeric_variables(input)?;

    let (input, mutexes) = parse_mutexes(input)?;
    let (input, state) = parse_state(input)?;
    println!("Parsed initial propositional states: {:?}", state);
    let (input, numeric_states) = parse_numeric_state(input)?;
    println!("Parsed initial numeric state: {:?}", numeric_states);

    let (input, goals) = parse_goal(input)?;
    println!("Parsed goals: {:?}", goals);

    let (input, operators) = parse_operators(input)?;

    let (input, axioms) = parse_axioms(input)?;
    let (input, comparison_axioms) = parse_comparison_axioms(input)?;
    let (input, assignment_axions) = parse_assignment_axioms(input)?;
    println!("Parsed axioms: {:?}", axioms);
    println!("Parsed comparison axioms: {:?}", comparison_axioms);

    let output = NumericRootTask::new(
        version,
        metric,
        variables,
        numeric_variables,
        goals,
        mutexes,
        state,
        numeric_states,
        operators,
        axioms,
        comparison_axioms,
        assignment_axions
    );

    Ok((input, output))
}
