use nom::{
    bytes::complete::tag,
    character::complete::{digit1, line_ending, alphanumeric1, i32, u32, not_line_ending},
    combinator::map_res,
    sequence::{delimited, preceded},
    IResult,
};
use std::fs;

// Struct to hold parsed data
#[derive(Debug)]
struct SasOutput {
    version: u32,
    metric: bool,
    variables: Vec<String>,
}

#[derive(Debug)]
struct ExplicitVariable {
    domain_size: u32,
    name: String, 
    fact_names: Vec<String>,
    axiom_layer: i32, 
    axiom_default_value: u32,
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
    let var = ExplicitVariable {
        domain_size,
        name: variable_name.to_string(),
        fact_names,
        axiom_layer: ws,
        axiom_default_value: 0,
    };
    Ok((input, var))
}

fn parse_all_variables(input: &str) -> IResult<&str, Vec<ExplicitVariable>> {
    let mut variables = Vec::new();
    let mut input = input;
    loop {
        let variable_result = parse_variable(input);
        match variable_result {
            Ok((loop_input, var)) => {
                variables.push(var);
                input = loop_input;
            }
            Err(_) => break,
            
        }
    }
    Ok((input, variables))
}


fn parse_sas_output(input: &str) -> IResult<&str, SasOutput> {
    let (input, version) = parse_version(input)?;
    println!("Parsed version: {}", version);
    let (input, metric) = parse_metric(input)?;
    println!("Parsed metric: {}", metric);

    let variables = vec![];
    let (input, var) = parse_all_variables(input)?;
    println!("Parsed variables: {:?}", var);



    Ok((input, SasOutput { version, metric, variables }))
}

fn main() {
    let content = fs::read_to_string("output.sas").expect("Could not read file");
    match parse_sas_output(&content) {
        Ok((_, sas_output)) => println!("Parsed output: {:?}", sas_output),
        Err(e) => println!("Failed to parse file: {:?}", e),
    }
}
