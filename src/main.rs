mod parser;
mod search;

use parser::numeric_parser::parse_numeric_sas_output;
use std::env;
use std::fs;

use crate::search::numeric::numeric_task::AbstractNumericTask;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} [sas_file]", args[0]);
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "No SAS file provided",
        ));
    }
    let sas_file = &args[1];
    let content = fs::read_to_string(sas_file).expect("Could not read file");
    match parse_numeric_sas_output(&content) {
        Ok((_, sas_output)) => {
            println!("Successfully parsed SAS file");
            let task: &dyn AbstractNumericTask = &sas_output;
        }
        Err(e) => println!("Failed to parse file: {:?}", e),
    }
    Ok(())
}
