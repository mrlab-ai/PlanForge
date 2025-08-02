mod parser;
mod search; 

use std::fs;
use std::env;
use parser::classical_parser::parse_sas_output;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} [sas_file]", args[0]);
        return;
    }
    let sas_file = &args[1];
    let content = fs::read_to_string(sas_file).expect("Could not read file");
    match parse_sas_output(&content) {
        Ok((_, sas_output)) => {println!("Successfully parsed SAS file")},
        Err(e) => println!("Failed to parse file: {:?}", e),
    }
}
