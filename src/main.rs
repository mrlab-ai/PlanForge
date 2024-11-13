mod parser;

use std::fs;
use parser::parse_sas_output;

fn main() {
    let content = fs::read_to_string("output.sas").expect("Could not read file");
    match parse_sas_output(&content) {
        Ok((_, sas_output)) => {},
        Err(e) => println!("Failed to parse file: {:?}", e),
    }
}
