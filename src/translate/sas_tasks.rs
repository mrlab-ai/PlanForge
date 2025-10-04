//! SAS tasks representation
//! Port of python/translate/sas_tasks.py

use crate::translate::sas::{SASTask, Variable, SASOperator};
use std::io::Write;

impl SASTask {
    /// Create a new empty SAS task
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a variable to the task
    pub fn add_variable(&mut self, variable: Variable) {
        self.variables.push(variable);
    }

    /// Add an operator to the task
    pub fn add_operator(&mut self, operator: SASOperator) {
        self.operators.push(operator);
    }

    /// Get the number of variables
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    /// Get the number of operators
    pub fn num_operators(&self) -> usize {
        self.operators.len()
    }

    /// Dump task information to stdout
    pub fn dump(&self) {
        println!("SAS Task:");
        println!("  Variables: {}", self.num_variables());
        println!("  Operators: {}", self.num_operators());
        println!("  Numeric Variables: {}", self.numeric_variables.len());
        println!("  Numeric Axioms: {}", self.numeric_axioms.len());
    }

    /// Write task to file in SAS+ format
    pub fn write_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = std::fs::File::create(path)?;
        
        // Write header
        writeln!(file, "begin_version")?;
        writeln!(file, "3")?;
        writeln!(file, "end_version")?;
        
        // Write metric
        writeln!(file, "begin_metric")?;
        writeln!(file, "0")?; // TODO: Handle actual metric
        writeln!(file, "end_metric")?;
        
        // Write variables
        writeln!(file, "{}", self.variables.len())?;
        for (i, var) in self.variables.iter().enumerate() {
            writeln!(file, "begin_variable")?;
            writeln!(file, "var{}", i)?;
            writeln!(file, "-1")?; // Axiom layer
            writeln!(file, "{}", var.value_names.len())?;
            for value in &var.value_names {
                writeln!(file, "{}", value)?;
            }
            writeln!(file, "end_variable")?;
        }
        
        // TODO: Write mutex groups, initial state, goal, operators
        
        Ok(())
    }
}
