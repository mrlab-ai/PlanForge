//! SAS tasks representation
//! Port of python/translate/sas_tasks.py

use crate::translate::sas::{SASTask};

impl SASTask {
    /// Write task to file in SAS+ format (convenience method)
    pub fn write_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = std::fs::File::create(path)?;
        self.output(&mut file)?;
        Ok(())
    }

    /// Get the number of variables
    pub fn num_variables(&self) -> usize {
        self.variables.ranges.len()
    }

    /// Get the number of operators
    pub fn num_operators(&self) -> usize {
        self.operators.len()
    }
}
