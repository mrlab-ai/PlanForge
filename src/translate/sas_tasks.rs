use std::path::Path;

use crate::translate::sas::SASTask;
use crate::translate::sas_writer;

pub const SAS_FILE_VERSION: i32 = 4;
pub const DEBUG: bool = false;

pub use crate::translate::sas::{
    CanonicalAssignEffect, CanonicalAssignRhs, CanonicalEffect, CanonicalOperator,
    CanonicalVariable, CompareAxiom, NumericAxiom, NumericPrecond, NumericVariable, SASAxiom,
    SASOperator, Variable,
};

pub fn output(task: &SASTask, path: &Path) -> anyhow::Result<()> {
    sas_writer::write_sas(task, path)
}
