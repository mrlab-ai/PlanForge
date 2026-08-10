use std::io::Write;

use tracing::debug;

use super::fact::ExplicitFact;
use super::variable::ExplicitVariable;
use crate::sas_tasks::SASMutexGroup;

#[derive(Debug, Clone)]
pub struct MutexGroup {
    facts: Vec<ExplicitFact>,
}

impl MutexGroup {
    pub fn from_sas(group: &SASMutexGroup) -> Self {
        let facts = group
            .facts
            .iter()
            .map(|&(var, value)| ExplicitFact { var, value })
            .collect();
        Self { facts }
    }

    pub fn strip_unimportant_facts(&mut self, vars: &[ExplicitVariable]) {
        self.facts.retain(|fact| vars[fact.var].get_level() != -1);
    }

    pub fn is_redundant(&self) -> bool {
        let num_facts = self.facts.len();
        for i in 1..num_facts {
            if self.facts[i].var != self.facts[i - 1].var {
                return false;
            }
        }
        true
    }

    pub fn get_encoding_size(&self) -> usize {
        self.facts.len()
    }

    pub fn to_sas<W: Write>(&self, out: &mut W, vars: &[ExplicitVariable]) {
        writeln!(out, "begin_mutex_group").unwrap();
        writeln!(out, "{}", self.facts.len()).unwrap();
        for fact in &self.facts {
            let var = fact.var;
            writeln!(out, "{} {}", vars[var].get_level(), fact.value).unwrap();
        }
        writeln!(out, "end_mutex_group").unwrap();
    }

    pub fn dump(&self, vars: &[ExplicitVariable]) {
        debug!("mutex group of size {}:", self.facts.len());
        for fact in &self.facts {
            let var = fact.var;
            let value = fact.value;
            debug!(
                "   {} = {} ({})",
                vars[var].get_name(),
                value,
                vars[var].get_fact_name(value)
            );
        }
    }
}
