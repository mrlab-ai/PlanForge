use std::io::Write;

use crate::helper_functions::check_magic;
use crate::helper_functions::InputStream;
use crate::variable::Variable;

#[derive(Debug, Clone)]
pub struct MutexGroup {
    facts: Vec<(*const Variable, i32)>,
}

impl MutexGroup {
    pub fn from_stream(stream: &mut InputStream, variables: &Vec<*mut Variable>) -> Self {
        check_magic(stream, "begin_mutex_group");
        let size = stream.read_i32();
        let mut facts = Vec::with_capacity(size as usize);
        for _ in 0..size {
            let var_no = stream.read_i32();
            let value = stream.read_i32();
            let var = variables[var_no as usize] as *const Variable;
            facts.push((var, value));
        }
        check_magic(stream, "end_mutex_group");
        Self { facts }
    }

    pub fn strip_unimportant_facts(&mut self) {
        self.facts.retain(|fact| {
            let var = unsafe { &*fact.0 };
            var.get_level() != -1
        });
    }

    pub fn is_redundant(&self) -> bool {
        let num_facts = self.facts.len();
        for i in 1..num_facts {
            if self.facts[i].0 != self.facts[i - 1].0 {
                return false;
            }
        }
        true
    }

    pub fn get_encoding_size(&self) -> usize {
        self.facts.len()
    }

    pub fn generate_cpp_input<W: Write>(&self, out: &mut W) {
        writeln!(out, "begin_mutex_group").unwrap();
        writeln!(out, "{}", self.facts.len()).unwrap();
        for fact in &self.facts {
            let var = unsafe { &*fact.0 };
            writeln!(out, "{} {}", var.get_level(), fact.1).unwrap();
        }
        writeln!(out, "end_mutex_group").unwrap();
    }

    pub fn dump(&self) {
        println!("mutex group of size {}:", self.facts.len());
        for fact in &self.facts {
            let var = unsafe { &*fact.0 };
            let value = fact.1;
            println!(
                "   {} = {} ({})",
                var.get_name(),
                value,
                var.get_fact_name(value as usize)
            );
        }
    }
}

pub fn strip_mutexes(mutexes: &mut Vec<MutexGroup>) {
    let old_count = mutexes.len();
    for mutex in mutexes.iter_mut() {
        mutex.strip_unimportant_facts();
    }
    mutexes.retain(|m| !m.is_redundant());
    println!("{} of {} mutex groups necessary.", mutexes.len(), old_count);
}
