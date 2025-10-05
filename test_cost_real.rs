#!/usr/bin/env rust-script
//! This script tests cost extraction from real PDDL files
//! ```cargo
//! [dependencies]
//! planners = { path = "." }
//! ```

use planners::translate::pddl::PddlTask;
use planners::translate::pddl_ast::{Domain, Problem};
use planners::translate::normalize::NormalizableTask;
use std::path::Path;

fn main() {
    let domain_path = Path::new("pddl/domain.pddl");
    let problem_path = Path::new("pddl/pfile1.pddl");
    
    let pddl = PddlTask::from_files(domain_path, problem_path).expect("Failed to parse PDDL");
    let domain = Domain::from_sexprs(&pddl.domain_forms).expect("domain parse");
    let problem = Problem::from_sexprs(&pddl.problem_forms).expect("problem parse");
    
    let task = NormalizableTask::from_ast(&domain, &problem);
    
    println!("Rust Effect Counts (before normalization, after from_ast which does extraction):");
    for action in task.actions.iter().take(3) {
        let has_cost = action.cost.is_some();
        println!("  {}: {} effects, cost={}", 
                 action.name, 
                 action.effects.len(), 
                 if has_cost { "present" } else { "None" });
    }
    
    println!("\nExpected (to match Python):");
    println!("  move: 4 effects (including cost), cost=present");
    println!("  pick: 6 effects (including cost), cost=present");
    println!("  drop: 6 effects (including cost), cost=present");
}
