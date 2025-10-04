//! PDDL tasks representation
//! Port of python/translate/pddl/tasks.py

use std::path::Path;
use std::fs::File;
use std::io::Read;
use std::collections::HashMap;

use crate::translate::pddl_parser::{parse_sexprs, SExpr};
use crate::translate::derived_function_admin::DerivedFunctionAdministrator;
use super::{
    Condition, Literal, TypedObject,
    actions::Action,
    axioms::Axiom,
    predicates::Predicate,
    functions::Function,
    pddl_types::Type,
};

/// Main PDDL Task class matching Python implementation
#[derive(Debug, Clone)]
pub struct Task {
    pub domain_name: String,
    pub task_name: String,
    pub requirements: Requirements,
    pub types: Vec<Type>,
    pub objects: Vec<TypedObject>,
    pub predicates: Vec<Predicate>,
    pub functions: Vec<Function>,
    pub init: Vec<Literal>,
    pub num_init: Vec<(String, f64)>, // Numeric initial values
    pub goal: Condition,
    pub actions: Vec<Action>,
    pub axioms: Vec<Axiom>,
    pub axiom_counter: usize,
    pub function_administrator: DerivedFunctionAdministrator,
    pub metric: Option<String>,
    pub global_constraint: Option<Literal>,
}

impl Task {
    /// Function symbols map (static in Python)
    pub fn function_symbols() -> HashMap<String, String> {
        HashMap::new() // TODO: Implement function symbols if needed
    }

    pub fn new(
        domain_name: String,
        task_name: String,
        requirements: Requirements,
        types: Vec<Type>,
        objects: Vec<TypedObject>,
        predicates: Vec<Predicate>,
        functions: Vec<Function>,
        init: Vec<Literal>,
        num_init: Vec<(String, f64)>,
        goal: Condition,
        actions: Vec<Action>,
        axioms: Vec<Axiom>,
        metric: Option<String>,
    ) -> Self {
        Self {
            domain_name,
            task_name,
            requirements,
            types,
            objects,
            predicates,
            functions,
            init,
            num_init,
            goal,
            actions,
            axioms,
            axiom_counter: 0,
            function_administrator: DerivedFunctionAdministrator::new(),
            metric,
            global_constraint: None,
        }
    }

    /// Add global constraints (matches Python implementation)
    pub fn add_global_constraints(&mut self) {
        let debug = false; // TODO: Make configurable
        
        if debug {
            println!("Adding global constraints");
        }
        
        let mut universal_constraints = vec![];
        let mut the_global_constraint = Condition::Truth;
        
        for axiom in &mut self.axioms {
            if axiom.is_global_constraint {
                axiom.is_global_constraint = false;
                let universe = Condition::Forall(super::conditions::UniversalCondition {
                    parameters: axiom.parameters.clone(),
                    parts: vec![axiom.condition.clone()],
                });
                universal_constraints.push(universe);
            }
        }
        
        if !universal_constraints.is_empty() {
            if debug {
                println!("There are {} universal constraints", universal_constraints.len());
            }
            the_global_constraint = Condition::And(universal_constraints);
        }
        
        if debug {
            println!("Adding axiom for global constraint");
        }
        
        let new_axiom = self.add_axiom(vec![], the_global_constraint);
        self.global_constraint = Some(Literal::new(new_axiom.name.clone(), vec![]));
        
        if debug {
            println!("The global constraint is: {:?}", self.global_constraint);
        }
    }

    /// Add axiom (matches Python implementation)
    pub fn add_axiom(&mut self, parameters: Vec<TypedObject>, condition: Condition) -> Axiom {
        let name = format!("new-axiom@{}", self.axiom_counter);
        self.axiom_counter += 1;
        
        let axiom = Axiom::new(name.clone(), parameters.clone(), parameters.len(), condition);
        
        // Add predicate for the axiom
        let predicate = Predicate::new(name.clone(), parameters.clone());
        self.predicates.push(predicate);
        
        self.axioms.push(axiom.clone());
        axiom
    }

    /// Dump task information (matches Python implementation)
    pub fn dump(&self) {
        println!("Problem {}: {} [{}]", self.domain_name, self.task_name, self.requirements);
        
        println!("Types:");
        for type_def in &self.types {
            println!("  {}", type_def);
        }
        
        println!("Objects:");
        for obj in &self.objects {
            println!("  {}", obj);
        }
        
        println!("Predicates:");
        for pred in &self.predicates {
            println!("  {}", pred);
        }
        
        println!("Functions:");
        for func in &self.functions {
            println!("  {}", func);
        }
        
        println!("Init:");
        for fact in &self.init {
            println!("  {}", fact);
        }
        
        println!("Numeric Init:");
        for (name, value) in &self.num_init {
            println!("  {} = {}", name, value);
        }
        
        println!("Goal:");
        // TODO: Implement condition dumping
        println!("  {:?}", self.goal);
        
        println!("Derived Functions:");
        // TODO: Implement function administrator dumping
        println!("  (function administrator dump not implemented)");
        
        println!("Actions:");
        for action in &self.actions {
            action.dump();
        }
        
        if !self.axioms.is_empty() {
            println!("Axioms:");
            for axiom in &self.axioms {
                axiom.dump();
            }
        }
        
        println!("Metric:");
        match &self.metric {
            Some(m) => println!("  {}", m),
            None => println!("  None"),
        }
    }
    
    /// Helper: return a small summary of the task for smoke-tests.
    pub fn summary(&self) -> String {
        format!("domain={}, task={}, types={}, objects={}, predicates={}, functions={}, actions={}, axioms={}",
                self.domain_name, self.task_name, self.types.len(), self.objects.len(),
                self.predicates.len(), self.functions.len(), self.actions.len(), self.axioms.len())
    }
}

/// PDDL Requirements class (matches Python implementation)
#[derive(Debug, Clone)]
pub struct Requirements {
    pub requirements: Vec<String>,
}

impl Requirements {
    pub fn new(requirements: Vec<String>) -> Result<Self, String> {
        let valid_requirements = [
            ":strips", ":adl", ":typing", ":negation", ":equality",
            ":negative-preconditions", ":disjunctive-preconditions",
            ":existential-preconditions", ":universal-preconditions",
            ":quantified-preconditions", ":conditional-effects",
            ":derived-predicates", ":action-costs", ":numeric-fluents",
            ":object-fluents", ":fluents"
        ];
        
        for req in &requirements {
            if !valid_requirements.contains(&req.as_str()) {
                return Err(format!("Invalid requirement: {}", req));
            }
            
            // Handle warnings like Python version
            if req == ":fluents" {
                eprintln!("WARNING: deprecated PDDL option :fluents treated as :numeric-fluents");
            }
            if req == ":object-fluents" {
                eprintln!("WARNING: :object-fluents are not entirely supported yet");
                return Err(":object-fluents are not supported yet".to_string());
            }
        }
        
        Ok(Self { requirements })
    }
}

impl std::fmt::Display for Requirements {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.requirements.join(", "))
    }
}

/// Utility function for pretty printing (matches Python)
pub fn prettyprint(mystring: &str) -> String {
    match mystring {
        "-" => "difference".to_string(),
        "+" => "sum".to_string(),
        _ => mystring.to_string(),
    }
}

/// Minimal placeholder AST for PDDL task used while porting the translator.
#[derive(Debug, Clone)]
pub struct PddlTask {
    pub domain_text: String,
    pub problem_text: String,
    pub domain_forms: Vec<SExpr>,
    pub problem_forms: Vec<SExpr>,
}

impl PddlTask {
    pub fn from_files(domain: &Path, problem: &Path) -> anyhow::Result<Self> {
        let mut d = String::new();
        let mut p = String::new();
        File::open(domain)?.read_to_string(&mut d)?;
        File::open(problem)?.read_to_string(&mut p)?;
        let domain_forms = parse_sexprs(&d).map_err(|e| anyhow::anyhow!(e))?;
        let problem_forms = parse_sexprs(&p).map_err(|e| anyhow::anyhow!(e))?;
        Ok(PddlTask { domain_text: d, problem_text: p, domain_forms, problem_forms })
    }

    /// Helper: return a small summary of the task for smoke-tests.
    pub fn summary(&self) -> String {
        format!("domain={} bytes ({} forms), problem={} bytes ({} forms)",
                self.domain_text.len(), self.domain_forms.len(),
                self.problem_text.len(), self.problem_forms.len())
    }
    
    /// Convert to full Task representation (placeholder for now)
    pub fn to_task(&self) -> Task {
        // TODO: Implement proper parsing from S-expressions to Task
        Task::new(
            "unknown".to_string(), // domain_name
            "unknown".to_string(), // task_name  
            Requirements::new(vec![]).unwrap_or_else(|_| Requirements { requirements: vec![] }),
            vec![], // types
            vec![], // objects
            vec![], // predicates
            vec![], // functions
            vec![], // init
            vec![], // num_init
            Condition::Truth, // goal
            vec![], // actions
            vec![], // axioms
            None, // metric
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_ast_smoke() {
        let task = PddlTask::from_files(std::path::Path::new("pddl/domain.pddl"), std::path::Path::new("pddl/pfile1.pddl")).unwrap();
        // Basic smoke test - just ensure we can load the files
        assert!(!task.domain_text.is_empty());
        assert!(!task.problem_text.is_empty());
    }
    
    #[test]
    fn test_requirements() {
        let req = Requirements::new(vec![":strips".to_string(), ":typing".to_string()]).unwrap();
        assert_eq!(req.requirements.len(), 2);
        
        // Test invalid requirement
        let invalid = Requirements::new(vec![":invalid".to_string()]);
        assert!(invalid.is_err());
    }
}
