//! PDDL axioms representation
//! Port of python/translate/pddl/axioms.py

use std::collections::HashMap;
use super::{Condition, Literal, TypedObject};

#[derive(Debug, Clone)]
pub struct Axiom {
    pub name: String,
    pub parameters: Vec<TypedObject>,
    pub num_external_parameters: usize,
    pub condition: Condition,
    pub is_global_constraint: bool,
    pub type_map: HashMap<String, String>,
}

impl Axiom {
    pub fn new(
        name: String,
        parameters: Vec<TypedObject>,
        num_external_parameters: usize,
        condition: Condition,
    ) -> Self {
        let mut axiom = Self {
            name,
            parameters,
            num_external_parameters,
            condition,
            is_global_constraint: false,
            type_map: HashMap::new(),
        };
        axiom.uniquify_variables();
        axiom
    }

    /// Create axiom with global constraint flag
    pub fn with_global_constraint(
        name: String,
        parameters: Vec<TypedObject>,
        num_external_parameters: usize,
        condition: Condition,
        is_global_constraint: bool,
    ) -> Self {
        let mut axiom = Self {
            name,
            parameters,
            num_external_parameters,
            condition,
            is_global_constraint,
            type_map: HashMap::new(),
        };
        axiom.uniquify_variables();
        axiom
    }

    pub fn dump(&self) {
        let args: Vec<String> = self.parameters[..self.num_external_parameters]
            .iter()
            .map(|p| p.to_string())
            .collect();
        println!("Axiom {}({})", self.name, args.join(", "));
        // TODO: Implement condition dumping
        println!("  Condition: {:?}", self.condition);
    }

    pub fn uniquify_variables(&mut self) {
        self.type_map = self.parameters
            .iter()
            .map(|par| (par.name.clone(), par.get_type_name().to_string()))
            .collect();
        // TODO: Implement condition variable uniquification
        // self.condition = self.condition.uniquify_variables(&self.type_map);
    }

    /// Instantiate the axiom with variable mappings
    pub fn instantiate(
        &self,
        var_mapping: &HashMap<String, String>,
        _init_facts: &[Literal],
        _fluent_facts: &[Literal],
        _init_function_vals: &HashMap<String, f64>,
        _fluent_functions: &[String],
        // TODO: Add task, new_axiom, new_modules parameters
    ) -> Option<PropositionalAxiom> {
        // Build argument list for the axiom name
        let mut arg_list = vec![self.name.clone()];
        for par in &self.parameters[..self.num_external_parameters] {
            if let Some(mapped_name) = var_mapping.get(&par.name) {
                arg_list.push(mapped_name.clone());
            } else {
                arg_list.push(par.name.clone());
            }
        }
        let name = format!("({})", arg_list.join(" "));

        // TODO: Instantiate condition
        let condition = vec![]; // Placeholder

        // Build effect
        let effect_args: Vec<String> = self.parameters[..self.num_external_parameters]
            .iter()
            .map(|arg| {
                var_mapping.get(&arg.name)
                    .unwrap_or(&arg.name)
                    .clone()
            })
            .collect();
        
        let effect = Literal {
            predicate: self.name.clone(),
            args: effect_args,
            negated: false,
        };

        Some(PropositionalAxiom::new(name, condition, effect))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropositionalAxiom {
    pub name: String,
    pub condition: Vec<Literal>,
    pub effect: Literal,
}

impl PropositionalAxiom {
    pub fn new(name: String, condition: Vec<Literal>, effect: Literal) -> Self {
        Self { name, condition, effect }
    }

    pub fn clone_axiom(&self) -> Self {
        Self {
            name: self.name.clone(),
            condition: self.condition.clone(),
            effect: self.effect.clone(),
        }
    }

    pub fn dump(&self) {
        if self.effect.negated {
            print!("not ");
        }
        println!("{}", self.name);
        for fact in &self.condition {
            println!(" PRE: {}", fact);
        }
        println!(" EFF: {}", self.effect);
    }

    /// Get key for comparison and sorting
    pub fn key(&self) -> (&String, &Vec<Literal>, &Literal) {
        (&self.name, &self.condition, &self.effect)
    }
}

impl PartialOrd for PropositionalAxiom {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PropositionalAxiom {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key().cmp(&other.key())
    }
}

impl std::fmt::Display for PropositionalAxiom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<PropositionalAxiom {} {:?} -> {}>", 
               self.name, self.condition, self.effect)
    }
}

#[derive(Debug, Clone)]
pub struct NumericAxiom {
    pub name: String,
    pub parameters: Vec<TypedObject>,
    pub op: String,
    pub parts: Vec<super::f_expression::FunctionExpression>,
}

impl NumericAxiom {
    pub fn new(
        name: String,
        parameters: Vec<TypedObject>,
        op: String,
        parts: Vec<super::f_expression::FunctionExpression>,
    ) -> Self {
        Self { name, parameters, op, parts }
    }
}

#[derive(Debug, Clone)]
pub struct DerivedPredicate {
    pub name: String,
    pub arguments: Vec<String>,
}

impl DerivedPredicate {
    pub fn new(name: String, arguments: Vec<String>) -> Self {
        Self { name, arguments }
    }

    pub fn get_atom(&self, args: Vec<String>) -> Literal {
        Literal {
            predicate: self.name.clone(),
            args,
            negated: false,
        }
    }
}

impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.negated {
            write!(f, "NegatedAtom {}({})", self.predicate, self.args.join(", "))
        } else {
            write!(f, "Atom {}({})", self.predicate, self.args.join(", "))
        }
    }
}
