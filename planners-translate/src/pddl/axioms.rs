/// Port of pddl/axioms.py
use std::collections::{HashMap, HashSet};
use std::fmt;

use super::conditions::{Atom, Condition, Conjunction, NegatedAtom};
use super::f_expression::{
    instantiate_expression, FunctionalExpression, PrimitiveNumericExpression,
};
use super::pddl_types::TypedObject;

/// Python: class Axiom(object)
/// Represents a derived predicate axiom.
#[derive(Debug, Clone)]
pub struct Axiom {
    pub name: String,
    pub parameters: Vec<TypedObject>,
    pub num_external_parameters: usize,
    pub condition: Condition,
    pub is_global_constraint: bool,
}

impl Axiom {
    pub fn new(
        name: String,
        parameters: Vec<TypedObject>,
        num_external_parameters: usize,
        condition: Condition,
    ) -> Self {
        Axiom {
            name,
            parameters,
            num_external_parameters,
            condition,
            is_global_constraint: false,
        }
    }

    pub fn new_global_constraint(
        name: String,
        parameters: Vec<TypedObject>,
        num_external_parameters: usize,
        condition: Condition,
    ) -> Self {
        Axiom {
            name,
            parameters,
            num_external_parameters,
            condition,
            is_global_constraint: true,
        }
    }

    /// Python: def dump(self)
    pub fn dump(&self) {
        println!(
            "Axiom {} ({} params, global={})",
            self.name,
            self.parameters.len(),
            self.is_global_constraint
        );
        println!("  condition: {}", self.condition);
    }

    /// Python: def uniquify_variables(self)
    pub fn uniquify_variables(&mut self) {
        let mut type_map: HashMap<String, usize> = HashMap::new();
        let mut renamings: HashMap<String, String> = HashMap::new();
        for p in &mut self.parameters {
            p.uniquify_name(&mut type_map, &mut renamings);
        }
        self.condition = self
            .condition
            .uniquify_variables(&mut type_map, &mut renamings);
    }

    /// Python: def instantiate(self, var_mapping, init_facts, fluent_facts, ...)
    /// Returns a PropositionalAxiom or None if statically false.
    pub fn instantiate(
        &self,
        var_mapping: &HashMap<String, String>,
        init_facts: &HashSet<Atom>,
        fluent_facts: &HashSet<String>,
        fluent_functions: &HashSet<PrimitiveNumericExpression>,
        init_function_vals: &HashMap<PrimitiveNumericExpression, f64>,
        task_function_admin: &mut super::tasks::DerivedFunctionAdministrator,
        new_constant_axioms: &mut Vec<InstantiatedNumericAxiom>,
    ) -> Option<PropositionalAxiom> {
        // Build the effect atom
        let arg_list: Vec<String> = self.parameters[..self.num_external_parameters]
            .iter()
            .map(|p| {
                var_mapping
                    .get(&p.name)
                    .cloned()
                    .unwrap_or_else(|| p.name.clone())
            })
            .collect();
        let effect = Atom::new(self.name.clone(), arg_list);

        // Instantiate condition
        let condition = match self.condition.instantiate_action(
            var_mapping,
            init_facts,
            fluent_facts,
            fluent_functions,
            init_function_vals,
            task_function_admin,
            new_constant_axioms,
        ) {
            Some(conds) => conds,
            None => return None,
        };

        Some(PropositionalAxiom {
            name: self.name.clone(),
            condition,
            effect: Condition::Atom(effect),
        })
    }
}

/// Python: class PropositionalAxiom(object)
#[derive(Debug, Clone)]
pub struct PropositionalAxiom {
    pub name: String,
    pub condition: Vec<Condition>,
    pub effect: Condition, // Always Atom
}

impl PropositionalAxiom {
    pub fn new(name: String, condition: Vec<Condition>, effect: Condition) -> Self {
        PropositionalAxiom {
            name,
            condition,
            effect,
        }
    }

    /// Python: def clone(self)
    pub fn clone_axiom(&self) -> Self {
        self.clone()
    }

    pub fn dump(&self) {
        println!("PropositionalAxiom");
        println!("  effect: {}", self.effect);
        println!("  conditions:");
        for c in &self.condition {
            println!("    {}", c);
        }
    }
}

impl fmt::Display for PropositionalAxiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PropAxiom({}, {:?} -> {})",
            self.name, self.condition, self.effect
        )
    }
}

/// Python: class NumericAxiom(object)
/// Represents an axiom for derived numeric expressions.
#[derive(Debug, Clone)]
pub struct NumericAxiom {
    pub name: String,
    pub parameters: Vec<TypedObject>,
    pub op: String,
    pub parts: Vec<FunctionalExpression>,
}

impl NumericAxiom {
    pub fn new(
        name: String,
        parameters: Vec<TypedObject>,
        op: String,
        parts: Vec<FunctionalExpression>,
    ) -> Self {
        NumericAxiom {
            name,
            parameters,
            op,
            parts,
        }
    }

    pub fn ntype(&self) -> char {
        if self.op.is_empty() {
            'C'
        } else {
            'D'
        }
    }

    /// Python: def get_head(self)
    pub fn get_head(&self) -> PrimitiveNumericExpression {
        let args: Vec<String> = self.parameters.iter().map(|p| p.name.clone()).collect();
        PrimitiveNumericExpression::with_type(self.name.clone(), args, self.ntype())
    }

    /// Python: def instantiate(self, var_mapping, fluent_functions, task, new_axiom)
    pub fn instantiate(
        &self,
        var_mapping: &HashMap<String, String>,
        fluent_functions: &HashSet<PrimitiveNumericExpression>,
        init_function_vals: &HashMap<PrimitiveNumericExpression, f64>,
        task_function_admin: &mut super::tasks::DerivedFunctionAdministrator,
        new_constant_axioms: &mut Vec<InstantiatedNumericAxiom>,
    ) -> InstantiatedNumericAxiom {
        let new_args: Vec<String> = self
            .parameters
            .iter()
            .map(|p| {
                var_mapping
                    .get(&p.name)
                    .cloned()
                    .unwrap_or_else(|| p.name.clone())
            })
            .collect();
        let effect =
            PrimitiveNumericExpression::with_type(self.name.clone(), new_args, self.ntype());

        let new_parts: Vec<FunctionalExpression> = self
            .parts
            .iter()
            .map(|part| {
                instantiate_expression(
                    part,
                    var_mapping,
                    fluent_functions,
                    init_function_vals,
                    task_function_admin,
                    new_constant_axioms,
                )
            })
            .collect();

        InstantiatedNumericAxiom {
            name: self.name.clone(),
            op: self.op.clone(),
            parts: new_parts,
            effect,
        }
    }

    pub fn dump(&self) {
        println!("NumericAxiom {} op={}", self.name, self.op);
        for p in &self.parts {
            println!("  part: {}", p);
        }
    }
}

impl fmt::Display for NumericAxiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NumericAxiom({}, {})", self.name, self.op)
    }
}

/// Python: class InstantiatedNumericAxiom(object)
#[derive(Debug, Clone)]
pub struct InstantiatedNumericAxiom {
    pub name: String,
    pub op: String,
    pub parts: Vec<FunctionalExpression>,
    pub effect: PrimitiveNumericExpression,
}

impl InstantiatedNumericAxiom {
    pub fn new(
        name: String,
        op: String,
        parts: Vec<FunctionalExpression>,
        effect: PrimitiveNumericExpression,
    ) -> Self {
        InstantiatedNumericAxiom {
            name,
            op,
            parts,
            effect,
        }
    }

    pub fn dump(&self) {
        println!("InstantiatedNumericAxiom {} op={}", self.name, self.op);
        println!("  effect: {}", self.effect);
        for p in &self.parts {
            println!("  part: {}", p);
        }
    }
}

impl fmt::Display for InstantiatedNumericAxiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "InstNumAxiom({}, {} -> {})",
            self.name, self.op, self.effect
        )
    }
}

impl PartialEq for InstantiatedNumericAxiom {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.op == other.op
            && self.parts == other.parts
            && self.effect == other.effect
    }
}

impl Eq for InstantiatedNumericAxiom {}

impl std::hash::Hash for InstantiatedNumericAxiom {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.op.hash(state);
        self.parts.hash(state);
        self.effect.hash(state);
    }
}
