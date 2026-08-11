use std::collections::{HashMap, HashSet};
use std::fmt;

use super::conditions::{Atom, Condition};
use super::f_expression::{
    FunctionalExpression, PrimitiveNumericExpression, instantiate_expression,
};
use super::pddl_types::TypedObject;

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

    /// Returns a PropositionalAxiom or None if statically false.
    pub fn instantiate(
        &self,
        var_mapping: &super::tasks::VarMapping,
        tables: &super::tasks::GroundingTables,
        task_function_admin: &mut super::tasks::DerivedFunctionAdministrator,
        new_constant_axioms: &mut Vec<InstantiatedNumericAxiom>,
    ) -> Option<PropositionalAxiom> {
        // Build the effect atom
        let effect = Atom::new(
            self.name.clone(),
            var_mapping.resolve_parameters(&self.parameters[..self.num_external_parameters]),
        );

        // Instantiate condition
        let condition = self.condition.instantiate_action(
            var_mapping,
            tables,
            task_function_admin,
            new_constant_axioms,
        )?;

        Some(PropositionalAxiom {
            name: self.name.clone(),
            condition,
            effect: Condition::Atom(effect),
        })
    }
}

#[derive(Debug, Clone)]
pub struct PropositionalAxiom {
    pub name: String,
    pub condition: Vec<Condition>,
    pub effect: Condition, // Always Atom
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
        if self.op.is_empty() { 'C' } else { 'D' }
    }

    pub fn get_head(&self) -> PrimitiveNumericExpression {
        let args: Vec<String> = self.parameters.iter().map(|p| p.name.clone()).collect();
        PrimitiveNumericExpression::with_type(self.name.clone(), args, self.ntype())
    }

    pub fn instantiate(
        &self,
        var_mapping: &super::tasks::VarMapping,
        fluent_functions: &HashSet<PrimitiveNumericExpression>,
        init_function_vals: &HashMap<PrimitiveNumericExpression, f64>,
        task_function_admin: &mut super::tasks::DerivedFunctionAdministrator,
        new_constant_axioms: &mut Vec<InstantiatedNumericAxiom>,
    ) -> InstantiatedNumericAxiom {
        let effect = PrimitiveNumericExpression::with_type(
            self.name.clone(),
            var_mapping.resolve_parameters(&self.parameters),
            self.ntype(),
        );

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
}

impl fmt::Display for NumericAxiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NumericAxiom({}, {})", self.name, self.op)
    }
}

#[derive(Debug, Clone)]
pub struct InstantiatedNumericAxiom {
    pub name: String,
    pub op: String,
    pub parts: Vec<FunctionalExpression>,
    pub effect: PrimitiveNumericExpression,
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
