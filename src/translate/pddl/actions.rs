//! PDDL actions representation
//! Port of python/translate/pddl/actions.py

use std::collections::HashMap;
use super::{Condition, TypedObject};
use super::effects::Effect;
use crate::translate::pddl_parser::SExpr;

#[derive(Debug, Clone)]
pub struct Action {
    pub name: String,
    pub parameters: Vec<TypedObject>,
    pub num_external_parameters: usize,
    pub precondition: Condition,
    pub effects: Vec<Effect>,
    pub cost: Option<NumericEffect>, // TODO: Implement NumericEffect properly
    pub type_map: HashMap<String, String>,
}

/// Placeholder for NumericEffect (matches Python)
#[derive(Debug, Clone)]
pub struct NumericEffect {
    pub fluent: String,
    pub expression: String,
    pub effect_type: String, // "increase", "decrease", "assign"
}

impl Action {
    pub fn new(
        name: String,
        parameters: Vec<TypedObject>,
        num_external_parameters: usize,
        precondition: Condition,
        effects: Vec<Effect>,
        cost: Option<NumericEffect>,
    ) -> Self {
        let mut action = Self {
            name,
            parameters,
            num_external_parameters,
            precondition,
            effects,
            cost,
            type_map: HashMap::new(),
        };
        action.uniquify_variables();
        action
    }

    /// Parse action from S-expression list (matches Python static method)
    pub fn parse(alist: &[SExpr]) -> Result<Option<Self>, String> {
        if alist.is_empty() {
            return Err("Empty action list".to_string());
        }
        
        let mut iter = alist.iter();
        
        // Check action tag
        match iter.next() {
            Some(SExpr::Atom(tag)) if tag == ":action" => {},
            _ => return Err("Expected :action tag".to_string()),
        }
        
        // Get action name
        let name = match iter.next() {
            Some(SExpr::Atom(n)) => n.clone(),
            _ => return Err("Expected action name".to_string()),
        };
        
        // Parse parameters (optional)
        let mut parameters = vec![];
        let mut next_tag = iter.next();
        
        if let Some(SExpr::Atom(tag)) = next_tag {
            if tag == ":parameters" {
                match iter.next() {
                    Some(SExpr::List(param_list)) => {
                        // TODO: Implement proper typed parameter parsing
                        // For now, create simple parameters
                        for param in param_list {
                            if let SExpr::Atom(param_name) = param {
                                if param_name.starts_with('?') {
                                    parameters.push(TypedObject::new(param_name.clone(), None));
                                }
                            }
                        }
                    },
                    _ => return Err("Expected parameter list".to_string()),
                }
                next_tag = iter.next();
            }
        }
        
        // Parse precondition (optional)
        let mut precondition = Condition::Truth;
        
        if let Some(SExpr::Atom(tag)) = next_tag {
            if tag == ":precondition" {
                match iter.next() {
                    Some(precond_expr) => {
                        // TODO: Implement proper condition parsing
                        precondition = super::sexpr_to_condition(precond_expr)
                            .unwrap_or(Condition::Truth);
                        precondition = precondition.simplified();
                    },
                    _ => return Err("Expected precondition expression".to_string()),
                }
                next_tag = iter.next();
            }
        }
        
        // Parse effects (required)
        if let Some(SExpr::Atom(tag)) = next_tag {
            if tag != ":effect" {
                return Err("Expected :effect tag".to_string());
            }
        } else {
            return Err("Expected :effect tag".to_string());
        }
        
        let mut effects = vec![];
        let cost = None;
        
        match iter.next() {
            Some(effect_expr) => {
                // TODO: Implement proper effect parsing
                if let Some(effect) = super::sexpr_to_effect(effect_expr) {
                    effects.push(effect);
                }
                // TODO: Extract cost from effects
            },
            _ => return Err("Expected effect expression".to_string()),
        }
        
        // Check for remaining elements
        if iter.next().is_some() {
            return Err("Unexpected elements after effect".to_string());
        }
        
        if !effects.is_empty() {
            Ok(Some(Action::new(
                name,
                parameters.clone(),
                parameters.len(),
                precondition,
                effects,
                cost,
            )))
        } else {
            Ok(None)
        }
    }

    pub fn dump(&self) {
        let params: Vec<String> = self.parameters.iter().map(|p| p.to_string()).collect();
        println!("{}({})", self.name, params.join(", "));
        
        println!("Precondition:");
        // TODO: Implement condition dumping
        println!("  {:?}", self.precondition);
        
        println!("Effects:");
        for eff in &self.effects {
            eff.dump();
        }
        
        println!("Cost:");
        match &self.cost {
            Some(c) => {
                // TODO: Implement cost dumping
                println!("  {:?}", c);
            },
            None => println!("  None"),
        }
    }

    pub fn uniquify_variables(&mut self) {
        self.type_map = self.parameters
            .iter()
            .map(|par| (par.name.clone(), par.get_type_name().to_string()))
            .collect();
        
        self.precondition = self.precondition.uniquify_variables(&mut self.type_map);
        
        for effect in &mut self.effects {
            effect.uniquify_variables(&mut self.type_map);
        }
        
        // TODO: uniquify variables in cost
    }

    /// Create relaxed version of action (remove negative effects)
    pub fn relaxed(&self) -> Self {
        let mut new_effects = vec![];
        for eff in &self.effects {
            if let Some(relaxed_eff) = eff.relaxed() {
                new_effects.push(relaxed_eff);
            }
        }
        
        // TODO: Implement condition.relaxed()
        let relaxed_precondition = self.precondition.clone().simplified();
        
        Action::new(
            self.name.clone(),
            self.parameters.clone(),
            self.num_external_parameters,
            relaxed_precondition,
            new_effects,
            self.cost.clone(),
        )
    }
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<String> = self.parameters.iter().map(|p| p.to_string()).collect();
        write!(f, "<Action {} [{}]>", self.name, params.join(", "))
    }
}
