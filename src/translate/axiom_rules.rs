//! Axiom rules handling
//! Port of python/translate/axiom_rules.py

use crate::translate::pddl::{Axiom, Condition};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AxiomRule {
    pub condition: Condition,
    pub effect: String, // Axiom name
    pub layer: usize,
}

impl AxiomRule {
    pub fn new(condition: Condition, effect: String, layer: usize) -> Self {
        Self { condition, effect, layer }
    }
}

pub struct AxiomRuleBuilder {
    pub axioms: Vec<Axiom>,
    pub rules: Vec<AxiomRule>,
    pub layer_map: HashMap<String, usize>,
}

impl AxiomRuleBuilder {
    pub fn new() -> Self {
        Self {
            axioms: Vec::new(),
            rules: Vec::new(),
            layer_map: HashMap::new(),
        }
    }

    pub fn add_axiom(&mut self, axiom: Axiom) {
        // TODO: Implement axiom layer computation
        let layer = self.compute_axiom_layer(&axiom);
        self.layer_map.insert(axiom.name.clone(), layer);
        
        let rule = AxiomRule::new(
            axiom.condition.clone(),
            axiom.name.clone(),
            layer,
        );
        
        self.rules.push(rule);
        self.axioms.push(axiom);
    }

    fn compute_axiom_layer(&self, _axiom: &Axiom) -> usize {
        // TODO: Implement proper layer computation based on dependencies
        0
    }

    pub fn get_rules(&self) -> &[AxiomRule] {
        &self.rules
    }
}
