//! Split rules handling
//! Port of python/translate/split_rules.py

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Rule {
    Join(JoinRule),
    Product(ProductRule),
    Project(ProjectRule),
}

#[derive(Debug, Clone)]
pub struct JoinRule {
    pub effect: String,
    pub conditions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ProductRule {
    pub effect: String,
    pub conditions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ProjectRule {
    pub effect: String,
    pub conditions: Vec<String>,
}

impl Rule {
    pub fn validate(&self) -> bool {
        // TODO: Implement rule validation
        true
    }
}

pub struct RuleSplitter {
    pub rules: Vec<Rule>,
}

impl RuleSplitter {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, rule: Rule) {
        if rule.validate() {
            self.rules.push(rule);
        }
    }

    pub fn split_rules(&mut self) -> Vec<Rule> {
        // TODO: Implement rule splitting algorithm
        self.rules.clone()
    }
}

pub fn variables_to_numbers(
    effect: &str,
    conditions: &[String],
) -> (String, Vec<String>) {
    // TODO: Implement variable to number conversion
    // This is used in the Prolog rule processing
    let mut rename_map: HashMap<String, usize> = HashMap::new();
    let mut counter = 0;

    let new_effect = effect.to_string(); // TODO: Apply renaming
    let new_conditions = conditions.to_vec(); // TODO: Apply renaming

    (new_effect, new_conditions)
}
