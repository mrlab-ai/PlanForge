// Minimal axiom_rules stub
#[derive(Debug, Clone)]
pub struct AxiomRule {
    pub name: String,
}

pub struct AxiomRuleBuilder {
    pub rules: Vec<AxiomRule>,
}

impl AxiomRuleBuilder {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }
    pub fn add(&mut self, name: String) {
        self.rules.push(AxiomRule { name });
    }
}
