//! PDDL to Prolog conversion
//! Port of python/translate/pddl_to_prolog.py

use crate::translate::pddl::{Action, Condition, Literal};

pub struct PrologConverter {
    pub rules: Vec<String>,
    pub facts: Vec<String>,
}

impl PrologConverter {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            facts: Vec::new(),
        }
    }

    pub fn convert_action(&mut self, action: &Action) {
        // Convert action to Prolog rules
        let action_name = format!("action_{}", action.name);
        
        // Convert precondition to rule body
        let precond_prolog = self.condition_to_prolog(&action.precondition);
        
        // Generate rule: action_name :- preconditions.
        let rule = format!("{} :- {}.", action_name, precond_prolog);
        self.rules.push(rule);
    }

    pub fn condition_to_prolog(&self, condition: &Condition) -> String {
        match condition {
            Condition::Literal(lit) => self.literal_to_prolog(lit),
            Condition::And(parts) => {
                let prolog_parts: Vec<String> = parts.iter()
                    .map(|p| self.condition_to_prolog(p))
                    .collect();
                prolog_parts.join(", ")
            }
            Condition::Or(parts) => {
                let prolog_parts: Vec<String> = parts.iter()
                    .map(|p| self.condition_to_prolog(p))
                    .collect();
                format!("({})", prolog_parts.join("; "))
            }
            Condition::Not(inner) => {
                format!("\\+ ({})", self.condition_to_prolog(inner))
            }
            _ => "true".to_string(), // TODO: Handle other condition types
        }
    }

    pub fn literal_to_prolog(&self, literal: &Literal) -> String {
        let atom = if literal.args.is_empty() {
            literal.predicate.clone()
        } else {
            format!("{}({})", literal.predicate, literal.args.join(", "))
        };

        if literal.negated {
            format!("\\+ {}", atom)
        } else {
            atom
        }
    }

    pub fn get_prolog_program(&self) -> String {
        let mut program = String::new();
        
        // Add facts
        for fact in &self.facts {
            program.push_str(fact);
            program.push('\n');
        }
        
        program.push('\n');
        
        // Add rules
        for rule in &self.rules {
            program.push_str(rule);
            program.push('\n');
        }
        
        program
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translate::pddl::*;

    #[test]
    fn test_literal_conversion() {
        let converter = PrologConverter::new();
        let literal = Literal {
            predicate: "at".to_string(),
            args: vec!["robot".to_string(), "room1".to_string()],
            negated: false,
        };
        
        let prolog = converter.literal_to_prolog(&literal);
        assert_eq!(prolog, "at(robot, room1)");
    }
}
