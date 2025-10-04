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
            Condition::Imply(antecedent, consequent) => {
                // In Prolog: A -> B is equivalent to \+ A ; B
                format!("(\\+ ({}) ; ({}))", 
                    self.condition_to_prolog(antecedent),
                    self.condition_to_prolog(consequent))
            }
            Condition::Exists(_existential) => {
                // Existential quantification - simplified as true for now
                // TODO: Implement proper existential handling with variable scoping
                "true".to_string()
            }
            Condition::Forall(_universal) => {
                // Universal quantification - simplified as true for now
                // TODO: Implement proper universal handling with variable scoping  
                "true".to_string()
            }
            Condition::FunctionComparison(_func_comp) => {
                // Function comparison like (> (fuel) 10)
                format!("comparison({}, {}, {})", 
                    "comparator",
                    "left_expr", // TODO: Convert function expression
                    "right_expr") // TODO: Convert function expression
            }
            Condition::Truth => {
                "true".to_string()
            }
            Condition::Atom(predicate, args) => {
                // Backward compatibility - convert to literal-like format
                if args.is_empty() {
                    predicate.clone()
                } else {
                    format!("{}({})", predicate, args.join(", "))
                }
            }
            Condition::Comparison(op, _left, _right) => {
                // Numeric comparison
                format!("comparison({}, {}, {})", op, "left_sexpr", "right_sexpr")
                // TODO: Convert SExpr to proper Prolog representation
            }
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

    #[test]
    fn test_negated_literal() {
        let converter = PrologConverter::new();
        let literal = Literal {
            predicate: "empty".to_string(),
            args: vec!["box1".to_string()],
            negated: true,
        };
        
        let prolog = converter.literal_to_prolog(&literal);
        assert_eq!(prolog, "\\+ empty(box1)");
    }

    #[test]
    fn test_and_condition() {
        let converter = PrologConverter::new();
        let lit1 = Literal::new("at".to_string(), vec!["robot".to_string(), "room1".to_string()]);
        let lit2 = Literal::new("holding".to_string(), vec!["robot".to_string(), "box".to_string()]);
        
        let and_condition = Condition::And(vec![
            Condition::Literal(lit1),
            Condition::Literal(lit2)
        ]);
        
        let prolog = converter.condition_to_prolog(&and_condition);
        assert_eq!(prolog, "at(robot, room1), holding(robot, box)");
    }

    #[test]
    fn test_or_condition() {
        let converter = PrologConverter::new();
        let lit1 = Literal::new("at".to_string(), vec!["robot".to_string(), "room1".to_string()]);
        let lit2 = Literal::new("at".to_string(), vec!["robot".to_string(), "room2".to_string()]);
        
        let or_condition = Condition::Or(vec![
            Condition::Literal(lit1),
            Condition::Literal(lit2)
        ]);
        
        let prolog = converter.condition_to_prolog(&or_condition);
        assert_eq!(prolog, "(at(robot, room1); at(robot, room2))");
    }

    #[test]
    fn test_not_condition() {
        let converter = PrologConverter::new();
        let lit = Literal::new("empty".to_string(), vec!["box1".to_string()]);
        let not_condition = Condition::Not(Box::new(Condition::Literal(lit)));
        
        let prolog = converter.condition_to_prolog(&not_condition);
        assert_eq!(prolog, "\\+ (empty(box1))");
    }

    #[test]
    fn test_imply_condition() {
        let converter = PrologConverter::new();
        let antecedent = Literal::new("holding".to_string(), vec!["robot".to_string(), "box".to_string()]);
        let consequent = Literal::new("not_empty".to_string(), vec!["robot".to_string()]);
        
        let imply = Condition::Imply(
            Box::new(Condition::Literal(antecedent)),
            Box::new(Condition::Literal(consequent))
        );
        
        let prolog = converter.condition_to_prolog(&imply);
        assert_eq!(prolog, "(\\+ (holding(robot, box)) ; (not_empty(robot)))");
    }

    #[test]
    fn test_truth_condition() {
        let converter = PrologConverter::new();
        let prolog = converter.condition_to_prolog(&Condition::Truth);
        assert_eq!(prolog, "true");
    }

    #[test]
    fn test_atom_condition() {
        let converter = PrologConverter::new();
        let atom = Condition::Atom("goal".to_string(), vec!["robot".to_string(), "room3".to_string()]);
        let prolog = converter.condition_to_prolog(&atom);
        assert_eq!(prolog, "goal(robot, room3)");
    }

    #[test]
    fn test_atom_condition_no_args() {
        let converter = PrologConverter::new();
        let atom = Condition::Atom("finished".to_string(), vec![]);
        let prolog = converter.condition_to_prolog(&atom);
        assert_eq!(prolog, "finished");
    }

    #[test]
    fn test_nested_conditions() {
        let converter = PrologConverter::new();
        let lit1 = Literal::new("at".to_string(), vec!["robot".to_string(), "room1".to_string()]);
        let lit2 = Literal::new("empty".to_string(), vec!["box1".to_string()]);
        let lit3 = Literal::new("goal".to_string(), vec!["room3".to_string()]);
        
        // (at(robot, room1) AND empty(box1)) OR goal(room3)
        let nested = Condition::Or(vec![
            Condition::And(vec![
                Condition::Literal(lit1),
                Condition::Literal(lit2)
            ]),
            Condition::Literal(lit3)
        ]);
        
        let prolog = converter.condition_to_prolog(&nested);
        assert_eq!(prolog, "(at(robot, room1), empty(box1); goal(room3))");
    }
}
