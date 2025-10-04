//! PDDL conditions representation
//! Port of python/translate/pddl/conditions.py

use std::collections::HashMap;
use super::pddl_types::TypedObject;
use super::f_expression::FunctionExpression;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Condition {
    // New structured variants
    Literal(Literal),
    And(Vec<Condition>),
    Or(Vec<Condition>),
    Not(Box<Condition>),
    Imply(Box<Condition>, Box<Condition>),
    Exists(ExistentialCondition),
    Forall(UniversalCondition),
    FunctionComparison(FunctionComparison),
    Truth,
    
    // Backward compatibility variants for porting
    Atom(String, Vec<String>),
    Comparison(String, Box<super::super::pddl_parser::SExpr>, Box<super::super::pddl_parser::SExpr>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Literal {
    pub predicate: String,
    pub args: Vec<String>,
    pub negated: bool,
}

impl Literal {
    pub fn new(predicate: String, args: Vec<String>) -> Self {
        Self { predicate, args, negated: false }
    }
    
    pub fn new_negated(predicate: String, args: Vec<String>) -> Self {
        Self { predicate, args, negated: true }
    }
    
    /// Get key for comparison (matches Python implementation)
    pub fn key(&self) -> (&String, &Vec<String>) {
        (&self.predicate, &self.args)
    }
    
    /// Rename variables according to mapping (matches Python)
    pub fn rename_variables(&self, renamings: &HashMap<String, String>) -> Self {
        let new_args = self.args.iter()
            .map(|arg| renamings.get(arg).unwrap_or(arg).clone())
            .collect();
        Self {
            predicate: self.predicate.clone(),
            args: new_args,
            negated: self.negated,
        }
    }
    
    /// Replace argument at position (matches Python)
    pub fn replace_argument(&self, position: usize, new_arg: String) -> Self {
        let mut new_args = self.args.clone();
        if position < new_args.len() {
            new_args[position] = new_arg;
        }
        Self {
            predicate: self.predicate.clone(),
            args: new_args,
            negated: self.negated,
        }
    }
    
    /// Get free variables (variables starting with "?")
    pub fn free_variables(&self) -> std::collections::HashSet<String> {
        self.args.iter()
            .filter(|arg| arg.starts_with('?'))
            .cloned()
            .collect()
    }
    
    /// Negate this literal
    pub fn negate(&self) -> Self {
        Self {
            predicate: self.predicate.clone(),
            args: self.args.clone(),
            negated: !self.negated,
        }
    }
    
    /// Get positive version
    pub fn positive(&self) -> Self {
        Self {
            predicate: self.predicate.clone(),
            args: self.args.clone(),
            negated: false,
        }
    }
    
    /// Instantiate with variable mapping
    pub fn instantiate(
        &self,
        var_mapping: &HashMap<String, String>,
        init_facts: &[Literal],
        fluent_facts: &[Literal],
    ) -> Result<Option<Literal>, String> {
        let args: Vec<String> = self.args.iter()
            .map(|arg| var_mapping.get(arg).unwrap_or(arg).clone())
            .collect();
        
        let instantiated = Literal {
            predicate: self.predicate.clone(),
            args,
            negated: self.negated,
        };
        
        if !self.negated {
            // For positive literals, check if in fluent or init facts
            if fluent_facts.contains(&instantiated) || init_facts.contains(&instantiated) {
                Ok(Some(instantiated))
            } else {
                Err("Atom not in init or fluent facts".to_string())
            }
        } else {
            // For negative literals, return the instantiated version
            Ok(Some(instantiated))
        }
    }
    
    /// Convert to untyped STRIPS (for positive literals)
    pub fn to_untyped_strips(&self) -> Vec<Literal> {
        if !self.negated {
            vec![self.clone()]
        } else {
            vec![] // Negative literals not supported in STRIPS
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExistentialCondition {
    pub parameters: Vec<TypedObject>,
    pub parts: Vec<Condition>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniversalCondition {
    pub parameters: Vec<TypedObject>,
    pub parts: Vec<Condition>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionComparison {
    pub comparator: String,
    pub parts: Vec<FunctionExpression>,
    pub negated: bool,
}

impl Condition {
    pub fn negate(self) -> Self {
        match self {
            Condition::Not(inner) => *inner,
            other => Condition::Not(Box::new(other)),
        }
    }

    pub fn simplified(self) -> Self {
        match self {
            Condition::And(parts) => {
                let mut result_parts = vec![];
                for part in parts {
                    let simplified_part = part.simplified();
                    match simplified_part {
                        Condition::And(nested_parts) => result_parts.extend(nested_parts),
                        Condition::Truth => {}, // Skip Truth in conjunction
                        other => result_parts.push(other),
                    }
                }
                if result_parts.is_empty() {
                    Condition::Truth
                } else if result_parts.len() == 1 {
                    result_parts.into_iter().next().unwrap()
                } else {
                    Condition::And(result_parts)
                }
            },
            Condition::Or(parts) => {
                let mut result_parts = vec![];
                for part in parts {
                    let simplified_part = part.simplified();
                    match simplified_part {
                        Condition::Or(nested_parts) => result_parts.extend(nested_parts),
                        Condition::Truth => return Condition::Truth, // Truth in disjunction makes whole thing true
                        other => result_parts.push(other),
                    }
                }
                if result_parts.is_empty() {
                    Condition::Truth // Empty disjunction is false, but we'll use Truth as default
                } else if result_parts.len() == 1 {
                    result_parts.into_iter().next().unwrap()
                } else {
                    Condition::Or(result_parts)
                }
            },
            Condition::Not(inner) => {
                match inner.simplified() {
                    Condition::Not(double_neg) => *double_neg, // Double negation elimination
                    simplified => Condition::Not(Box::new(simplified)),
                }
            },
            other => other, // Literals, Truth, etc. are already simplified
        }
    }

    pub fn has_universal_part(&self) -> bool {
        match self {
            Condition::Forall(_) => true,
            Condition::And(parts) | Condition::Or(parts) => {
                parts.iter().any(|p| p.has_universal_part())
            }
            Condition::Not(inner) => inner.has_universal_part(),
            Condition::Exists(exists) => {
                exists.parts.iter().any(|p| p.has_universal_part())
            }
            _ => false,
        }
    }

    pub fn has_existential_part(&self) -> bool {
        match self {
            Condition::Exists(_) => true,
            Condition::And(parts) | Condition::Or(parts) => {
                parts.iter().any(|p| p.has_existential_part())
            }
            Condition::Not(inner) => inner.has_existential_part(),
            Condition::Forall(forall) => {
                forall.parts.iter().any(|p| p.has_existential_part())
            }
            _ => false,
        }
    }

    pub fn has_disjunction(&self) -> bool {
        match self {
            Condition::Or(_) => true,
            Condition::And(parts) => parts.iter().any(|p| p.has_disjunction()),
            Condition::Not(inner) => inner.has_disjunction(),
            Condition::Exists(exists) => exists.parts.iter().any(|p| p.has_disjunction()),
            Condition::Forall(forall) => forall.parts.iter().any(|p| p.has_disjunction()),
            _ => false,
        }
    }

    pub fn change_parts(self, new_parts: Vec<Condition>) -> Self {
        match self {
            Condition::And(_) => Condition::And(new_parts),
            Condition::Or(_) => Condition::Or(new_parts),
            _ => self, // TODO: Handle other cases
        }
    }

    pub fn free_variables(&self) -> std::collections::HashSet<String> {
        match self {
            Condition::Literal(lit) => lit.free_variables(),
            Condition::And(parts) | Condition::Or(parts) => {
                let mut result = std::collections::HashSet::new();
                for part in parts {
                    result.extend(part.free_variables());
                }
                result
            },
            Condition::Not(inner) => inner.free_variables(),
            Condition::Imply(left, right) => {
                let mut result = left.free_variables();
                result.extend(right.free_variables());
                result
            },
            Condition::Exists(exists) => {
                let mut result = exists.parts.iter()
                    .flat_map(|part| part.free_variables())
                    .collect::<std::collections::HashSet<_>>();
                // Remove bound variables
                for param in &exists.parameters {
                    result.remove(&param.name);
                }
                result
            },
            Condition::Forall(forall) => {
                let mut result = forall.parts.iter()
                    .flat_map(|part| part.free_variables())
                    .collect::<std::collections::HashSet<_>>();
                // Remove bound variables
                for param in &forall.parameters {
                    result.remove(&param.name);
                }
                result
            },
            Condition::FunctionComparison(_func_comp) => {
                // TODO: Implement for function expressions
                std::collections::HashSet::new()
            },
            Condition::Truth => std::collections::HashSet::new(),
            Condition::Atom(_, args) => {
                args.iter()
                    .filter(|arg| arg.starts_with('?'))
                    .cloned()
                    .collect()
            },
            Condition::Comparison(_, _left, _right) => {
                // TODO: Extract variables from S-expressions
                std::collections::HashSet::new()
            },
        }
    }

    pub fn uniquify_variables(&self, type_map: &mut HashMap<String, String>) -> Self {
        match self {
            Condition::Literal(lit) => {
                Condition::Literal(lit.rename_variables(type_map))
            },
            Condition::And(parts) => {
                Condition::And(parts.iter()
                    .map(|part| part.uniquify_variables(type_map))
                    .collect())
            },
            Condition::Or(parts) => {
                Condition::Or(parts.iter()
                    .map(|part| part.uniquify_variables(type_map))
                    .collect())
            },
            Condition::Not(inner) => {
                Condition::Not(Box::new(inner.uniquify_variables(type_map)))
            },
            Condition::Imply(left, right) => {
                Condition::Imply(
                    Box::new(left.uniquify_variables(type_map)),
                    Box::new(right.uniquify_variables(type_map))
                )
            },
            Condition::Exists(exists) => {
                // For quantified conditions, we need to handle parameter renaming
                let mut new_type_map = type_map.clone();
                let mut renamings = HashMap::new();
                let new_parameters = exists.parameters.iter()
                    .map(|param| param.uniquify_name(&mut new_type_map, &mut renamings))
                    .collect();
                let new_parts = exists.parts.iter()
                    .map(|part| part.uniquify_variables(&mut new_type_map))
                    .collect();
                Condition::Exists(ExistentialCondition {
                    parameters: new_parameters,
                    parts: new_parts,
                })
            },
            Condition::Forall(forall) => {
                // Similar to Exists
                let mut new_type_map = type_map.clone();
                let mut renamings = HashMap::new();
                let new_parameters = forall.parameters.iter()
                    .map(|param| param.uniquify_name(&mut new_type_map, &mut renamings))
                    .collect();
                let new_parts = forall.parts.iter()
                    .map(|part| part.uniquify_variables(&mut new_type_map))
                    .collect();
                Condition::Forall(UniversalCondition {
                    parameters: new_parameters,
                    parts: new_parts,
                })
            },
            // For the rest, just clone for now
            other => other.clone(),
        }
    }
}
