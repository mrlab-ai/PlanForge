//! Axiom rules handling for propositional axioms.
//!
//! This module deals with propositional axioms. Numeric axioms are treated in numeric_axiom_rules.rs.
//! 
//! Ported from Python's axiom_rules.py in numeric-fd.

use std::collections::{HashMap, HashSet};

use crate::translate::instantiate::GroundedOp;

/// A literal is a predicate with optional negation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Literal {
    pub predicate: String,
    pub args: Vec<String>,
    pub negated: bool,
}

impl Literal {
    pub fn new(predicate: String, args: Vec<String>, negated: bool) -> Self {
        Self { predicate, args, negated }
    }
    
    /// Get the positive version of this literal
    pub fn positive(&self) -> Self {
        Self {
            predicate: self.predicate.clone(),
            args: self.args.clone(),
            negated: false,
        }
    }
    
    /// Get the negated version of this literal
    pub fn negate(&self) -> Self {
        Self {
            predicate: self.predicate.clone(),
            args: self.args.clone(),
            negated: !self.negated,
        }
    }
    
    /// Create a key for HashMap lookups (just the predicate and args, ignoring negation for atom-based lookups)
    pub fn atom_key(&self) -> (String, Vec<String>) {
        (self.predicate.clone(), self.args.clone())
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

/// A propositional axiom: if all conditions hold, then effect becomes true
#[derive(Debug, Clone)]
pub struct PropositionalAxiom {
    pub name: String,
    pub condition: Vec<Literal>,
    pub effect: Literal,
}

impl PropositionalAxiom {
    pub fn new(name: String, condition: Vec<Literal>, effect: Literal) -> Self {
        Self { name, condition, effect }
    }
    
    /// Clone the axiom
    pub fn clone_axiom(&self) -> Self {
        Self {
            name: self.name.clone(),
            condition: self.condition.clone(),
            effect: self.effect.clone(),
        }
    }
}

/// Result of handling axioms
pub struct AxiomHandleResult {
    /// The processed axioms
    pub axioms: Vec<PropositionalAxiom>,
    /// Initial atoms that should be true for axioms
    pub axiom_init: Vec<Literal>,
    /// Mapping from atom to its axiom layer
    pub axiom_layer_dict: HashMap<(String, Vec<String>), i32>,
}

/// Main entry point: handle propositional axioms
/// 
/// This function:
/// 1. Groups axioms by their effect atom
/// 2. Computes which axiom literals are necessary (used in goals/operators)
/// 3. Determines initial truth values for axiom atoms
/// 4. Simplifies axioms (removes duplicates and dominated axioms)
/// 5. Computes negative axioms for negated literals
/// 6. Stratifies axioms into layers
pub fn handle_axioms(
    operators: &[GroundedOp],
    axioms: &[PropositionalAxiom],
    goal_list: &[Literal],
    global_constraint: &Literal,
) -> AxiomHandleResult {
    let axioms_by_atom = get_axioms_by_atom(axioms);
    let axiom_literals = compute_necessary_axiom_literals(&axioms_by_atom, operators, goal_list, global_constraint);
    let axiom_init = get_axiom_init(&axioms_by_atom, &axiom_literals);
    
    // Simplify axioms
    let simplified_axioms = simplify_axioms(&axioms_by_atom, &axiom_literals);
    
    // Compute negative axioms
    let all_axioms = compute_negative_axioms(&axioms_by_atom, &axiom_literals, &simplified_axioms);
    
    // Compute axiom layers
    let axiom_layer_dict = compute_axiom_layers(&all_axioms, &axiom_init);
    
    AxiomHandleResult {
        axioms: all_axioms,
        axiom_init: axiom_init.into_iter().collect(),
        axiom_layer_dict,
    }
}

/// Group axioms by their effect atom (positive version)
fn get_axioms_by_atom(axioms: &[PropositionalAxiom]) -> HashMap<(String, Vec<String>), Vec<PropositionalAxiom>> {
    let mut axioms_by_atom: HashMap<(String, Vec<String>), Vec<PropositionalAxiom>> = HashMap::new();
    for axiom in axioms {
        let key = axiom.effect.atom_key();
        axioms_by_atom.entry(key).or_default().push(axiom.clone());
    }
    axioms_by_atom
}

/// Compute which axiom literals are necessary.
/// 
/// A literal is necessary if it appears in:
/// - Goal conditions
/// - Global constraint
/// - Operator preconditions
/// - Effect conditions
/// - Conditions of other necessary axioms (transitively)
fn compute_necessary_axiom_literals(
    axioms_by_atom: &HashMap<(String, Vec<String>), Vec<PropositionalAxiom>>,
    operators: &[GroundedOp],
    goal_list: &[Literal],
    global_constraint: &Literal,
) -> HashSet<Literal> {
    let mut necessary_literals: HashSet<Literal> = HashSet::new();
    let mut queue: Vec<Literal> = Vec::new();
    
    // Helper to register literals
    let mut register_literals = |literals: &[Literal], negated: bool, necessary: &mut HashSet<Literal>, q: &mut Vec<Literal>| {
        for literal in literals {
            let key = literal.positive().atom_key();
            if axioms_by_atom.contains_key(&key) {
                // This is an axiom literal
                let mut lit = literal.clone();
                if negated {
                    lit = lit.negate();
                }
                if !necessary.contains(&lit) {
                    necessary.insert(lit.clone());
                    q.push(lit);
                }
            }
        }
    };
    
    // Initialize queue with axioms required for goal_list and global constraint
    register_literals(goal_list, false, &mut necessary_literals, &mut queue);
    register_literals(&[global_constraint.clone()], false, &mut necessary_literals, &mut queue);
    
    // Add from operator preconditions and effect conditions
    for op in operators {
        if let Some(pre) = &op.pre {
            let pre_literals = condition_to_literals(pre);
            register_literals(&pre_literals, false, &mut necessary_literals, &mut queue);
        }
        // Note: In full implementation, we'd also extract effect conditions
        // For now, we handle preconditions which is the main source
    }
    
    // BFS to find all transitively needed axioms
    while let Some(literal) = queue.pop() {
        let key = literal.positive().atom_key();
        if let Some(axioms) = axioms_by_atom.get(&key) {
            for axiom in axioms {
                let negated = literal.negated;
                register_literals(&axiom.condition, negated, &mut necessary_literals, &mut queue);
            }
        }
    }
    
    necessary_literals
}

/// Convert a Condition to a list of Literals
fn condition_to_literals(cond: &crate::translate::pddl_ast::Condition) -> Vec<Literal> {
    use crate::translate::pddl_ast::Condition;
    
    match cond {
        Condition::Atom(name, args) => {
            vec![Literal::new(name.clone(), args.clone(), false)]
        }
        Condition::Not(inner) => {
            match inner.as_ref() {
                Condition::Atom(name, args) => {
                    vec![Literal::new(name.clone(), args.clone(), true)]
                }
                _ => condition_to_literals(inner).into_iter().map(|l| l.negate()).collect(),
            }
        }
        Condition::And(parts) => {
            parts.iter().flat_map(condition_to_literals).collect()
        }
        Condition::Or(parts) => {
            parts.iter().flat_map(condition_to_literals).collect()
        }
        Condition::Forall(_, inner) | Condition::Exists(_, inner) => {
            condition_to_literals(inner)
        }
        Condition::True | Condition::Comparison(_, _, _) => vec![],
    }
}

/// Get initial values for axiom atoms.
/// 
/// Initial value for axiom: False (which is omitted due to closed world assumption)
/// unless it is only needed negatively.
fn get_axiom_init(
    axioms_by_atom: &HashMap<(String, Vec<String>), Vec<PropositionalAxiom>>,
    necessary_literals: &HashSet<Literal>,
) -> HashSet<Literal> {
    let mut result: HashSet<Literal> = HashSet::new();
    
    for key in axioms_by_atom.keys() {
        let atom = Literal::new(key.0.clone(), key.1.clone(), false);
        let negated_atom = atom.negate();
        
        // If atom is not needed positively but is needed negatively,
        // then its initial value should be true
        if !necessary_literals.contains(&atom) && necessary_literals.contains(&negated_atom) {
            result.insert(atom);
        }
    }
    
    result
}

/// Simplify axioms for necessary atoms
fn simplify_axioms(
    axioms_by_atom: &HashMap<(String, Vec<String>), Vec<PropositionalAxiom>>,
    necessary_literals: &HashSet<Literal>,
) -> HashMap<(String, Vec<String>), Vec<PropositionalAxiom>> {
    let necessary_atoms: HashSet<(String, Vec<String>)> = necessary_literals
        .iter()
        .map(|l| l.positive().atom_key())
        .collect();
    
    let mut result: HashMap<(String, Vec<String>), Vec<PropositionalAxiom>> = HashMap::new();
    
    for atom_key in &necessary_atoms {
        if let Some(axioms) = axioms_by_atom.get(atom_key) {
            let simplified = simplify(axioms);
            result.insert(atom_key.clone(), simplified);
        }
    }
    
    result
}

/// Remove duplicates from a sorted list in-place style (returns new vec)
fn remove_duplicates<T: PartialEq + Clone>(list: &[T]) -> Vec<T> {
    if list.is_empty() {
        return vec![];
    }
    let mut result = vec![list[0].clone()];
    for item in &list[1..] {
        if result.last() != Some(item) {
            result.push(item.clone());
        }
    }
    result
}

/// Remove duplicate axioms, duplicates within axioms, and dominated axioms.
fn simplify(axioms: &[PropositionalAxiom]) -> Vec<PropositionalAxiom> {
    if axioms.is_empty() {
        return vec![];
    }
    
    // Remove duplicates from axiom conditions
    let mut processed_axioms: Vec<PropositionalAxiom> = axioms.iter().map(|ax| {
        let mut new_ax = ax.clone();
        // Sort conditions by their string representation for consistent ordering
        new_ax.condition.sort_by(|a, b| format!("{}", a).cmp(&format!("{}", b)));
        new_ax.condition = remove_duplicates(&new_ax.condition);
        new_ax
    }).collect();
    
    // Build index: literal -> axioms containing it
    let mut axioms_by_literal: HashMap<String, HashSet<usize>> = HashMap::new();
    for (idx, axiom) in processed_axioms.iter().enumerate() {
        for literal in &axiom.condition {
            let key = format!("{}", literal);
            axioms_by_literal.entry(key).or_default().insert(idx);
        }
    }
    
    // Find dominated axioms (supersets of other axioms' conditions)
    let mut axioms_to_skip: HashSet<usize> = HashSet::new();
    
    for (idx, axiom) in processed_axioms.iter().enumerate() {
        if axioms_to_skip.contains(&idx) {
            continue; // Required to keep one of multiple identical axioms
        }
        
        if axiom.condition.is_empty() {
            // Empty condition dominates everything
            return vec![axiom.clone()];
        }
        
        // Find axioms with superset conditions (those that contain all our literals)
        let mut dominated_axioms: HashSet<usize> = {
            let first_lit = format!("{}", &axiom.condition[0]);
            axioms_by_literal.get(&first_lit).cloned().unwrap_or_default()
        };
        
        for literal in &axiom.condition[1..] {
            let key = format!("{}", literal);
            if let Some(containing) = axioms_by_literal.get(&key) {
                dominated_axioms = dominated_axioms.intersection(containing).cloned().collect();
            } else {
                dominated_axioms.clear();
                break;
            }
        }
        
        for dominated_idx in dominated_axioms {
            if dominated_idx != idx {
                axioms_to_skip.insert(dominated_idx);
            }
        }
    }
    
    processed_axioms
        .into_iter()
        .enumerate()
        .filter(|(idx, _)| !axioms_to_skip.contains(idx))
        .map(|(_, ax)| ax)
        .collect()
}

/// Compute negative axioms for literals that are needed negatively
fn compute_negative_axioms(
    axioms_by_atom: &HashMap<(String, Vec<String>), Vec<PropositionalAxiom>>,
    necessary_literals: &HashSet<Literal>,
    simplified_axioms: &HashMap<(String, Vec<String>), Vec<PropositionalAxiom>>,
) -> Vec<PropositionalAxiom> {
    let mut new_axioms: Vec<PropositionalAxiom> = Vec::new();
    
    for literal in necessary_literals {
        let key = literal.positive().atom_key();
        
        if literal.negated {
            // Need the negation - compute negative axioms
            if let Some(axioms) = axioms_by_atom.get(&key) {
                new_axioms.extend(negate(axioms));
            }
        } else {
            // Need the positive version - use simplified axioms
            if let Some(axioms) = simplified_axioms.get(&key) {
                new_axioms.extend(axioms.clone());
            }
        }
    }
    
    new_axioms
}

/// Create axioms for the negation of a derived predicate.
/// 
/// If the original axioms are: effect :- cond1 AND cond2 AND ...
/// Then negation is: NOT effect :- NOT cond1 OR NOT cond2 OR ...
/// 
/// Which in DNF (for multiple axioms) becomes a cross product.
fn negate(axioms: &[PropositionalAxiom]) -> Vec<PropositionalAxiom> {
    if axioms.is_empty() {
        return vec![];
    }
    
    // Start with a single axiom with empty condition and negated effect
    let negated_effect = axioms[0].effect.negate();
    let mut result: Vec<PropositionalAxiom> = vec![
        PropositionalAxiom::new(
            axioms[0].name.clone(),
            vec![],
            negated_effect.clone(),
        )
    ];
    
    for axiom in axioms {
        let condition = &axiom.condition;
        
        if condition.is_empty() {
            // The derived fact we want to negate is triggered with an empty condition,
            // so it is always true and its negation is always false.
            return vec![];
        } else if condition.len() == 1 {
            // Handle easy special case quickly
            let new_literal = condition[0].negate();
            for result_axiom in &mut result {
                result_axiom.condition.push(new_literal.clone());
            }
        } else {
            // Multiply out: (A ∧ B) -> ¬A ∨ ¬B becomes multiple axioms
            let mut new_result: Vec<PropositionalAxiom> = Vec::new();
            for literal in condition {
                let negated_literal = literal.negate();
                for result_axiom in &result {
                    let mut new_axiom = result_axiom.clone_axiom();
                    new_axiom.condition.push(negated_literal.clone());
                    new_result.push(new_axiom);
                }
            }
            result = new_result;
        }
    }
    
    // Simplify the result
    simplify(&result)
}

/// Compute axiom layers using DFS with cycle detection.
/// 
/// Axiom layers determine the order in which axioms must be evaluated.
/// An axiom at layer N can only depend on axioms at layers < N.
fn compute_axiom_layers(
    axioms: &[PropositionalAxiom],
    axiom_init: &HashSet<Literal>,
) -> HashMap<(String, Vec<String>), i32> {
    const NO_AXIOM: i32 = -1;
    const UNKNOWN_LAYER: i32 = -2;
    const FIRST_MARKER: i32 = -3;
    
    // Build dependency graph
    let mut depends_on: HashMap<(String, Vec<String>), HashSet<((String, Vec<String>), i32)>> = HashMap::new();
    
    for axiom in axioms {
        let effect_atom = axiom.effect.positive();
        let effect_key = effect_atom.atom_key();
        let effect_sign = !axiom.effect.negated;
        let effect_init_sign = axiom_init.contains(&effect_atom);
        
        if effect_sign != effect_init_sign {
            depends_on.entry(effect_key.clone()).or_default();
            
            for condition in &axiom.condition {
                let condition_atom = condition.positive();
                let condition_key = condition_atom.atom_key();
                let condition_sign = !condition.negated;
                let condition_init_sign = axiom_init.contains(&condition_atom);
                
                let bonus = if condition_sign == condition_init_sign { 1 } else { 0 };
                depends_on.get_mut(&effect_key).unwrap().insert((condition_key, bonus));
            }
        }
    }
    
    // Initialize layers
    let mut layers: HashMap<(String, Vec<String>), i32> = depends_on
        .keys()
        .map(|k| (k.clone(), UNKNOWN_LAYER))
        .collect();
    
    // Find level with DFS and cycle detection
    fn find_level(
        atom: &(String, Vec<String>),
        marker: i32,
        layers: &mut HashMap<(String, Vec<String>), i32>,
        depends_on: &HashMap<(String, Vec<String>), HashSet<((String, Vec<String>), i32)>>,
    ) -> i32 {
        let layer = *layers.get(atom).unwrap_or(&NO_AXIOM);
        
        if layer == NO_AXIOM {
            return 0;
        }
        
        if layer == marker {
            // Found positive cycle: May return 0 but not set value
            return 0;
        } else if layer <= FIRST_MARKER {
            // Found negative cycle: Error
            panic!("Cyclic dependencies in axioms; cannot stratify.");
        }
        
        if layer == UNKNOWN_LAYER {
            layers.insert(atom.clone(), marker);
            let mut computed_layer = 0;
            
            if let Some(deps) = depends_on.get(atom) {
                for (condition_atom, bonus) in deps {
                    let dep_layer = find_level(condition_atom, marker - bonus, layers, depends_on);
                    computed_layer = computed_layer.max(dep_layer + bonus);
                }
            }
            
            layers.insert(atom.clone(), computed_layer);
            return computed_layer;
        }
        
        layer
    }
    
    // Compute layers for all atoms
    let atoms: Vec<_> = depends_on.keys().cloned().collect();
    for atom in &atoms {
        find_level(atom, FIRST_MARKER, &mut layers, &depends_on);
    }
    
    layers
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_literal_positive() {
        let lit = Literal::new("pred".to_string(), vec!["a".to_string()], true);
        let pos = lit.positive();
        assert!(!pos.negated);
        assert_eq!(pos.predicate, "pred");
    }
    
    #[test]
    fn test_literal_negate() {
        let lit = Literal::new("pred".to_string(), vec!["a".to_string()], false);
        let neg = lit.negate();
        assert!(neg.negated);
    }
    
    #[test]
    fn test_simplify_removes_duplicates() {
        let axiom = PropositionalAxiom::new(
            "test".to_string(),
            vec![
                Literal::new("p".to_string(), vec![], false),
                Literal::new("p".to_string(), vec![], false), // duplicate
            ],
            Literal::new("q".to_string(), vec![], false),
        );
        let simplified = simplify(&[axiom]);
        assert_eq!(simplified.len(), 1);
        assert_eq!(simplified[0].condition.len(), 1);
    }
    
    #[test]
    fn test_empty_condition_dominates() {
        let axiom1 = PropositionalAxiom::new(
            "test".to_string(),
            vec![],
            Literal::new("q".to_string(), vec![], false),
        );
        let axiom2 = PropositionalAxiom::new(
            "test".to_string(),
            vec![Literal::new("p".to_string(), vec![], false)],
            Literal::new("q".to_string(), vec![], false),
        );
        let simplified = simplify(&[axiom1, axiom2]);
        assert_eq!(simplified.len(), 1);
        assert!(simplified[0].condition.is_empty());
    }
}
