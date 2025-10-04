use crate::translate::constraints::ConstraintSystem;
use crate::translate::pddl_ast::Condition;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InvariantPart {
    pub predicate: String,
    pub order: Vec<usize>,
    pub omitted_pos: i32,
}

impl InvariantPart {
    pub fn new(predicate: String, order: Vec<usize>, omitted_pos: i32) -> Self { Self { predicate, order, omitted_pos } }
    pub fn arity(&self) -> usize { self.order.len() }
    pub fn get_parameters(&self, atom: &crate::translate::pddl_ast::Condition) -> Vec<String> {
        // expect atom to be Condition::Atom(name, args)
        if let Condition::Atom(_, args) = atom {
            self.order.iter().map(|&p| args[p].clone()).collect()
        } else { vec![] }
    }
    pub fn instantiate(&self, parameters: &[String]) -> String {
        let mut args = vec!["?X".to_string(); self.order.len() + if self.omitted_pos != -1 {1} else {0}];
        for (arg, &pos) in parameters.iter().zip(self.order.iter()) { args[pos] = arg.clone(); }
        format!("{}({})", self.predicate, args.join(", "))
    }
    pub fn get_assignment(&self, parameters: &[String], literal: &crate::translate::pddl_ast::Condition) -> crate::translate::constraints::Assignment {
        // Build equalities: [(param_name, literal_arg_at_position), ...]
        let mut equalities: Vec<(String, String)> = Vec::new();
        if let Condition::Atom(_, args) = literal {
            for (arg, &pos) in parameters.iter().zip(self.order.iter()) {
                let lit_arg = args[pos].clone();
                equalities.push((arg.clone(), lit_arg));
            }
        }
        crate::translate::constraints::Assignment::new(equalities)
    }
    
    // Placeholder for possible_matches: returns empty list until full mapping
    // logic is implemented. Signature mirrors Python: given another literal,
    // produce InvariantPart candidates that would match the other literal.
    pub fn possible_matches(&self, _own_literal: &crate::translate::pddl_ast::Condition, _other_literal: &crate::translate::pddl_ast::Condition) -> Vec<InvariantPart> {
        Vec::new()
    }

    // Rough equality check used in some refinement heuristics: compare parameters
    pub fn matches(&self, other: &InvariantPart, own_literal: &crate::translate::pddl_ast::Condition, other_literal: &crate::translate::pddl_ast::Condition) -> bool {
        self.get_parameters(own_literal) == other.get_parameters(other_literal)
    }
}

#[derive(Clone, Debug)]
pub struct Invariant {
    pub parts: Vec<InvariantPart>,
    pub predicates: HashSet<String>,
}

impl Invariant {
    pub fn new(parts: Vec<InvariantPart>) -> Self {
        let mut preds = HashSet::new();
        for p in &parts { preds.insert(p.predicate.clone()); }
        Invariant { parts, predicates: preds }
    }
    pub fn get_parameters_for_atom(&self, atom: &crate::translate::pddl_ast::Condition) -> Vec<String> {
        // find the part corresponding to atom.predicate and return parameters
        if let Condition::Atom(pred, _args) = atom {
            for part in &self.parts {
                if &part.predicate == pred {
                    return part.get_parameters(atom);
                }
            }
        }
        Vec::new()
    }
    pub fn get_covering_assignments(&self, parameters: &[String], atom: &crate::translate::pddl_ast::Condition) -> Vec<crate::translate::constraints::Assignment> {
        // assume each predicate appears at most once in invariant.parts
        if let Condition::Atom(pred, _args) = atom {
            for part in &self.parts {
                if &part.predicate == pred {
                    return vec![part.get_assignment(parameters, atom)];
                }
            }
        }
        Vec::new()
    }
    pub fn instantiate(&self, parameters: &[String]) -> Vec<String> {
        self.parts.iter().map(|part| part.instantiate(parameters)).collect()
    }
}

use std::hash::{Hash, Hasher};
impl PartialEq for Invariant {
    fn eq(&self, other: &Self) -> bool {
        // Compare parts as sets: same parts implies equality
        let mut a = self.parts.clone();
        let mut b = other.parts.clone();
        a.sort_by(|x,y| x.predicate.cmp(&y.predicate));
        b.sort_by(|x,y| x.predicate.cmp(&y.predicate));
        a == b
    }
}
impl Eq for Invariant {}
impl Hash for Invariant {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash parts sorted by predicate to get deterministic hash
        let mut parts = self.parts.clone();
        parts.sort_by(|x,y| x.predicate.cmp(&y.predicate));
        for p in parts {
            p.predicate.hash(state);
            p.order.hash(state);
            p.omitted_pos.hash(state);
        }
    }
}

// helper: extract literal Conditions from a Condition (literal or conjunction)
pub fn get_literals(cond: &Condition) -> Vec<Condition> {
    match cond {
        Condition::Atom(_, _) => vec![cond.clone()],
        Condition::And(v) => v.iter().filter(|c| matches!(c, Condition::Atom(_, _))).cloned().collect(),
        _ => vec![],
    }
}

pub fn ensure_conjunction_sat(_system: &mut ConstraintSystem, parts: &[Vec<Condition>]) {
    // Simplified: map positive/negative atoms by predicate name into vectors.
    let mut pos: HashMap<String, Vec<Condition>> = HashMap::new();
    for part in parts.iter().flatten() {
        match part {
            Condition::Comparison(_, _, _) => { /* comparisons handled elsewhere */ }
            Condition::Atom(name, _args) => { pos.entry(name.clone()).or_default().push(part.clone()); }
            _ => {}
        }
    }
    // No-op further translation for now; the full implementation will convert
    // these into ConstraintSystem negative clauses and assignments.
}

pub fn ensure_cover(system: &mut ConstraintSystem, literal: &crate::translate::pddl_ast::Condition, invariant: &Invariant, inv_vars: &[String]) {
    // Convert to assignment(s) and add to the system.
    let assignments = invariant.get_covering_assignments(inv_vars, literal);
    for a in assignments {
        system.add_assignment_disjunction(vec![a]);
    }
}

pub fn ensure_inequality(system: &mut ConstraintSystem, lit1: &crate::translate::pddl_ast::Condition, lit2: &crate::translate::pddl_ast::Condition) {
    // If both are atoms and have parts, add a NegativeClause with paired positions
    if let (Condition::Atom(_, args1), Condition::Atom(_, args2)) = (lit1, lit2) {
        let mut parts: Vec<(String, String)> = Vec::new();
        let len = std::cmp::min(args1.len(), args2.len());
        for i in 0..len {
            parts.push((args1[i].clone(), args2[i].clone()));
        }
        if !parts.is_empty() {
            system.add_negative_clause(crate::translate::constraints::NegativeClause::new(parts));
        }
    }
}

pub fn invert_list(list: &[String]) -> HashMap<String, Vec<usize>> {
    let mut result: HashMap<String, Vec<usize>> = HashMap::new();
    for (pos, arg) in list.iter().enumerate() {
        result.entry(arg.clone()).or_default().push(pos);
    }
    result
}
