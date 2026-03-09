/// Port of constraints.py
/// Constraint system for invariant checking.

use std::collections::{HashMap, HashSet};

/// Python: class NegativeClause(object)
/// Represents a disjunction of inequalities: (v1 != v2) or (v3 != v4) or ...
#[derive(Debug, Clone)]
pub struct NegativeClause {
    pub parts: Vec<(String, String)>,
}

impl NegativeClause {
    pub fn new(parts: Vec<(String, String)>) -> Self {
        assert!(!parts.is_empty());
        NegativeClause { parts }
    }

    /// Python: def is_satisfiable(self)
    /// Returns true if at least one pair (v1, v2) has v1 != v2.
    pub fn is_satisfiable(&self) -> bool {
        for (v1, v2) in &self.parts {
            if v1 != v2 {
                return true;
            }
        }
        false
    }

    /// Python: def apply_mapping(self, m)
    pub fn apply_mapping(&self, mapping: &HashMap<String, String>) -> NegativeClause {
        let new_parts = self.parts.iter()
            .map(|(v1, v2)| {
                let new_v1 = mapping.get(v1).cloned().unwrap_or_else(|| v1.clone());
                let new_v2 = mapping.get(v2).cloned().unwrap_or_else(|| v2.clone());
                (new_v1, new_v2)
            })
            .collect();
        NegativeClause::new(new_parts)
    }
}

/// Python: class Assignment(object)
/// Represents a conjunction of equalities: (v1 = v2) and (v3 = v4) and ...
/// Uses union-find equivalence classes to compute a mapping.
#[derive(Debug, Clone)]
pub struct Assignment {
    pub equalities: Vec<(String, String)>,
    consistent: Option<bool>,
    mapping: Option<HashMap<String, String>>,
}

impl Assignment {
    pub fn new(equalities: Vec<(String, String)>) -> Self {
        Assignment {
            equalities,
            consistent: None,
            mapping: None,
        }
    }

    /// Python: def _compute_equivalence_classes(self)
    fn compute_equivalence_classes(&self) -> HashMap<String, HashSet<String>> {
        // Union-find style equivalence class computation
        let mut eq_classes: HashMap<String, HashSet<String>> = HashMap::new();

        for (v1, v2) in &self.equalities {
            let c1 = eq_classes.entry(v1.clone())
                .or_insert_with(|| {
                    let mut s = HashSet::new();
                    s.insert(v1.clone());
                    s
                })
                .clone();
            let c2 = eq_classes.entry(v2.clone())
                .or_insert_with(|| {
                    let mut s = HashSet::new();
                    s.insert(v2.clone());
                    s
                })
                .clone();

            // Check if they're already the same class (by pointer/content identity)
            if c1 == c2 && c1.contains(v2) {
                continue;
            }

            // Merge: always merge smaller into larger
            let (big, small) = if c1.len() >= c2.len() {
                (c1, c2)
            } else {
                (c2, c1)
            };

            let mut merged = big;
            merged.extend(small.iter().cloned());

            // Update all entries that point to either class
            for elem in merged.iter() {
                eq_classes.insert(elem.clone(), merged.clone());
            }
        }

        eq_classes
    }

    /// Python: def _compute_mapping(self)
    fn compute_mapping(&mut self) {
        let eq_classes = self.compute_equivalence_classes();

        let mut mapping = HashMap::new();
        let mut seen_classes: HashSet<Vec<String>> = HashSet::new();

        for eq_class in eq_classes.values() {
            let mut sorted_class: Vec<String> = eq_class.iter().cloned().collect();
            sorted_class.sort();
            if seen_classes.contains(&sorted_class) {
                continue;
            }
            seen_classes.insert(sorted_class);

            let variables: Vec<&String> = eq_class.iter()
                .filter(|item| item.starts_with('?'))
                .collect();
            let constants: Vec<&String> = eq_class.iter()
                .filter(|item| !item.starts_with('?'))
                .collect();

            if constants.len() >= 2 {
                self.consistent = Some(false);
                self.mapping = None;
                return;
            }

            let set_val = if !constants.is_empty() {
                constants[0].clone()
            } else {
                variables.iter().min().unwrap().to_string()
            };

            for entry in eq_class {
                mapping.insert(entry.clone(), set_val.clone());
            }
        }

        self.consistent = Some(true);
        self.mapping = Some(mapping);
    }

    /// Python: def is_consistent(self)
    pub fn is_consistent(&self) -> bool {
        if self.consistent.is_none() {
            // Need interior mutability or clone-and-compute
            let mut clone = self.clone();
            clone.compute_mapping();
            return clone.consistent.unwrap();
        }
        self.consistent.unwrap()
    }

    /// Python: def get_mapping(self)
    pub fn get_mapping(&self) -> HashMap<String, String> {
        if self.mapping.is_none() {
            let mut clone = self.clone();
            clone.compute_mapping();
            return clone.mapping.unwrap_or_default();
        }
        self.mapping.clone().unwrap_or_default()
    }
}

/// Python: class ConstraintSystem(object)
#[derive(Debug, Clone)]
pub struct ConstraintSystem {
    pub combinatorial_assignments: Vec<Vec<Assignment>>,
    pub neg_clauses: Vec<NegativeClause>,
}

impl ConstraintSystem {
    pub fn new() -> Self {
        ConstraintSystem {
            combinatorial_assignments: vec![],
            neg_clauses: vec![],
        }
    }

    /// Python: def _all_clauses_satisfiable(self, assignment)
    fn all_clauses_satisfiable(&self, assignment: &Assignment) -> bool {
        let mapping = assignment.get_mapping();
        for neg_clause in &self.neg_clauses {
            let clause = neg_clause.apply_mapping(&mapping);
            if !clause.is_satisfiable() {
                return false;
            }
        }
        true
    }

    /// Python: def _combine_assignments(self, assignments)
    fn combine_assignments(assignments: &[&Assignment]) -> Assignment {
        let mut new_equalities = vec![];
        for a in assignments {
            new_equalities.extend(a.equalities.clone());
        }
        Assignment::new(new_equalities)
    }

    /// Python: def add_assignment(self, assignment)
    pub fn add_assignment(&mut self, assignment: Assignment) {
        self.add_assignment_disjunction(vec![assignment]);
    }

    /// Python: def add_assignment_disjunction(self, assignments)
    pub fn add_assignment_disjunction(&mut self, assignments: Vec<Assignment>) {
        self.combinatorial_assignments.push(assignments);
    }

    /// Python: def add_negative_clause(self, clause)
    pub fn add_negative_clause(&mut self, clause: NegativeClause) {
        self.neg_clauses.push(clause);
    }

    /// Python: def combine(self, other)
    pub fn combine(&self, other: &ConstraintSystem) -> ConstraintSystem {
        let mut combined = ConstraintSystem::new();
        combined.combinatorial_assignments = self.combinatorial_assignments.clone();
        combined.combinatorial_assignments.extend(other.combinatorial_assignments.clone());
        combined.neg_clauses = self.neg_clauses.clone();
        combined.neg_clauses.extend(other.neg_clauses.clone());
        combined
    }

    /// Python: def copy(self)
    pub fn copy(&self) -> Self {
        let mut other = ConstraintSystem::new();
        other.combinatorial_assignments = self.combinatorial_assignments.clone();
        other.neg_clauses = self.neg_clauses.clone();
        other
    }

    /// Python: def is_solvable(self)
    pub fn is_solvable(&self) -> bool {
        // Cartesian product of combinatorial_assignments
        let combos = cartesian_product_refs(&self.combinatorial_assignments);
        for combo in &combos {
            let refs: Vec<&Assignment> = combo.iter().copied().collect();
            let combined = Self::combine_assignments(&refs);
            if !combined.is_consistent() {
                continue;
            }
            if self.all_clauses_satisfiable(&combined) {
                return true;
            }
        }
        false
    }
}

/// Cartesian product of assignment reference lists
fn cartesian_product_refs(lists: &[Vec<Assignment>]) -> Vec<Vec<&Assignment>> {
    if lists.is_empty() {
        return vec![vec![]];
    }

    let rest = cartesian_product_refs(&lists[1..]);
    let mut result = vec![];
    for item in &lists[0] {
        for seq in &rest {
            let mut combined = vec![item];
            combined.extend(seq.iter());
            result.push(combined);
        }
    }
    result
}
