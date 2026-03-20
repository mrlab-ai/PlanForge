use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct NegativeClause {
    pub parts: Vec<(String, String)>,
}

impl NegativeClause {
    pub fn new(parts: Vec<(String, String)>) -> Self {
        assert!(!parts.is_empty());
        Self { parts }
    }
    pub fn is_satisfiable(&self) -> bool {
        for (a, b) in &self.parts {
            if a != b {
                return true;
            }
        }
        false
    }
    pub fn apply_mapping(&self, m: &HashMap<String, String>) -> NegativeClause {
        let new_parts = self
            .parts
            .iter()
            .map(|(a, b)| {
                (
                    m.get(a).cloned().unwrap_or(a.clone()),
                    m.get(b).cloned().unwrap_or(b.clone()),
                )
            })
            .collect();
        NegativeClause::new(new_parts)
    }
}

#[derive(Clone, Debug)]
pub struct Assignment {
    pub equalities: Vec<(String, String)>,
    // cached
    pub consistent: Option<bool>,
    pub mapping: Option<HashMap<String, String>>,
    pub eq_classes: Option<HashMap<String, HashSet<String>>>,
}

impl Assignment {
    pub fn new(equalities: Vec<(String, String)>) -> Self {
        Self {
            equalities,
            consistent: None,
            mapping: None,
            eq_classes: None,
        }
    }
    fn compute_equivalence_classes(&mut self) {
        // union-find (disjoint set) over the variables/values
        let mut parent: HashMap<String, String> = HashMap::new();
        // initialize parents
        for (v1, v2) in &self.equalities {
            parent.entry(v1.clone()).or_insert_with(|| v1.clone());
            parent.entry(v2.clone()).or_insert_with(|| v2.clone());
        }

        fn find(parent: &mut HashMap<String, String>, x: &str) -> String {
            let mut cur = x.to_string();
            // find root
            while let Some(p) = parent.get(&cur) {
                if p == &cur {
                    break;
                }
                cur = p.clone();
            }
            let root = cur.clone();
            // path compression
            let mut node = x.to_string();
            while let Some(p) = parent.get(&node) {
                if p == &root {
                    break;
                }
                let next = p.clone();
                parent.insert(node.clone(), root.clone());
                node = next;
            }
            root
        }

        for (v1, v2) in &self.equalities {
            let r1 = find(&mut parent, v1);
            let r2 = find(&mut parent, v2);
            if r1 != r2 {
                parent.insert(r1.clone(), r2.clone());
            }
        }

        // collect classes
        let mut classes: HashMap<String, HashSet<String>> = HashMap::new();
        let keys: Vec<String> = parent.keys().cloned().collect();
        for key in keys.iter() {
            let root = find(&mut parent, key);
            classes
                .entry(root)
                .or_insert_with(HashSet::new)
                .insert(key.clone());
        }
        self.eq_classes = Some(classes);
    }
    fn compute_mapping(&mut self) {
        if self.eq_classes.is_none() {
            self.compute_equivalence_classes();
        }
        let eq_classes = self.eq_classes.as_ref().unwrap();
        if eq_classes.is_empty() {
            self.consistent = Some(true);
            self.mapping = Some(HashMap::new());
            return;
        }
        let mut mapping: HashMap<String, String> = HashMap::new();
        for eq in eq_classes.values() {
            // variables start with '?'
            let mut variables: Vec<String> =
                eq.iter().filter(|s| s.starts_with('?')).cloned().collect();
            let constants: Vec<String> =
                eq.iter().filter(|s| !s.starts_with('?')).cloned().collect();
            if constants.len() >= 2 {
                self.consistent = Some(false);
                self.mapping = None;
                return;
            }
            let set_val = if !constants.is_empty() {
                constants[0].clone()
            } else {
                variables.sort();
                variables[0].clone()
            };
            for entry in eq.iter() {
                mapping.insert(entry.clone(), set_val.clone());
            }
        }
        self.consistent = Some(true);
        self.mapping = Some(mapping);
    }
    pub fn is_consistent(&mut self) -> bool {
        if self.consistent.is_none() {
            self.compute_mapping();
        }
        self.consistent.unwrap_or(false)
    }
    pub fn get_mapping(&mut self) -> Option<HashMap<String, String>> {
        if self.consistent.is_none() {
            self.compute_mapping();
        }
        self.mapping.clone()
    }
}

#[derive(Clone, Debug)]
pub struct ConstraintSystem {
    pub combinatorial_assignments: Vec<Vec<Assignment>>,
    pub neg_clauses: Vec<NegativeClause>,
}

impl ConstraintSystem {
    pub fn new() -> Self {
        Self {
            combinatorial_assignments: Vec::new(),
            neg_clauses: Vec::new(),
        }
    }
    fn all_clauses_satisfiable(&self, assignment: &mut Assignment) -> bool {
        if let Some(mapping) = assignment.get_mapping() {
            for neg in &self.neg_clauses {
                let clause = neg.apply_mapping(&mapping);
                if !clause.is_satisfiable() {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }
    fn combine_assignments(&self, assignments: &[Assignment]) -> Assignment {
        let mut new_equalities = Vec::new();
        for a in assignments {
            for e in &a.equalities {
                new_equalities.push(e.clone());
            }
        }
        Assignment::new(new_equalities)
    }
    pub fn add_assignment(&mut self, a: Assignment) {
        self.add_assignment_disjunction(vec![a]);
    }
    pub fn add_assignment_disjunction(&mut self, assignments: Vec<Assignment>) {
        self.combinatorial_assignments.push(assignments);
    }
    pub fn add_negative_clause(&mut self, c: NegativeClause) {
        self.neg_clauses.push(c);
    }
    pub fn combine(&self, other: &ConstraintSystem) -> ConstraintSystem {
        let mut combined = ConstraintSystem::new();
        combined.combinatorial_assignments = [
            self.combinatorial_assignments.clone(),
            other.combinatorial_assignments.clone(),
        ]
        .concat();
        combined.neg_clauses = [self.neg_clauses.clone(), other.neg_clauses.clone()].concat();
        combined
    }
    pub fn copy(&self) -> ConstraintSystem {
        ConstraintSystem {
            combinatorial_assignments: self.combinatorial_assignments.clone(),
            neg_clauses: self.neg_clauses.clone(),
        }
    }
    pub fn is_solvable(&self) -> bool {
        // product over combinatorial_assignments
        if self.combinatorial_assignments.is_empty() {
            let mut combined = Assignment::new(Vec::new());
            if combined.is_consistent() {
                return self.all_clauses_satisfiable(&mut combined);
            }
            return false;
        }
        let mut indices = vec![0usize; self.combinatorial_assignments.len()];
        loop {
            let mut selected: Vec<Assignment> = Vec::new();
            for (i, choices) in self.combinatorial_assignments.iter().enumerate() {
                selected.push(choices[indices[i]].clone());
            }
            let mut combined = self.combine_assignments(&selected);
            if combined.is_consistent() {
                if self.all_clauses_satisfiable(&mut combined) {
                    return true;
                }
            }
            // increment indices
            let mut carry = true;
            for i in 0..indices.len() {
                if carry {
                    indices[i] += 1;
                    if indices[i] >= self.combinatorial_assignments[i].len() {
                        indices[i] = 0;
                        carry = true;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry {
                break;
            }
        }
        false
    }
}
