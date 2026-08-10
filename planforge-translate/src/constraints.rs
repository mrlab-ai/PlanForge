/// Constraint system for invariant checking.
use std::collections::HashMap;

/// The representative of `term`'s equivalence class under an assignment. A term
/// that none of the equalities mentions forms its own class, so a missing entry
/// stands for the term itself.
pub fn class_of<'a>(representative: &'a HashMap<String, String>, term: &'a str) -> &'a str {
    representative.get(term).map_or(term, String::as_str)
}

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

    /// Whether some pair of the clause falls into two different equivalence
    /// classes.
    fn is_satisfied_by(&self, representative: &HashMap<String, String>) -> bool {
        self.parts
            .iter()
            .any(|(v1, v2)| class_of(representative, v1) != class_of(representative, v2))
    }
}

/// The representative of every term of `equalities`: the object of its
/// equivalence class if there is one, otherwise the smallest of its variables.
///
/// `None` if some class holds two objects, in which case no substitution can
/// satisfy the conjunction.
fn compute_representatives(equalities: &[(String, String)]) -> Option<HashMap<String, String>> {
    // Union-find over the mentioned terms: `class` maps a term to the index of
    // its class in `classes`, and merging empties the class that is given up.
    let mut class: HashMap<&str, usize> = HashMap::new();
    let mut classes: Vec<Vec<&str>> = Vec::new();
    for (v1, v2) in equalities {
        for term in [v1.as_str(), v2.as_str()] {
            class.entry(term).or_insert_with(|| {
                classes.push(vec![term]);
                classes.len() - 1
            });
        }
        let (mut keep, mut given_up) = (class[v1.as_str()], class[v2.as_str()]);
        if keep == given_up {
            continue;
        }
        if classes[keep].len() < classes[given_up].len() {
            std::mem::swap(&mut keep, &mut given_up);
        }
        let merged = std::mem::take(&mut classes[given_up]);
        for term in &merged {
            class.insert(term, keep);
        }
        classes[keep].extend(merged);
    }

    let mut representative = HashMap::with_capacity(class.len());
    for terms in classes.iter().filter(|terms| !terms.is_empty()) {
        let mut objects = terms.iter().filter(|term| !term.starts_with('?'));
        let chosen = match objects.next() {
            Some(object) => {
                if objects.next().is_some() {
                    return None;
                }
                *object
            }
            None => terms.iter().min().copied().expect("classes are non-empty"),
        };
        for term in terms {
            representative.insert((*term).to_string(), chosen.to_string());
        }
    }
    Some(representative)
}

/// Represents a conjunction of equalities: (v1 = v2) and (v3 = v4) and ...
#[derive(Debug, Clone)]
pub struct Assignment {
    pub equalities: Vec<(String, String)>,
    /// The memoized result of `compute_representatives`, `None` until it is
    /// first asked for. The inner `None` means inconsistent, and is why an
    /// empty map -- the answer for a conjunction without equalities -- must not
    /// be conflated with an absent one.
    representative: Option<Option<HashMap<String, String>>>,
}

impl Assignment {
    pub fn new(equalities: Vec<(String, String)>) -> Self {
        Assignment {
            equalities,
            representative: None,
        }
    }

    pub fn is_consistent(&mut self) -> bool {
        self.representative().is_some()
    }

    pub fn representative(&mut self) -> Option<&HashMap<String, String>> {
        if self.representative.is_none() {
            self.representative = Some(compute_representatives(&self.equalities));
        }
        self.representative
            .as_ref()
            .expect("just computed")
            .as_ref()
    }
}

/// A conjunction of an equality DNF, a set of inequality disjunctions, and a
/// set of terms that may not become equivalent to an object.
///
/// The system is solvable if one `Assignment` can be picked from each entry of
/// `combinatorial_assignments` such that the finest equivalence relation
/// induced by all their equalities is consistent, leaves every `not_constant`
/// term in a class without an object, and satisfies every negative clause.
#[derive(Debug, Clone, Default)]
pub struct ConstraintSystem {
    pub combinatorial_assignments: Vec<Vec<Assignment>>,
    pub neg_clauses: Vec<NegativeClause>,
    not_constant: Vec<String>,
}

impl ConstraintSystem {
    pub fn new() -> Self {
        Self::default()
    }

    fn is_satisfied_by(&self, assignment: &mut Assignment) -> bool {
        let representative = assignment
            .representative()
            .expect("constraints are only checked against consistent assignments");
        self.not_constant
            .iter()
            .all(|term| class_of(representative, term).starts_with('?'))
            && self
                .neg_clauses
                .iter()
                .all(|clause| clause.is_satisfied_by(representative))
    }

    fn combine_assignments(assignments: &[&Assignment]) -> Assignment {
        Assignment::new(
            assignments
                .iter()
                .flat_map(|assignment| assignment.equalities.iter().cloned())
                .collect(),
        )
    }

    pub fn add_assignment(&mut self, assignment: Assignment) {
        self.add_assignment_disjunction(vec![assignment]);
    }

    pub fn add_assignment_disjunction(&mut self, assignments: Vec<Assignment>) {
        self.combinatorial_assignments.push(assignments);
    }

    pub fn add_negative_clause(&mut self, clause: NegativeClause) {
        self.neg_clauses.push(clause);
    }

    /// Forbids solutions that put `term` into the same equivalence class as an
    /// object.
    pub fn add_not_constant(&mut self, term: String) {
        self.not_constant.push(term);
    }

    pub fn extend(&mut self, other: &ConstraintSystem) {
        self.combinatorial_assignments
            .extend_from_slice(&other.combinatorial_assignments);
        self.neg_clauses.extend_from_slice(&other.neg_clauses);
        self.not_constant.extend_from_slice(&other.not_constant);
    }

    pub fn is_solvable(&self) -> bool {
        cartesian_product_refs(&self.combinatorial_assignments)
            .into_iter()
            .any(|combo| {
                let mut combined = Self::combine_assignments(&combo);
                combined.is_consistent() && self.is_satisfied_by(&mut combined)
            })
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
