/// Port of pddl/conditions.py
/// Full condition hierarchy for PDDL conditions.
use std::collections::{HashMap, HashSet};
use std::fmt;

/// The root condition enum, mirroring Python's Condition class hierarchy.
/// Python used class inheritance; Rust uses an enum.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Condition {
    Truth,
    Falsity,
    Conjunction(Conjunction),
    Disjunction(Disjunction),
    UniversalCondition(UniversalCondition),
    ExistentialCondition(ExistentialCondition),
    Atom(Atom),
    NegatedAtom(NegatedAtom),
    FunctionComparison(FunctionComparison),
    NegatedFunctionComparison(NegatedFunctionComparison),
}

// Type aliases for cleaner references
pub type Truth = ();
pub type Falsity = ();

/// Helper enum for ConstantCondition
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstantCondition {
    Truth,
    Falsity,
}

// ----- Conjunction -----
/// Python: class Conjunction(JunctorCondition)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Conjunction {
    pub parts: Vec<Condition>,
}

impl Conjunction {
    pub fn new(parts: Vec<Condition>) -> Self {
        Conjunction { parts }
    }
}

// ----- Disjunction -----
/// Python: class Disjunction(JunctorCondition)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Disjunction {
    pub parts: Vec<Condition>,
}

impl Disjunction {
    pub fn new(parts: Vec<Condition>) -> Self {
        Disjunction { parts }
    }
}

// ----- UniversalCondition -----
/// Python: class UniversalCondition(QuantifiedCondition)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UniversalCondition {
    pub parameters: Vec<super::pddl_types::TypedObject>,
    pub parts: Vec<Condition>,
}

impl UniversalCondition {
    pub fn new(parameters: Vec<super::pddl_types::TypedObject>, parts: Vec<Condition>) -> Self {
        UniversalCondition { parameters, parts }
    }
}

// ----- ExistentialCondition -----
/// Python: class ExistentialCondition(QuantifiedCondition)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExistentialCondition {
    pub parameters: Vec<super::pddl_types::TypedObject>,
    pub parts: Vec<Condition>,
}

impl ExistentialCondition {
    pub fn new(parameters: Vec<super::pddl_types::TypedObject>, parts: Vec<Condition>) -> Self {
        ExistentialCondition { parameters, parts }
    }
}

// ----- Literal (base for Atom / NegatedAtom) -----

/// Python: class Atom(Literal): negated = False
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Atom {
    pub predicate: String,
    pub args: Vec<String>,
}

impl Atom {
    pub fn new(predicate: String, args: Vec<String>) -> Self {
        Atom { predicate, args }
    }

    /// Python: def negate(self) -> NegatedAtom
    pub fn negate(&self) -> NegatedAtom {
        NegatedAtom {
            predicate: self.predicate.clone(),
            args: self.args.clone(),
        }
    }

    /// Python: def positive(self) -> Atom
    pub fn positive(&self) -> Atom {
        self.clone()
    }
}

impl fmt::Display for Atom {
    /// Python: def __str__(self) -> "Atom %s(%s)"
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Atom {}({})", self.predicate, self.args.join(", "))
    }
}

/// Python: class NegatedAtom(Literal): negated = True
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NegatedAtom {
    pub predicate: String,
    pub args: Vec<String>,
}

impl NegatedAtom {
    pub fn new(predicate: String, args: Vec<String>) -> Self {
        NegatedAtom { predicate, args }
    }

    /// Python: def negate(self) -> Atom
    pub fn negate(&self) -> Atom {
        Atom {
            predicate: self.predicate.clone(),
            args: self.args.clone(),
        }
    }

    /// Python: def positive(self) -> Atom
    pub fn positive(&self) -> Atom {
        Atom {
            predicate: self.predicate.clone(),
            args: self.args.clone(),
        }
    }
}

impl fmt::Display for NegatedAtom {
    /// Python: def __str__(self) -> "NegatedAtom %s(%s)"
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NegatedAtom {}({})",
            self.predicate,
            self.args.join(", ")
        )
    }
}

/// Unified Literal type that wraps both Atom and NegatedAtom
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Literal {
    Positive(Atom),
    Negative(NegatedAtom),
}

impl Literal {
    pub fn predicate(&self) -> &str {
        match self {
            Literal::Positive(a) => &a.predicate,
            Literal::Negative(a) => &a.predicate,
        }
    }

    pub fn args(&self) -> &[String] {
        match self {
            Literal::Positive(a) => &a.args,
            Literal::Negative(a) => &a.args,
        }
    }

    pub fn is_negated(&self) -> bool {
        matches!(self, Literal::Negative(_))
    }

    pub fn negate(&self) -> Literal {
        match self {
            Literal::Positive(a) => Literal::Negative(a.negate()),
            Literal::Negative(a) => Literal::Positive(a.negate()),
        }
    }

    pub fn positive(&self) -> Atom {
        match self {
            Literal::Positive(a) => a.clone(),
            Literal::Negative(a) => a.positive(),
        }
    }

    pub fn as_atom(&self) -> Option<&Atom> {
        match self {
            Literal::Positive(a) => Some(a),
            Literal::Negative(_) => None,
        }
    }

    pub fn as_negated_atom(&self) -> Option<&NegatedAtom> {
        match self {
            Literal::Positive(_) => None,
            Literal::Negative(a) => Some(a),
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Positive(a) => write!(f, "{}", a),
            Literal::Negative(a) => write!(f, "{}", a),
        }
    }
}

// ----- FunctionComparison -----
/// Python: class FunctionComparison(Condition)
/// comparator is one of "<", "<=", "=", ">=", ">"
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionComparison {
    pub comparator: String,
    pub parts: Vec<super::f_expression::FunctionalExpression>,
    pub negated: bool,
}

impl FunctionComparison {
    pub fn new(comparator: String, parts: Vec<super::f_expression::FunctionalExpression>) -> Self {
        FunctionComparison {
            comparator,
            parts,
            negated: false,
        }
    }

    /// Python: def negate(self) -> NegatedFunctionComparison
    pub fn negate(&self) -> NegatedFunctionComparison {
        NegatedFunctionComparison {
            comparator: self.comparator.clone(),
            parts: self.parts.clone(),
            negated: true,
        }
    }
}

impl fmt::Display for FunctionComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FunctionComparison({}, {:?})",
            self.comparator, self.parts
        )
    }
}

/// Python: class NegatedFunctionComparison(FunctionComparison)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NegatedFunctionComparison {
    pub comparator: String,
    pub parts: Vec<super::f_expression::FunctionalExpression>,
    pub negated: bool,
}

impl NegatedFunctionComparison {
    pub fn new(comparator: String, parts: Vec<super::f_expression::FunctionalExpression>) -> Self {
        NegatedFunctionComparison {
            comparator,
            parts,
            negated: true,
        }
    }

    /// Python: def negate(self) -> FunctionComparison
    pub fn negate(&self) -> FunctionComparison {
        FunctionComparison {
            comparator: self.comparator.clone(),
            parts: self.parts.clone(),
            negated: false,
        }
    }
}

impl fmt::Display for NegatedFunctionComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NegatedFunctionComparison({}, {:?})",
            self.comparator, self.parts
        )
    }
}

// =========================================================================
// Methods on Condition enum (Python's polymorphic dispatch)
// =========================================================================

impl Condition {
    /// Python: def simplified(self)
    pub fn simplified(&self) -> Condition {
        match self {
            Condition::Truth => Condition::Truth,
            Condition::Falsity => Condition::Falsity,
            Condition::Conjunction(conj) => {
                let mut result_parts: Vec<Condition> = vec![];
                for p in conj.parts.iter().map(|p| p.simplified()) {
                    match p {
                        Condition::Conjunction(inner) => {
                            result_parts.extend(inner.parts);
                        }
                        Condition::Falsity => return Condition::Falsity,
                        Condition::Truth => {} // skip
                        other => result_parts.push(other),
                    }
                }
                if result_parts.is_empty() {
                    Condition::Truth
                } else if result_parts.len() == 1 {
                    result_parts.into_iter().next().unwrap()
                } else {
                    Condition::Conjunction(Conjunction::new(result_parts))
                }
            }
            Condition::Disjunction(disj) => {
                let mut result_parts: Vec<Condition> = vec![];
                for p in disj.parts.iter().map(|p| p.simplified()) {
                    match p {
                        Condition::Disjunction(inner) => {
                            result_parts.extend(inner.parts);
                        }
                        Condition::Truth => return Condition::Truth,
                        Condition::Falsity => {} // skip
                        other => result_parts.push(other),
                    }
                }
                if result_parts.is_empty() {
                    Condition::Falsity
                } else if result_parts.len() == 1 {
                    result_parts.into_iter().next().unwrap()
                } else {
                    Condition::Disjunction(Disjunction::new(result_parts))
                }
            }
            Condition::UniversalCondition(uc) => {
                let new_parts: Vec<Condition> = uc.parts.iter().map(|p| p.simplified()).collect();
                // Python: if isinstance(parts[0], ConstantCondition): return parts[0]
                if new_parts.len() == 1
                    && matches!(&new_parts[0], Condition::Truth | Condition::Falsity)
                {
                    new_parts.into_iter().next().unwrap()
                } else {
                    Condition::UniversalCondition(UniversalCondition::new(
                        uc.parameters.clone(),
                        new_parts,
                    ))
                }
            }
            Condition::ExistentialCondition(ec) => {
                let new_parts: Vec<Condition> = ec.parts.iter().map(|p| p.simplified()).collect();
                // Python: if isinstance(parts[0], ConstantCondition): return parts[0]
                if new_parts.len() == 1
                    && matches!(&new_parts[0], Condition::Truth | Condition::Falsity)
                {
                    new_parts.into_iter().next().unwrap()
                } else {
                    Condition::ExistentialCondition(ExistentialCondition::new(
                        ec.parameters.clone(),
                        new_parts,
                    ))
                }
            }
            // Atoms, NegatedAtoms, FunctionComparisons are already simplified
            other => other.clone(),
        }
    }

    /// Python: def relaxed(self)
    pub fn relaxed(&self) -> Condition {
        match self {
            Condition::Truth => Condition::Truth,
            Condition::Falsity => Condition::Falsity,
            Condition::Conjunction(conj) => Condition::Conjunction(Conjunction::new(
                conj.parts.iter().map(|p| p.relaxed()).collect(),
            )),
            Condition::Disjunction(disj) => Condition::Disjunction(Disjunction::new(
                disj.parts.iter().map(|p| p.relaxed()).collect(),
            )),
            Condition::UniversalCondition(uc) => {
                Condition::UniversalCondition(UniversalCondition::new(
                    uc.parameters.clone(),
                    uc.parts.iter().map(|p| p.relaxed()).collect(),
                ))
            }
            Condition::ExistentialCondition(ec) => {
                Condition::ExistentialCondition(ExistentialCondition::new(
                    ec.parameters.clone(),
                    ec.parts.iter().map(|p| p.relaxed()).collect(),
                ))
            }
            // NegatedAtom relaxes to Truth
            Condition::NegatedAtom(_) => Condition::Truth,
            // Everything else stays the same
            other => other.clone(),
        }
    }

    /// Python: def untyped(self)
    pub fn untyped(&self) -> Condition {
        // Replaces typed quantifiers with untyped ones by adding type predicates.
        match self {
            Condition::UniversalCondition(uc) => {
                let type_lits: Vec<Condition> = uc
                    .parameters
                    .iter()
                    .map(|p| {
                        let atom = p.get_atom();
                        Condition::NegatedAtom(atom.negate())
                    })
                    .collect();
                let mut parts: Vec<Condition> = type_lits;
                parts.extend(uc.parts.iter().map(|p| p.untyped()));
                Condition::UniversalCondition(UniversalCondition::new(
                    uc.parameters.clone(),
                    vec![Condition::Disjunction(Disjunction::new(parts))],
                ))
            }
            Condition::ExistentialCondition(ec) => {
                let type_lits: Vec<Condition> = ec
                    .parameters
                    .iter()
                    .map(|p| Condition::Atom(p.get_atom()))
                    .collect();
                let mut parts: Vec<Condition> = type_lits;
                parts.extend(ec.parts.iter().map(|p| p.untyped()));
                Condition::ExistentialCondition(ExistentialCondition::new(
                    ec.parameters.clone(),
                    vec![Condition::Conjunction(Conjunction::new(parts))],
                ))
            }
            Condition::Conjunction(conj) => Condition::Conjunction(Conjunction::new(
                conj.parts.iter().map(|p| p.untyped()).collect(),
            )),
            Condition::Disjunction(disj) => Condition::Disjunction(Disjunction::new(
                disj.parts.iter().map(|p| p.untyped()).collect(),
            )),
            other => other.clone(),
        }
    }

    /// Python: def uniquify_variables(self, type_map, renamings)
    pub fn uniquify_variables(
        &self,
        type_map: &mut HashMap<String, usize>,
        renamings: &mut HashMap<String, String>,
    ) -> Condition {
        match self {
            Condition::UniversalCondition(uc) => {
                let mut new_params = uc.parameters.clone();
                for p in &mut new_params {
                    p.uniquify_name(type_map, renamings);
                }
                let new_parts = uc
                    .parts
                    .iter()
                    .map(|p| p.uniquify_variables(type_map, renamings))
                    .collect();
                Condition::UniversalCondition(UniversalCondition::new(new_params, new_parts))
            }
            Condition::ExistentialCondition(ec) => {
                let mut new_params = ec.parameters.clone();
                for p in &mut new_params {
                    p.uniquify_name(type_map, renamings);
                }
                let new_parts = ec
                    .parts
                    .iter()
                    .map(|p| p.uniquify_variables(type_map, renamings))
                    .collect();
                Condition::ExistentialCondition(ExistentialCondition::new(new_params, new_parts))
            }
            Condition::Conjunction(conj) => Condition::Conjunction(Conjunction::new(
                conj.parts
                    .iter()
                    .map(|p| p.uniquify_variables(type_map, renamings))
                    .collect(),
            )),
            Condition::Disjunction(disj) => Condition::Disjunction(Disjunction::new(
                disj.parts
                    .iter()
                    .map(|p| p.uniquify_variables(type_map, renamings))
                    .collect(),
            )),
            Condition::Atom(atom) => {
                let new_args = atom
                    .args
                    .iter()
                    .map(|a| renamings.get(a).cloned().unwrap_or_else(|| a.clone()))
                    .collect();
                Condition::Atom(Atom::new(atom.predicate.clone(), new_args))
            }
            Condition::NegatedAtom(natom) => {
                let new_args = natom
                    .args
                    .iter()
                    .map(|a| renamings.get(a).cloned().unwrap_or_else(|| a.clone()))
                    .collect();
                Condition::NegatedAtom(NegatedAtom::new(natom.predicate.clone(), new_args))
            }
            Condition::FunctionComparison(fc) => {
                let new_parts = fc
                    .parts
                    .iter()
                    .map(|p| p.rename_variables(renamings))
                    .collect();
                Condition::FunctionComparison(FunctionComparison::new(
                    fc.comparator.clone(),
                    new_parts,
                ))
            }
            Condition::NegatedFunctionComparison(nfc) => {
                let new_parts = nfc
                    .parts
                    .iter()
                    .map(|p| p.rename_variables(renamings))
                    .collect();
                Condition::NegatedFunctionComparison(NegatedFunctionComparison::new(
                    nfc.comparator.clone(),
                    new_parts,
                ))
            }
            other => other.clone(),
        }
    }

    /// Python: def free_variables(self)
    pub fn free_variables(&self) -> HashSet<String> {
        match self {
            Condition::Truth | Condition::Falsity => HashSet::new(),
            Condition::Conjunction(conj) => {
                let mut result = HashSet::new();
                for p in &conj.parts {
                    result.extend(p.free_variables());
                }
                result
            }
            Condition::Disjunction(disj) => {
                let mut result = HashSet::new();
                for p in &disj.parts {
                    result.extend(p.free_variables());
                }
                result
            }
            Condition::UniversalCondition(uc) => {
                let mut result = HashSet::new();
                for p in &uc.parts {
                    result.extend(p.free_variables());
                }
                for param in &uc.parameters {
                    result.remove(&param.name);
                }
                result
            }
            Condition::ExistentialCondition(ec) => {
                let mut result = HashSet::new();
                for p in &ec.parts {
                    result.extend(p.free_variables());
                }
                for param in &ec.parameters {
                    result.remove(&param.name);
                }
                result
            }
            Condition::Atom(atom) => atom
                .args
                .iter()
                .filter(|a| a.starts_with('?'))
                .cloned()
                .collect(),
            Condition::NegatedAtom(natom) => natom
                .args
                .iter()
                .filter(|a| a.starts_with('?'))
                .cloned()
                .collect(),
            Condition::FunctionComparison(fc) => {
                let mut result = HashSet::new();
                for p in &fc.parts {
                    result.extend(p.free_variables());
                }
                result
            }
            Condition::NegatedFunctionComparison(nfc) => {
                let mut result = HashSet::new();
                for p in &nfc.parts {
                    result.extend(p.free_variables());
                }
                result
            }
        }
    }

    /// Python: def has_disjunction(self)
    pub fn has_disjunction(&self) -> bool {
        match self {
            Condition::Disjunction(_) => true,
            Condition::Conjunction(conj) => conj.parts.iter().any(|p| p.has_disjunction()),
            Condition::UniversalCondition(uc) => uc.parts.iter().any(|p| p.has_disjunction()),
            Condition::ExistentialCondition(ec) => ec.parts.iter().any(|p| p.has_disjunction()),
            _ => false,
        }
    }

    /// Python: def has_existential_part(self)
    pub fn has_existential_part(&self) -> bool {
        match self {
            Condition::ExistentialCondition(_) => true,
            Condition::Conjunction(conj) => conj.parts.iter().any(|p| p.has_existential_part()),
            Condition::Disjunction(disj) => disj.parts.iter().any(|p| p.has_existential_part()),
            Condition::UniversalCondition(uc) => uc.parts.iter().any(|p| p.has_existential_part()),
            _ => false,
        }
    }

    /// Python: def has_universal_part(self)
    pub fn has_universal_part(&self) -> bool {
        match self {
            Condition::UniversalCondition(_) => true,
            Condition::Conjunction(conj) => conj.parts.iter().any(|p| p.has_universal_part()),
            Condition::Disjunction(disj) => disj.parts.iter().any(|p| p.has_universal_part()),
            Condition::ExistentialCondition(ec) => ec.parts.iter().any(|p| p.has_universal_part()),
            _ => false,
        }
    }

    /// Check if this is an Atom
    pub fn is_atom(&self) -> bool {
        matches!(self, Condition::Atom(_))
    }

    /// Check if this is a NegatedAtom
    pub fn is_negated_atom(&self) -> bool {
        matches!(self, Condition::NegatedAtom(_))
    }

    /// Check if this is a Literal (Atom or NegatedAtom)
    pub fn is_literal(&self) -> bool {
        matches!(self, Condition::Atom(_) | Condition::NegatedAtom(_))
    }

    /// Check if this condition is negated (NegatedAtom or NegatedFunctionComparison)
    pub fn is_negated(&self) -> bool {
        matches!(
            self,
            Condition::NegatedAtom(_) | Condition::NegatedFunctionComparison(_)
        )
    }

    /// Get the Atom if this is Condition::Atom
    pub fn as_atom(&self) -> Option<&Atom> {
        match self {
            Condition::Atom(a) => Some(a),
            _ => None,
        }
    }

    /// Get the NegatedAtom if this is Condition::NegatedAtom
    pub fn as_negated_atom(&self) -> Option<&NegatedAtom> {
        match self {
            Condition::NegatedAtom(a) => Some(a),
            _ => None,
        }
    }

    /// Get the Conjunction if this is Condition::Conjunction
    pub fn as_conjunction(&self) -> Option<&Conjunction> {
        match self {
            Condition::Conjunction(c) => Some(c),
            _ => None,
        }
    }

    /// Get the predicate name if this is a literal
    pub fn literal_predicate(&self) -> Option<&str> {
        match self {
            Condition::Atom(a) => Some(&a.predicate),
            Condition::NegatedAtom(a) => Some(&a.predicate),
            _ => None,
        }
    }

    /// Get the arguments if this is a literal
    pub fn literal_args(&self) -> Option<&[String]> {
        match self {
            Condition::Atom(a) => Some(&a.args),
            Condition::NegatedAtom(a) => Some(&a.args),
            _ => None,
        }
    }

    /// Get positive version of a literal
    pub fn literal_positive(&self) -> Option<Atom> {
        match self {
            Condition::Atom(a) => Some(a.clone()),
            Condition::NegatedAtom(a) => Some(a.positive()),
            _ => None,
        }
    }

    /// Negate a literal condition
    pub fn negate_literal(&self) -> Option<Condition> {
        match self {
            Condition::Atom(a) => Some(Condition::NegatedAtom(a.negate())),
            Condition::NegatedAtom(a) => Some(Condition::Atom(a.negate())),
            _ => None,
        }
    }

    /// Python: def instantiate(self, var_mapping, init_facts, fluent_facts, init_function_vals, fluent_functions, task, new_axiom, new_modules, result)
    /// Instantiate the condition with a variable mapping.
    pub fn instantiate(
        &self,
        var_mapping: &HashMap<String, String>,
        init_facts: &HashSet<Atom>,
        fluent_facts: &HashSet<String>, // set of predicate names
        result: &mut Vec<Condition>,
    ) {
        match self {
            Condition::Truth => {}
            Condition::Falsity => {
                // This should signal unsatisfiable
                panic!("Cannot instantiate Falsity");
            }
            Condition::Conjunction(conj) => {
                for part in &conj.parts {
                    part.instantiate(var_mapping, init_facts, fluent_facts, result);
                }
            }
            Condition::Atom(atom) => {
                let new_args: Vec<String> = atom
                    .args
                    .iter()
                    .map(|a| var_mapping.get(a).cloned().unwrap_or_else(|| a.clone()))
                    .collect();
                let new_atom = Atom::new(atom.predicate.clone(), new_args);
                if fluent_facts.contains(&atom.predicate) {
                    result.push(Condition::Atom(new_atom));
                } else if !init_facts.contains(&new_atom) {
                    // Static fact not in init -> unsatisfiable
                    panic!("Static atom not in init: {}", new_atom);
                }
                // else: static fact in init -> always true, skip
            }
            Condition::NegatedAtom(natom) => {
                let new_args: Vec<String> = natom
                    .args
                    .iter()
                    .map(|a| var_mapping.get(a).cloned().unwrap_or_else(|| a.clone()))
                    .collect();
                let new_atom = Atom::new(natom.predicate.clone(), new_args.clone());
                if fluent_facts.contains(&natom.predicate) {
                    result.push(Condition::NegatedAtom(NegatedAtom::new(
                        natom.predicate.clone(),
                        new_args,
                    )));
                } else if init_facts.contains(&new_atom) {
                    // Static fact in init but we need it negated -> unsatisfiable
                    panic!("Static atom in init but needed negated: {}", new_atom);
                }
                // else: static fact not in init -> always false = negation is true, skip
            }
            other => {
                // FunctionComparison, etc. are handled separately in the full instantiate
                result.push(other.clone());
            }
        }
    }
}

impl fmt::Display for Condition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Condition::Truth => write!(f, "Truth"),
            Condition::Falsity => write!(f, "Falsity"),
            Condition::Conjunction(c) => {
                write!(f, "Conjunction([")?;
                for (i, p) in c.parts.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, "])")
            }
            Condition::Disjunction(d) => {
                write!(f, "Disjunction([")?;
                for (i, p) in d.parts.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, "])")
            }
            Condition::UniversalCondition(u) => {
                write!(f, "UniversalCondition({:?}, {:?})", u.parameters, u.parts)
            }
            Condition::ExistentialCondition(e) => {
                write!(f, "ExistentialCondition({:?}, {:?})", e.parameters, e.parts)
            }
            Condition::Atom(a) => write!(f, "{}", a),
            Condition::NegatedAtom(a) => write!(f, "{}", a),
            Condition::FunctionComparison(fc) => write!(f, "{}", fc),
            Condition::NegatedFunctionComparison(nfc) => write!(f, "{}", nfc),
        }
    }
}
