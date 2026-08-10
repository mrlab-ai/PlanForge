/// Full condition hierarchy for PDDL conditions.
use std::collections::HashMap;
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

// ----- Conjunction -----
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Atom {
    pub predicate: String,
    pub args: Vec<String>,
}

impl Atom {
    pub fn new(predicate: String, args: Vec<String>) -> Self {
        Atom { predicate, args }
    }

    pub fn negate(&self) -> NegatedAtom {
        NegatedAtom {
            predicate: self.predicate.clone(),
            args: self.args.clone(),
        }
    }
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Atom {}({})", self.predicate, self.args.join(", "))
    }
}

/// Orders atoms the way comparing their `Debug` output does, without
/// formatting anything. The SAS variable order is this order, so it has to
/// stay exactly what it was when it was spelled `format!("{:?}", atom)`.
pub fn cmp_atoms(left: &Atom, right: &Atom) -> std::cmp::Ordering {
    crate::tools::cmp_quoted(&left.predicate, &right.predicate)
        .then_with(|| crate::tools::cmp_quoted_slice(&left.args, &right.args))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NegatedAtom {
    pub predicate: String,
    pub args: Vec<String>,
}

impl NegatedAtom {
    pub fn new(predicate: String, args: Vec<String>) -> Self {
        NegatedAtom { predicate, args }
    }

    pub fn negate(&self) -> Atom {
        Atom {
            predicate: self.predicate.clone(),
            args: self.args.clone(),
        }
    }

    pub fn positive(&self) -> Atom {
        Atom {
            predicate: self.predicate.clone(),
            args: self.args.clone(),
        }
    }
}

impl fmt::Display for NegatedAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NegatedAtom {}({})",
            self.predicate,
            self.args.join(", ")
        )
    }
}

// ----- FunctionComparison -----
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

    pub fn has_disjunction(&self) -> bool {
        match self {
            Condition::Disjunction(_) => true,
            Condition::Conjunction(conj) => conj.parts.iter().any(|p| p.has_disjunction()),
            Condition::UniversalCondition(uc) => uc.parts.iter().any(|p| p.has_disjunction()),
            Condition::ExistentialCondition(ec) => ec.parts.iter().any(|p| p.has_disjunction()),
            _ => false,
        }
    }

    pub fn has_existential_part(&self) -> bool {
        match self {
            Condition::ExistentialCondition(_) => true,
            Condition::Conjunction(conj) => conj.parts.iter().any(|p| p.has_existential_part()),
            Condition::Disjunction(disj) => disj.parts.iter().any(|p| p.has_existential_part()),
            Condition::UniversalCondition(uc) => uc.parts.iter().any(|p| p.has_existential_part()),
            _ => false,
        }
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
