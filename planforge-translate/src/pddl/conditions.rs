/// Full condition hierarchy for PDDL conditions.
use std::collections::HashMap;
use std::fmt;

use super::f_expression::FunctionalExpression;

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
    FunctionComparison(Comparison),
    NegatedFunctionComparison(Comparison),
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

    /// The same atom with its arguments put through `substitution`.
    pub fn substituted(&self, substitution: &impl super::Substitution) -> Atom {
        Atom::new(
            self.predicate.clone(),
            super::substitute(&self.args, substitution),
        )
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

/// A denied atom.
///
/// Field for field an [`Atom`], and deliberately still its own type rather than
/// an `Atom` payload the way [`Comparison`] is: the two differ in how they print,
/// and a negated atom's `Display` is what names the second value of a binary SAS
/// variable in `output.sas` (`build_translation_key`). Sharing the payload would
/// need a wrapper type to restore that name, which is more code than the shared
/// fields save.
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

    /// The same literal with its arguments put through `substitution`.
    pub fn substituted(&self, substitution: &impl super::Substitution) -> NegatedAtom {
        NegatedAtom::new(
            self.predicate.clone(),
            super::substitute(&self.args, substitution),
        )
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

// ----- Comparison -----

/// Two numeric expressions related by a comparator, one of `<`, `<=`, `=`, `>=`
/// and `>`.
///
/// Whether the comparison is asserted or denied is the [`Condition`] variant
/// holding it, and nothing else: the two used to be separate structs with the
/// same three fields, of which the third was a `negated` flag that duplicated
/// the variant and was never read. Negating a comparison is therefore moving the
/// same payload to the other variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Comparison {
    pub comparator: String,
    pub operands: Vec<FunctionalExpression>,
}

impl Comparison {
    pub fn new(comparator: String, operands: Vec<FunctionalExpression>) -> Self {
        Comparison {
            comparator,
            operands,
        }
    }
}

// =========================================================================
// Methods on Condition enum
// =========================================================================

impl Condition {
    /// The subconditions of a compound condition, in order.
    ///
    /// Empty for every condition that has none: a constant, a literal, and a
    /// comparison, whose operands are numeric expressions rather than
    /// conditions. Naming the structure once is what lets the traversals below
    /// -- and the normalization passes -- be written once instead of once per
    /// compound variant.
    pub fn parts(&self) -> &[Condition] {
        match self {
            Condition::Conjunction(conjunction) => &conjunction.parts,
            Condition::Disjunction(disjunction) => &disjunction.parts,
            Condition::UniversalCondition(universal) => &universal.parts,
            Condition::ExistentialCondition(existential) => &existential.parts,
            Condition::Truth
            | Condition::Falsity
            | Condition::Atom(_)
            | Condition::NegatedAtom(_)
            | Condition::FunctionComparison(_)
            | Condition::NegatedFunctionComparison(_) => &[],
        }
    }

    /// As [`Self::parts`], but taking the subconditions out of the condition
    /// rather than borrowing them.
    pub fn into_parts(self) -> Vec<Condition> {
        match self {
            Condition::Conjunction(conjunction) => conjunction.parts,
            Condition::Disjunction(disjunction) => disjunction.parts,
            Condition::UniversalCondition(universal) => universal.parts,
            Condition::ExistentialCondition(existential) => existential.parts,
            _ => Vec::new(),
        }
    }

    /// The same kind of condition over `parts`, keeping whatever else the
    /// condition carries -- the parameters a quantifier binds.
    ///
    /// A condition without subconditions has none to replace, so handing this
    /// one any is a caller bug rather than something to drop quietly.
    pub fn with_parts(&self, parts: Vec<Condition>) -> Condition {
        match self {
            Condition::Conjunction(_) => Condition::Conjunction(Conjunction::new(parts)),
            Condition::Disjunction(_) => Condition::Disjunction(Disjunction::new(parts)),
            Condition::UniversalCondition(universal) => Condition::UniversalCondition(
                UniversalCondition::new(universal.parameters.clone(), parts),
            ),
            Condition::ExistentialCondition(existential) => Condition::ExistentialCondition(
                ExistentialCondition::new(existential.parameters.clone(), parts),
            ),
            leaf => {
                assert!(parts.is_empty(), "{leaf} has no subconditions to replace");
                leaf.clone()
            }
        }
    }

    /// The same condition with `map` applied to each of its subconditions. A
    /// condition without subconditions is its own image.
    pub fn map_parts(&self, map: impl FnMut(&Condition) -> Condition) -> Condition {
        self.with_parts(self.parts().iter().map(map).collect())
    }

    /// The numeric operands a comparison relates, in order; empty for every
    /// condition that is not one.
    ///
    /// This is the other half of the structure [`Self::parts`] names: a
    /// comparison is a leaf of the condition tree and the root of a pair of
    /// expression trees, and both kinds of comparison hold theirs the same way.
    pub fn comparison_operands(&self) -> &[FunctionalExpression] {
        match self.as_comparison() {
            Some(comparison) => &comparison.operands,
            None => &[],
        }
    }

    /// The comparison this condition asserts or denies, if it is one.
    pub fn as_comparison(&self) -> Option<&Comparison> {
        match self {
            Condition::FunctionComparison(comparison)
            | Condition::NegatedFunctionComparison(comparison) => Some(comparison),
            _ => None,
        }
    }

    /// The comparator a comparison relates its operands by. Only a comparison
    /// has one, so asking anything else is a caller bug.
    pub fn comparator(&self) -> &str {
        &self
            .as_comparison()
            .unwrap_or_else(|| panic!("{self} is not a comparison"))
            .comparator
    }

    /// The same comparison with `map` applied to each of its operands. A
    /// condition that is not a comparison is its own image.
    pub fn map_comparison_operands(
        &self,
        map: impl FnMut(&FunctionalExpression) -> FunctionalExpression,
    ) -> Condition {
        let Some(comparison) = self.as_comparison() else {
            return self.clone();
        };
        let mapped = Comparison::new(
            comparison.comparator.clone(),
            comparison.operands.iter().map(map).collect(),
        );
        match self {
            Condition::NegatedFunctionComparison(_) => Condition::NegatedFunctionComparison(mapped),
            _ => Condition::FunctionComparison(mapped),
        }
    }

    pub fn simplified(&self) -> Condition {
        match self {
            Condition::Conjunction(_) => {
                self.simplified_connective(&Condition::Falsity, &Condition::Truth)
            }
            Condition::Disjunction(_) => {
                self.simplified_connective(&Condition::Truth, &Condition::Falsity)
            }
            // A quantifier whose whole body simplified to a constant is that
            // constant: there is nothing left for it to quantify over.
            Condition::UniversalCondition(_) | Condition::ExistentialCondition(_) => {
                let simplified = self.map_parts(Condition::simplified);
                match simplified.parts() {
                    [constant @ (Condition::Truth | Condition::Falsity)] => constant.clone(),
                    _ => simplified,
                }
            }
            // Constants, literals and comparisons are already simplified.
            leaf => leaf.clone(),
        }
    }

    /// The simplification the two connectives share: a nested connective of the
    /// same kind is spliced in, the `neutral` constant is dropped, and the
    /// `absorbing` constant swallows the whole connective. What is left of a
    /// connective over one part is that part, and over none it is `neutral`.
    ///
    /// A conjunction absorbs `Falsity` and is neutral on `Truth`; a disjunction
    /// is its dual, which is the only difference between the two.
    fn simplified_connective(&self, absorbing: &Condition, neutral: &Condition) -> Condition {
        let mut parts: Vec<Condition> = Vec::with_capacity(self.parts().len());
        for part in self.parts().iter().map(Condition::simplified) {
            if &part == absorbing {
                return absorbing.clone();
            }
            if &part == neutral {
                continue;
            }
            if std::mem::discriminant(&part) == std::mem::discriminant(self) {
                parts.append(&mut part.into_parts());
            } else {
                parts.push(part);
            }
        }
        match parts.len() {
            0 => neutral.clone(),
            1 => parts.pop().expect("length checked"),
            _ => self.with_parts(parts),
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
            Condition::Conjunction(_) | Condition::Disjunction(_) => {
                self.map_parts(|part| part.uniquify_variables(type_map, renamings))
            }
            Condition::Atom(atom) => Condition::Atom(atom.substituted(renamings)),
            Condition::NegatedAtom(natom) => Condition::NegatedAtom(natom.substituted(renamings)),
            Condition::FunctionComparison(_) | Condition::NegatedFunctionComparison(_) => {
                self.map_comparison_operands(|operand| operand.rename_variables(renamings))
            }
            other => other.clone(),
        }
    }

    pub fn has_disjunction(&self) -> bool {
        matches!(self, Condition::Disjunction(_))
            || self.parts().iter().any(Condition::has_disjunction)
    }

    pub fn has_existential_part(&self) -> bool {
        matches!(self, Condition::ExistentialCondition(_))
            || self.parts().iter().any(Condition::has_existential_part)
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
            Condition::FunctionComparison(comparison) => write!(
                f,
                "FunctionComparison({}, {:?})",
                comparison.comparator, comparison.operands
            ),
            Condition::NegatedFunctionComparison(comparison) => write!(
                f,
                "NegatedFunctionComparison({}, {:?})",
                comparison.comparator, comparison.operands
            ),
        }
    }
}
