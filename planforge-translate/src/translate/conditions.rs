//! Grounded conditions as SAS+ conditions: the assignments of facts to
//! variable values, multiplied out over the disjunctions a fact's negation
//! leaves, and numeric comparisons as the facts of their comparison axioms.

use super::*;

/// A partial assignment of values to SAS variables.
///
/// A condition constrains a handful of variables, so a linear scan beats
/// hashing a `usize`, which profiling showed to dominate the translation of
/// large tasks. Insertion order is preserved on purpose: Fast Downward's
/// dictionaries are ordered and the translation picks "the smallest" and
/// "the first" entry out of them, so a hashed port would make the SAS task
/// depend on the hash seed.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub(super) struct Assignment {
    pairs: Vec<(usize, usize)>,
}

impl Assignment {
    pub(super) fn get(&self, var: usize) -> Option<usize> {
        self.pairs
            .iter()
            .find_map(|&(other, value)| (other == var).then_some(value))
    }

    pub(super) fn set(&mut self, var: usize, value: usize) {
        match self.pairs.iter_mut().find(|(other, _)| *other == var) {
            Some(pair) => pair.1 = value,
            None => self.pairs.push((var, value)),
        }
    }

    pub(super) fn remove(&mut self, var: usize) {
        self.pairs.retain(|&(other, _)| other != var);
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.pairs.iter().copied()
    }

    pub(super) fn len(&self) -> usize {
        self.pairs.len()
    }

    pub(super) fn into_pairs(self) -> Vec<(usize, usize)> {
        self.pairs
    }

    pub(super) fn sorted_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = self.pairs.clone();
        pairs.sort_unstable();
        pairs
    }
}

/// The values each variable may still take while a condition is translated.
/// Almost every entry is a singleton; only a negative literal widens one.
#[derive(Default)]
pub(super) struct Domains {
    entries: Vec<(usize, Vec<usize>)>,
}

impl Domains {
    pub(super) fn get(&self, var: usize) -> Option<&[usize]> {
        self.entries
            .iter()
            .find_map(|(other, values)| (*other == var).then_some(values.as_slice()))
    }

    pub(super) fn set(&mut self, var: usize, values: Vec<usize>) {
        debug_assert!(!values.is_empty(), "a variable with no value is a conflict");
        match self.entries.iter_mut().find(|(other, _)| *other == var) {
            Some(entry) => entry.1 = values,
            None => self.entries.push((var, values)),
        }
    }

    pub(super) fn set_single(&mut self, var: usize, value: usize) {
        self.set(var, vec![value]);
    }

    pub(super) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ============================================================
// translate_strips_conditions_aux
// ============================================================

/// Translates a conjunction of ground literals into the finite-domain
/// conditions that entail it, or `None` if it is unsatisfiable.
pub(super) fn translate_strips_conditions_aux(
    translation: &mut Translation,
    conditions: &[Condition],
    mutex_check: bool,
) -> Option<Vec<Assignment>> {
    // The mutex pass encodes the same conditions against the full mutex
    // groups; everything else about the translation is shared.
    let (dictionary, ranges) = if mutex_check {
        (
            &mut translation.mutex_dictionary,
            &mut translation.mutex_ranges,
        )
    } else {
        (&mut translation.dictionary, &mut translation.ranges)
    };
    let numeric_dictionary = &translation.numeric;
    let constants = &translation.constants;
    let comp_axiom_dict = &mut translation.comparison_axioms;
    let sas_comp_axioms = &mut translation.sas_comparison_axioms;
    let mut condition = Domains::default();

    for fact in conditions {
        match fact {
            Condition::FunctionComparison(_) | Condition::NegatedFunctionComparison(_) => {
                let (comparator, parts_fexpr, negated) = (
                    fact.comparator(),
                    fact.comparison_operands(),
                    fact.is_negated(),
                );

                // Check if fact is already in dictionary
                let atom = condition_to_atom(fact);
                if let Some(pairs) = dictionary.get(&atom) {
                    let (var, val) = pairs[0];
                    if condition.get(var).is_some_and(|vals| !vals.contains(&val)) {
                        return None; // conflicting
                    }
                    condition.set_single(var, val);
                    continue;
                }

                // Build parts lookup - extract PNE from FunctionalExpression
                let parts: Vec<usize> = parts_fexpr
                    .iter()
                    .map(|p| match p {
                        FunctionalExpression::PrimitiveNumericExpression(pne) => {
                            *numeric_dictionary.get(pne).unwrap_or_else(|| {
                                panic!("PNE {:?} not in numeric dictionary", pne)
                            })
                        }
                        FunctionalExpression::NumericConstant(_)
                        | FunctionalExpression::ArithmeticExpression(_)
                        | FunctionalExpression::AdditiveInverse(_) => {
                            panic!("comparison operand {p} survived arithmetic normalization")
                        }
                    })
                    .collect();

                if let [left, right] = parts[..]
                    && let (Some(left), Some(right)) = (constants[left], constants[right])
                {
                    if comparison_holds(comparator, left, right) == negated {
                        return None;
                    }
                    continue;
                }

                let key = (comparator.to_owned(), parts.clone());

                if let Some(existing_fact) = comp_axiom_dict.get(&key) {
                    // Already have this comparison axiom
                    let lookup_fact = if negated {
                        negate_condition(existing_fact)
                    } else {
                        existing_fact.clone()
                    };
                    let atom = condition_to_atom(&lookup_fact);
                    if let Some(pairs) = dictionary.get(&atom) {
                        let (var, val) = pairs[0];
                        if condition.get(var).is_some_and(|vals| !vals.contains(&val)) {
                            return None;
                        }
                        condition.set_single(var, val);
                    }
                } else {
                    // Create new comparison axiom
                    let axiom =
                        SASCompareAxiom::new(comparator.to_owned(), parts.clone(), ranges.len());

                    // Create positive and negative atoms for lookup
                    let pos_fact = make_fc_condition(comparator, parts_fexpr, false);
                    let neg_fact = make_fc_condition(comparator, parts_fexpr, true);

                    let pos_atom = condition_to_atom(&pos_fact);
                    let neg_atom = condition_to_atom(&neg_fact);

                    if !mutex_check {
                        sas_comp_axioms.push(axiom);
                        comp_axiom_dict.insert(key, pos_fact.clone());
                    }

                    dictionary
                        .entry(pos_atom.clone())
                        .or_default()
                        .push((ranges.len(), 0));
                    dictionary
                        .entry(neg_atom.clone())
                        .or_default()
                        .push((ranges.len(), 1));
                    ranges.push(ConditionValue::DOMAIN_SIZE);

                    // Now use the fact
                    let lookup_fact = if negated { &neg_fact } else { &pos_fact };
                    let atom = condition_to_atom(lookup_fact);
                    if let Some(pairs) = dictionary.get(&atom) {
                        let (var, val) = pairs[0];
                        condition.set_single(var, val);
                    }
                }
            }
            Condition::Atom(atom) => {
                // In the mutex pass the dictionary holds only the atoms some
                // mutex group covers, so a miss means "unconstrained". In the
                // real pass every reachable atom has a variable, and
                // instantiation has already removed the statically true and
                // false ones, so a miss is a broken invariant: silently
                // dropping the condition would make the operator or axiom hold
                // in states where it must not.
                let Some(pairs) = dictionary.get(atom) else {
                    if !mutex_check {
                        let same_predicate: Vec<&Atom> = dictionary
                            .keys()
                            .filter(|other| other.predicate == atom.predicate)
                            .take(4)
                            .collect();
                        panic!(
                            "condition atom {atom:?} has no SAS variable; it is neither \
                             reachable nor statically decided. {} other atoms of that \
                             predicate do have one, for example {same_predicate:?}",
                            dictionary
                                .keys()
                                .filter(|other| other.predicate == atom.predicate)
                                .count()
                        );
                    }
                    continue;
                };
                for &(var, val) in pairs {
                    if condition.get(var).is_some_and(|vals| !vals.contains(&val)) {
                        return None; // conflicting
                    }
                    condition.set_single(var, val);
                }
            }
            Condition::NegatedAtom(_) => {
                // Handle negative conditions later
                continue;
            }
            // Comparisons are handled by their own arm above; they get a
            // condition variable of their own rather than a propositional one.
            Condition::Truth => continue,
            // Normalization leaves a condition a conjunction of literals and
            // comparisons, so anything compound here escaped a pass. Skipping it
            // drops a restriction and makes the operator or axiom hold in states
            // where it must not.
            other => panic!("condition {other} reached SAS translation un-normalized"),
        }
    }

    // Now handle negative conditions
    for fact in conditions {
        match fact {
            Condition::FunctionComparison(_) | Condition::NegatedFunctionComparison(_) => {
                continue; // Already handled
            }
            Condition::NegatedAtom(natom) => {
                let pos_atom = Atom::new(natom.predicate.clone(), natom.args.clone());
                let mut constrained_existing = false;
                let mut fresh = Domains::default();

                if let Some(pairs) = dictionary.get(&pos_atom) {
                    for &(var, val) in pairs {
                        match condition.get(var) {
                            Some(existing) => {
                                constrained_existing = true;
                                let intersection: Vec<usize> =
                                    existing.iter().copied().filter(|&v| v != val).collect();
                                if intersection.is_empty() {
                                    return None; // conflicting
                                }
                                condition.set(var, intersection);
                            }
                            None => {
                                fresh.set(var, (0..ranges[var]).filter(|&v| v != val).collect())
                            }
                        }
                    }
                }

                // A negative literal is satisfied as soon as one of the
                // variables covering it moves off the deleted value, so the
                // variable with the fewest remaining values is enough.
                if !constrained_existing && !fresh.is_empty() {
                    let (var, values) = fresh
                        .entries
                        .into_iter()
                        .min_by_key(|(_, values)| values.len())
                        .expect("checked non-empty");
                    condition.set(var, values);
                }
            }
            // This is the second pass over the same condition list, and it exists
            // only to handle the negative literals the first pass deferred. A
            // positive one is already in `condition`.
            Condition::Atom(_) => continue,
            // As above: comparisons have their own arm, Truth restricts nothing,
            // and nothing compound should have survived normalization.
            Condition::Truth => continue,
            other => panic!("condition {other} reached SAS translation un-normalized"),
        }
    }

    Some(multiply_out(condition))
}

/// Expands a condition that allows several values per variable into the flat
/// assignments it stands for, fewest alternatives first.
pub(super) fn multiply_out(condition: Domains) -> Vec<Assignment> {
    let mut entries = condition.entries;
    entries.sort_by_key(|(_, values)| values.len());

    let mut flat = vec![Assignment::default()];
    for (var, values) in entries {
        match values.as_slice() {
            [value] => {
                for assignment in &mut flat {
                    assignment.set(var, *value);
                }
            }
            values => {
                flat = flat
                    .iter()
                    .flat_map(|assignment| {
                        values.iter().map(move |&value| {
                            let mut extended = assignment.clone();
                            extended.set(var, value);
                            extended
                        })
                    })
                    .collect();
            }
        }
    }
    flat
}

// ============================================================
// translate_strips_conditions
// ============================================================

pub(super) fn translate_strips_conditions(
    translation: &mut Translation,
    conditions: &[Condition],
) -> Option<Vec<Assignment>> {
    if conditions.is_empty() {
        return Some(vec![Assignment::default()]); // Quick exit for common case
    }

    // Check if the condition violates any mutexes
    let mutex_result = translate_strips_conditions_aux(translation, conditions, true);
    mutex_result.as_ref()?;

    translate_strips_conditions_aux(translation, conditions, false)
}

// ============================================================
// negate_and_translate_condition
// ============================================================

pub(super) fn negate_and_translate_condition(
    translation: &mut Translation,
    add_conds: &[Vec<Condition>],
) -> Option<Vec<Assignment>> {
    // condition is a list of lists of literals (DNF)
    // the result is the negation of the condition in DNF in FDR

    if add_conds.iter().any(|c| c.is_empty()) {
        return None; // condition always satisfied, negation unsatisfiable
    }

    let mut negation = vec![];

    tools::for_each_product(add_conds, &mut |combination| {
        let cond: Vec<Condition> = combination
            .iter()
            .map(|condition| negate_condition(condition))
            .collect();
        let translated = translate_strips_conditions(translation, &cond);
        if let Some(t) = translated {
            negation.extend(t);
        }
        true
    });

    if negation.is_empty() {
        None
    } else {
        Some(negation)
    }
}

// ============================================================
// Helper functions
// ============================================================

/// Convert a single normalized fact to its dictionary atom.
pub(super) fn condition_to_atom(cond: &Condition) -> Atom {
    match cond {
        Condition::Atom(a) => a.clone(),
        Condition::NegatedAtom(na) => Atom::new(format!("NOT-{}", na.predicate), na.args.clone()),
        // A comparison has no atom of its own, so it is given one, named after
        // the comparison it stands for. The two kinds need different names: the
        // dictionary holds a variable's positive and its negative fact, and
        // sharing a name would make them the same fact.
        Condition::FunctionComparison(_) | Condition::NegatedFunctionComparison(_) => {
            let prefix = if cond.is_negated() { "__nfc" } else { "__fc" };
            let operands: Vec<String> = cond
                .comparison_operands()
                .iter()
                .map(FunctionalExpression::to_string)
                .collect();
            let name = format!("{prefix}_{}_{}", cond.comparator(), operands.join("_"));
            Atom::new(name, vec![])
        }
        Condition::Truth
        | Condition::Falsity
        | Condition::Conjunction(_)
        | Condition::Disjunction(_)
        | Condition::UniversalCondition(_)
        | Condition::ExistentialCondition(_) => {
            panic!("condition {cond} is not a single dictionary fact")
        }
    }
}

/// The comparison `comparator` relates `operands` by, asserted or denied.
pub(super) fn make_fc_condition(
    comparator: &str,
    operands: &[FunctionalExpression],
    negated: bool,
) -> Condition {
    let comparison = Comparison::new(comparator.to_owned(), operands.to_vec());
    if negated {
        Condition::NegatedFunctionComparison(comparison)
    } else {
        Condition::FunctionComparison(comparison)
    }
}

/// Negate a condition
pub(super) fn negate_condition(cond: &Condition) -> Condition {
    cond.negate()
}
