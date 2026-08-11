/// Main translation from STRIPS/PDDL ground representation to SAS+ finite-domain representation.
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use planforge_sas::numeric_conditions::ConditionValue;

use tracing::info;

use super::axiom_rules;
use super::fact_groups;
use super::normalize::NormalizableTask;
use super::numeric_axiom_rules;
use super::options;
use super::pddl::actions::PropositionalAction;
use super::pddl::axioms::{InstantiatedNumericAxiom, PropositionalAxiom};
use super::pddl::conditions::*;
use super::pddl::f_expression::*;
use super::sas_tasks::*;
use super::simplify;

/// A partial assignment of values to SAS variables.
///
/// A condition constrains a handful of variables, so a linear scan beats
/// hashing a `usize`, which profiling showed to dominate the translation of
/// large tasks. Insertion order is preserved on purpose: Fast Downward's
/// dictionaries are ordered and the translation picks "the smallest" and
/// "the first" entry out of them, so a hashed port would make the SAS task
/// depend on the hash seed.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
struct Assignment {
    pairs: Vec<(usize, usize)>,
}

impl Assignment {
    fn get(&self, var: usize) -> Option<usize> {
        self.pairs
            .iter()
            .find_map(|&(other, value)| (other == var).then_some(value))
    }

    fn set(&mut self, var: usize, value: usize) {
        match self.pairs.iter_mut().find(|(other, _)| *other == var) {
            Some(pair) => pair.1 = value,
            None => self.pairs.push((var, value)),
        }
    }

    fn remove(&mut self, var: usize) {
        self.pairs.retain(|&(other, _)| other != var);
    }

    fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.pairs.iter().copied()
    }

    fn len(&self) -> usize {
        self.pairs.len()
    }

    fn into_pairs(self) -> Vec<(usize, usize)> {
        self.pairs
    }

    fn sorted_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = self.pairs.clone();
        pairs.sort_unstable();
        pairs
    }
}

/// The values each variable may still take while a condition is translated.
/// Almost every entry is a singleton; only a negative literal widens one.
#[derive(Default)]
struct Domains {
    entries: Vec<(usize, Vec<usize>)>,
}

impl Domains {
    fn get(&self, var: usize) -> Option<&[usize]> {
        self.entries
            .iter()
            .find_map(|(other, values)| (*other == var).then_some(values.as_slice()))
    }

    fn set(&mut self, var: usize, values: Vec<usize>) {
        debug_assert!(!values.is_empty(), "a variable with no value is a conflict");
        match self.entries.iter_mut().find(|(other, _)| *other == var) {
            Some(entry) => entry.1 = values,
            None => self.entries.push((var, values)),
        }
    }

    fn set_single(&mut self, var: usize, value: usize) {
        self.set(var, vec![value]);
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Everything the translation of one task carries from step to step: the SAS
/// encoding it is filling in, the parallel encoding it tests conditions
/// against mutexes with, the comparison axioms it invents for numeric
/// conditions, and the two counters it reports at the end.
///
/// These travelled as seven to fourteen parameters through every function
/// below, always in the same order and always all together.
struct Translation {
    ranges: Vec<usize>,
    dictionary: HashMap<Atom, Vec<SasFact>>,
    numeric: HashMap<PrimitiveNumericExpression, usize>,
    /// One variable per mutex group, including the groups no SAS variable was
    /// built from, so that a condition can be tested against all of them.
    mutex_ranges: Vec<usize>,
    mutex_dictionary: HashMap<Atom, Vec<SasFact>>,
    /// The variable a `(comparator, operands)` comparison was given, so that
    /// the same comparison is not encoded twice.
    comparison_axioms: HashMap<(String, Vec<usize>), Condition>,
    sas_comparison_axioms: Vec<SASCompareAxiom>,
    /// The expression each SAS numeric variable is named after, one per
    /// variable. Its length is the number of SAS numeric variables.
    numeric_names: Vec<PrimitiveNumericExpression>,
    /// The human-readable name of each value of each variable.
    translation_key: Vec<Vec<String>>,
    /// One entry per mutex group, in SAS facts.
    mutex_key: Vec<Vec<SasFact>>,
    implied_facts: HashMap<SasFact, Vec<SasFact>>,
    /// The initial value of every numeric variable, for static costs.
    num_vals: Vec<f64>,
    relevant_numeric: Vec<usize>,
    simplified_effect_conditions: usize,
    added_implied_preconditions: usize,
}

// ============================================================
// the SAS encoding of facts and numeric fluents
// ============================================================

/// One SAS variable per group of mutually exclusive facts, plus a value for
/// "none of those", and the map from a fact to the variable/value pairs that
/// express it.
struct FactEncoding {
    ranges: Vec<usize>,
    dictionary: HashMap<Atom, Vec<SasFact>>,
}

fn encode_facts(groups: &[Vec<Atom>], assert_partial: bool) -> FactEncoding {
    let mut dictionary: HashMap<Atom, Vec<SasFact>> = HashMap::new();
    for (var_no, group) in groups.iter().enumerate() {
        for (val_no, atom) in group.iter().enumerate() {
            dictionary
                .entry(atom.clone())
                .or_default()
                .push((var_no, val_no));
        }
    }

    if assert_partial {
        for (atom, sas_pairs) in &dictionary {
            assert_eq!(
                sas_pairs.len(),
                1,
                "partial encoding covers {atom:?} by {} variables",
                sas_pairs.len()
            );
        }
    }

    FactEncoding {
        ranges: groups.iter().map(|group| group.len() + 1).collect(),
        dictionary,
    }
}

/// One SAS numeric variable per axiom effect that no equivalent axiom already
/// defines, then one per fluent none of them defines. An axiom made redundant
/// by an equivalent one shares that one's variable.
struct NumericEncoding {
    /// The expression a SAS numeric variable is named after: the effect of the
    /// axiom that defines it, or the fluent it stands for. Several expressions
    /// can map onto the same variable, so the name cannot be recovered from
    /// `variables` -- it is fixed here, where the variable is created.
    names: Vec<PrimitiveNumericExpression>,
    variables: HashMap<PrimitiveNumericExpression, usize>,
}

fn encode_numeric_fluents(
    num_axioms: &[InstantiatedNumericAxiom],
    equivalent: &HashMap<PrimitiveNumericExpression, PrimitiveNumericExpression>,
    num_fluents: &[PrimitiveNumericExpression],
) -> NumericEncoding {
    let mut variables: HashMap<PrimitiveNumericExpression, usize> = HashMap::new();
    let mut names: Vec<PrimitiveNumericExpression> = Vec::new();

    let mut redundant = vec![];
    for axiom in num_axioms {
        if equivalent.contains_key(&axiom.effect) {
            redundant.push(axiom.effect.clone());
        } else {
            let previous = variables.insert(axiom.effect.clone(), names.len());
            assert!(
                previous.is_none(),
                "numeric axiom effect {} defines a SAS variable twice",
                axiom.effect
            );
            names.push(axiom.effect.clone());
        }
    }
    for effect in &redundant {
        if let Some(kept) = equivalent.get(effect)
            && let Some(&variable) = variables.get(kept)
        {
            variables.insert(effect.clone(), variable);
        }
    }

    let mut fluents: Vec<PrimitiveNumericExpression> = num_fluents.to_vec();
    fluents.sort_by_cached_key(PrimitiveNumericExpression::to_string);
    for fluent in fluents {
        if let Entry::Vacant(entry) = variables.entry(fluent) {
            names.push(entry.key().clone());
            entry.insert(names.len() - 1);
        }
    }

    NumericEncoding { names, variables }
}

// ============================================================
// translate_strips_conditions_aux
// ============================================================

/// Translates a conjunction of ground literals into the finite-domain
/// conditions that entail it, or `None` if it is unsatisfiable.
fn translate_strips_conditions_aux(
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
                if let Some(atom) = condition_to_atom(fact)
                    && let Some(pairs) = dictionary.get(&atom)
                {
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
                    .filter_map(|p| match p {
                        FunctionalExpression::PrimitiveNumericExpression(pne) => {
                            Some(*numeric_dictionary.get(pne).unwrap_or_else(|| {
                                panic!("PNE {:?} not in numeric dictionary", pne)
                            }))
                        }
                        _ => None,
                    })
                    .collect();

                let key = (comparator.to_owned(), parts.clone());

                if let Some(existing_fact) = comp_axiom_dict.get(&key) {
                    // Already have this comparison axiom
                    let lookup_fact = if negated {
                        negate_condition(existing_fact)
                    } else {
                        existing_fact.clone()
                    };
                    if let Some(atom) = condition_to_atom(&lookup_fact)
                        && let Some(pairs) = dictionary.get(&atom)
                    {
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

                    let pos_atom = condition_to_atom(&pos_fact).unwrap();
                    let neg_atom = condition_to_atom(&neg_fact).unwrap();

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
                    if let Some(atom) = condition_to_atom(lookup_fact)
                        && let Some(pairs) = dictionary.get(&atom)
                    {
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
                    assert!(
                        mutex_check,
                        "condition atom {atom:?} has no SAS variable; it is neither reachable \
                         nor statically decided"
                    );
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
            _ => continue,
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
            _ => continue,
        }
    }

    Some(multiply_out(condition))
}

/// Expands a condition that allows several values per variable into the flat
/// assignments it stands for, fewest alternatives first.
fn multiply_out(condition: Domains) -> Vec<Assignment> {
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

fn translate_strips_conditions(
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
// translate_strips_operator
// ============================================================

fn translate_strips_operator(
    translation: &mut Translation,
    operator: &PropositionalAction,
) -> Vec<SASOperator> {
    let conditions = translate_strips_conditions(translation, &operator.precondition);

    if conditions.is_none() {
        return vec![];
    }

    let mut sas_operators = vec![];
    for condition in conditions.unwrap() {
        if let Some(op) = translate_strips_operator_aux(translation, operator, &condition) {
            sas_operators.push(op);
        }
    }
    sas_operators
}

// ============================================================
// negate_and_translate_condition
// ============================================================

fn negate_and_translate_condition(
    translation: &mut Translation,
    add_conds: &[Vec<Condition>],
) -> Option<Vec<Assignment>> {
    // condition is a list of lists of literals (DNF)
    // the result is the negation of the condition in DNF in FDR

    if add_conds.iter().any(|c| c.is_empty()) {
        return None; // condition always satisfied, negation unsatisfiable
    }

    let mut negation = vec![];

    // Cartesian product of all condition lists
    let combinations = cartesian_product_conditions(add_conds);
    for combination in &combinations {
        let cond: Vec<Condition> = combination.iter().map(negate_condition).collect();
        let translated = translate_strips_conditions(translation, &cond);
        if let Some(t) = translated {
            negation.extend(t);
        }
    }

    if negation.is_empty() {
        None
    } else {
        Some(negation)
    }
}

/// Cartesian product of condition lists
fn cartesian_product_conditions(lists: &[Vec<Condition>]) -> Vec<Vec<Condition>> {
    if lists.is_empty() {
        return vec![vec![]];
    }
    let first = &lists[0];
    let rest = cartesian_product_conditions(&lists[1..]);
    let mut result = vec![];
    for item in first {
        for r in &rest {
            let mut combo = vec![item.clone()];
            combo.extend(r.clone());
            result.push(combo);
        }
    }
    result
}

// ============================================================
// translate_strips_operator_aux
// ============================================================

fn translate_strips_operator_aux(
    translation: &mut Translation,
    operator: &PropositionalAction,
    condition: &Assignment,
) -> Option<SASOperator> {
    // Collect all add effects
    let mut effects_by_variable: HashMap<usize, HashMap<usize, Vec<Assignment>>> = HashMap::new();
    let mut add_conds_by_variable: HashMap<usize, Vec<Vec<Condition>>> = HashMap::new();

    for (conditions_list, fact) in &operator.add_effects {
        let eff_condition_list = translate_strips_conditions(translation, conditions_list);
        if eff_condition_list.is_none() {
            continue; // Impossible condition
        }
        if let Some(pairs) = translation.dictionary.get(fact) {
            for &(var, val) in pairs {
                effects_by_variable
                    .entry(var)
                    .or_default()
                    .entry(val)
                    .or_default()
                    .extend(eff_condition_list.clone().unwrap());
                add_conds_by_variable
                    .entry(var)
                    .or_default()
                    .push(conditions_list.clone());
            }
        }
    }

    // Collect all del effects
    let mut del_effects_by_variable: HashMap<usize, HashMap<usize, Vec<Assignment>>> =
        HashMap::new();

    for (conditions_list, fact) in &operator.del_effects {
        let eff_condition_list = translate_strips_conditions(translation, conditions_list);
        if eff_condition_list.is_none() {
            continue;
        }
        if let Some(pairs) = translation.dictionary.get(fact) {
            for &(var, val) in pairs {
                del_effects_by_variable
                    .entry(var)
                    .or_default()
                    .entry(val)
                    .or_default()
                    .extend(eff_condition_list.clone().unwrap());
            }
        }
    }

    // Collect all (numeric) assignment effects
    let mut ass_effects_by_variable: HashMap<usize, HashMap<(String, usize), Vec<Assignment>>> =
        HashMap::new();

    for (conditions_list, assignment) in &operator.assign_effects {
        let eff_condition_list = translate_strips_conditions(translation, conditions_list);
        if eff_condition_list.is_none() {
            continue;
        }
        if let Some(expr_pne) = assignment.expression.as_pne() {
            if let Some(&expr_var) = translation.numeric.get(expr_pne)
                && let Some(&fluent_var) = translation.numeric.get(&assignment.fluent)
            {
                ass_effects_by_variable
                    .entry(fluent_var)
                    .or_default()
                    .entry((assignment.symbol.clone(), expr_var))
                    .or_default()
                    .extend(eff_condition_list.unwrap());
            }
        } else {
            // Expression might be in numeric translation.dictionary directly
            // Check if expression can be looked up
        }
    }

    if let Some(cost_assignment) = &operator.cost
        && let Some(expr_pne) = cost_assignment.expression.as_pne()
        && let Some(&expr_var) = translation.numeric.get(expr_pne)
        && let Some(&fluent_var) = translation.numeric.get(&cost_assignment.fluent)
    {
        ass_effects_by_variable
            .entry(fluent_var)
            .or_default()
            .entry((cost_assignment.symbol.clone(), expr_var))
            .or_default()
            .push(Assignment::default());
    }

    // Handle del effects: add var=none_of_those when deleted and no add effect
    for (&var, del_vals) in &del_effects_by_variable {
        let add_conds = add_conds_by_variable
            .get(&var)
            .cloned()
            .unwrap_or_else(Vec::new);

        let no_add_effect_condition = negate_and_translate_condition(translation, &add_conds);

        if no_add_effect_condition.is_none() {
            continue; // Always an add effect
        }

        let none_of_those = translation.ranges[var] - 1;
        for (&val, conds) in del_vals {
            for cond in conds {
                let mut guard_cond = cond.clone();
                if guard_cond.get(var).is_some_and(|existing| existing != val) {
                    continue; // Condition inconsistent with deleted atom
                }
                guard_cond.set(var, val);

                for no_add_cond in no_add_effect_condition.as_ref().unwrap() {
                    let mut new_cond = guard_cond.clone();
                    let mut contradicts = false;
                    for (cvar, cval) in no_add_cond.iter() {
                        if new_cond.get(cvar).is_some_and(|existing| existing != cval) {
                            contradicts = true;
                            break;
                        }
                        new_cond.set(cvar, cval);
                    }
                    if !contradicts {
                        effects_by_variable
                            .entry(var)
                            .or_default()
                            .entry(none_of_those)
                            .or_default()
                            .push(new_cond);
                    }
                }
            }
        }
    }

    let cost = sas_operator_cost(translation, &operator.name, operator.cost.as_ref());
    build_sas_operator(
        translation,
        &operator.name,
        condition,
        &effects_by_variable,
        &ass_effects_by_variable,
        cost,
    )
}

/// Scalar cost of a SAS operator.
///
/// An action without an `(increase (total-cost) ...)` effect costs 1, PDDL's
/// default action cost. A cost effect is either a literal constant or — after
/// normalization, which rewrites every compound cost expression into a derived
/// function — a primitive numeric expression; the latter is evaluated in the
/// initial state, mirroring how Fast Downward resolves static cost functions.
/// A state-dependent cost is additionally recorded as an assignment effect on
/// the metric fluent by the caller, which is what the search actually uses for
/// tasks with a metric (`metric_operator_cost_from_initial_values`).
///
/// Anything else cannot be turned into a number here; returning a default
/// would silently replace the action's real cost, so it is a hard error.
fn sas_operator_cost(
    translation: &Translation,
    action_name: &str,
    cost: Option<&FunctionAssignment>,
) -> f64 {
    let (numeric_dictionary, num_vals) = (&translation.numeric, &translation.num_vals);
    let Some(cost) = cost else {
        return 1.0;
    };
    match &cost.expression {
        FunctionalExpression::NumericConstant(nc) => nc.value.into_inner(),
        FunctionalExpression::PrimitiveNumericExpression(pne) => {
            let &var = numeric_dictionary.get(pne).unwrap_or_else(|| {
                panic!("cost expression {pne} of action {action_name} has no numeric variable")
            });
            num_vals[var]
        }
        other => panic!("action {action_name} has an unsupported cost expression {other}"),
    }
}

// ============================================================
// build_sas_operator
// ============================================================

fn build_sas_operator(
    translation: &mut Translation,
    name: &str,
    condition: &Assignment,
    effects_by_variable: &HashMap<usize, HashMap<usize, Vec<Assignment>>>,
    ass_effects_by_variable: &HashMap<usize, HashMap<(String, usize), Vec<Assignment>>>,
    cost: f64,
) -> Option<SASOperator> {
    let Translation {
        ranges,
        implied_facts,
        relevant_numeric: relevant_numeric_variables,
        simplified_effect_conditions: simplified_effect_condition_counter,
        added_implied_preconditions: added_implied_precondition_counter,
        ..
    } = translation;
    let implied_precondition: HashSet<(usize, usize)> = if options::ADD_IMPLIED_PRECONDITIONS {
        let mut ip = HashSet::new();
        for fact in condition.iter() {
            if let Some(implied) = implied_facts.get(&fact) {
                for &f in implied {
                    ip.insert(f);
                }
            }
        }
        ip
    } else {
        HashSet::new()
    };

    let mut prevail_and_pre = condition.clone();
    let mut pre_post: Vec<PrePost> = vec![];
    let mut num_pre_post: Vec<AssignEffect> = vec![];

    for (&var, effects) in effects_by_variable {
        let orig_pre = condition.get(var).map_or(-1, |value| value as i32);
        let mut added_effect = false;

        for (&post, eff_conditions) in effects {
            let mut pre = orig_pre;
            // If the effect does not change the variable value, ignore it
            if pre == post as i32 {
                continue;
            }

            let mut eff_condition_lists: Vec<Vec<(usize, usize)>> = eff_conditions
                .iter()
                .map(Assignment::sorted_pairs)
                .collect();

            if ranges[var] == 2 {
                // Apply simplifications for binary variables
                if prune_stupid_effect_conditions(var, post, &mut eff_condition_lists) {
                    *simplified_effect_condition_counter += 1;
                }
                if options::ADD_IMPLIED_PRECONDITIONS
                    && pre == -1
                    && implied_precondition.contains(&(var, 1 - post))
                {
                    *added_implied_precondition_counter += 1;
                    pre = (1 - post) as i32;
                }
            }

            for eff_condition in &eff_condition_lists {
                let mut filtered_eff_condition: Vec<(usize, usize)> = vec![];
                let mut eff_condition_contradicts = false;

                for &(variable, value) in eff_condition {
                    if let Some(prevail_val) = prevail_and_pre.get(variable) {
                        if prevail_val != value {
                            eff_condition_contradicts = true;
                            break;
                        }
                    } else {
                        filtered_eff_condition.push((variable, value));
                    }
                }

                if eff_condition_contradicts {
                    continue;
                }

                pre_post.push((var, pre, post, filtered_eff_condition));
                added_effect = true;
            }
        }

        if added_effect {
            prevail_and_pre.remove(var);
        }
    }

    for (&numvar, effects) in ass_effects_by_variable {
        for ((ass_op, post_var), eff_conditions) in effects {
            let eff_condition_lists: Vec<Vec<(usize, usize)>> = eff_conditions
                .iter()
                .map(Assignment::sorted_pairs)
                .collect();

            for eff_condition in &eff_condition_lists {
                num_pre_post.push((numvar, ass_op.clone(), *post_var, eff_condition.clone()));
            }
        }
    }

    if pre_post.is_empty() {
        // Check if any numeric effect is relevant
        let mut irrelevant = true;
        for (eff_var, _, _, _) in &num_pre_post {
            if relevant_numeric_variables.contains(eff_var) {
                irrelevant = false;
                break;
            }
        }
        if irrelevant {
            return None;
        }
    }

    // Remove effect variables from prevail
    let prevail = prevail_and_pre.into_pairs();

    Some(SASOperator::new(
        name.to_string(),
        prevail,
        pre_post,
        num_pre_post,
        cost,
    ))
}

// ============================================================
// prune_stupid_effect_conditions
// ============================================================

fn prune_stupid_effect_conditions(
    var: usize,
    val: usize,
    conditions: &mut Vec<Vec<(usize, usize)>>,
) -> bool {
    if conditions == &[vec![]] {
        return false; // Quick exit for common case
    }

    assert!(val == 0 || val == 1);
    let dual_fact = (var, 1 - val);
    let mut simplified = false;

    for condition in conditions.iter_mut() {
        // Rule 1: remove dual fact from condition
        let len_before = condition.len();
        condition.retain(|f| *f != dual_fact);
        if condition.len() != len_before {
            simplified = true;
        }
        // Rule 2 is checked below
    }

    // Rule 2: if any condition is empty, simplify to [[]]
    if conditions.iter().any(|c| c.is_empty()) {
        *conditions = vec![vec![]];
        simplified = true;
    }

    simplified
}

// ============================================================
// translate_strips_axiom
// ============================================================

fn translate_strips_axiom(
    translation: &mut Translation,
    axiom: &PropositionalAxiom,
) -> Vec<SASAxiom> {
    let conditions = translate_strips_conditions(translation, &axiom.condition);
    if conditions.is_none() {
        return vec![];
    }

    // Since issue454 every axiom *proves* its head: the rules that refute a
    // derived variable are computed by the consumer that needs them, from the SAS
    // task. A negated head reaching here would silently become a rule writing the
    // variable's `<none of those>` value, which is not the same fact.
    let Condition::Atom(atom) = &axiom.effect else {
        panic!("an axiom head is a positive atom, got {}", axiom.effect);
    };
    let Some(pairs) = translation.dictionary.get(atom) else {
        return vec![];
    };
    let effect = pairs[0];

    let mut axioms = vec![];
    for condition in conditions.unwrap() {
        axioms.push(SASAxiom::new(condition.into_pairs(), effect));
    }
    axioms
}

// ============================================================
// translate_numeric_axiom
// ============================================================

fn translate_numeric_axiom(
    axiom: &InstantiatedNumericAxiom,
    _prop_dictionary: &HashMap<Atom, Vec<(usize, usize)>>,
    num_dictionary: &HashMap<PrimitiveNumericExpression, usize>,
) -> Option<SASNumericAxiom> {
    let effect = num_dictionary.get(&axiom.effect)?;
    let op = &axiom.op;
    let mut parts = vec![];
    for part in &axiom.parts {
        match part {
            FunctionalExpression::PrimitiveNumericExpression(pne) => {
                if let Some(&idx) = num_dictionary.get(pne) {
                    parts.push(idx);
                } else {
                    return None;
                }
            }
            FunctionalExpression::NumericConstant(_) => {
                // Constants should have been resolved
                return None;
            }
            _ => {
                return None;
            }
        }
    }
    Some(SASNumericAxiom::new(op.clone(), parts, *effect))
}

// ============================================================
// translate_strips_operators
// ============================================================

fn translate_strips_operators(
    translation: &mut Translation,
    actions: &[PropositionalAction],
) -> Vec<SASOperator> {
    let mut result = vec![];
    for action in actions {
        let sas_ops = translate_strips_operator(translation, action);
        result.extend(sas_ops);
    }
    result
}

// ============================================================
// translate_strips_axioms
// ============================================================

fn translate_strips_axioms(
    translation: &mut Translation,
    axioms: &[PropositionalAxiom],
) -> Vec<SASAxiom> {
    let mut result = vec![];
    for axiom in axioms {
        let sas_axioms = translate_strips_axiom(translation, axiom);
        result.extend(sas_axioms);
    }
    result
}

// ============================================================
// add_key_to_comp_axioms
// ============================================================

fn add_key_to_comp_axioms(
    sas_comp_axioms: &[SASCompareAxiom],
    translation_key: &mut Vec<Vec<String>>,
) {
    for axiom in sas_comp_axioms {
        assert_eq!(
            axiom.effect,
            translation_key.len(),
            "current effect {} != next variable {}",
            axiom.effect,
            translation_key.len()
        );
        translation_key.push(vec![
            axiom.to_string(),
            axiom.invert_comparator().to_string(),
        ]);
    }
}

// ============================================================
// translate_task
// ============================================================

/// The grounded task an SAS+ encoding is built from.
struct GroundedTask<'a> {
    init: &'a [Atom],
    num_init: &'a [FunctionAssignment],
    goal_list: &'a [Condition],
    global_constraint: &'a Condition,
    actions: &'a [PropositionalAction],
    axioms: Vec<PropositionalAxiom>,
    metric: &'a (String, PrimitiveNumericExpression),
}

fn translate_task(
    translation: &mut Translation,
    task: GroundedTask,
    numeric_axioms: &numeric_axiom_rules::NumericAxioms,
    layer_strategy: options::LayerStrategy,
) -> Result<SASTask, String> {
    // Process axioms. Derived variables are not listed here: every one of them
    // defaults to false, which the closed-world assumption below already gives
    // them, and the axiom closure computes the real value on top of that.
    let (processed_axioms, axiom_layer_dict) = axiom_rules::handle_axioms(
        task.actions,
        task.axioms,
        task.goal_list,
        task.global_constraint,
        layer_strategy,
    )?;

    // Initialize init_values: Closed World Assumption
    let mut init_values: Vec<i32> = translation.ranges.iter().map(|&r| (r as i32) - 1).collect();
    for fact in task.init {
        if let Some(pairs) = translation.dictionary.get(fact) {
            for &(var, val) in pairs {
                let curr_val = init_values[var];
                if curr_val != (translation.ranges[var] as i32 - 1) && curr_val != val as i32 {
                    return Err(format!("Inconsistent init facts! [fact = {:?}]", fact));
                }
                init_values[var] = val as i32;
            }
        }
    }

    // Translate goal
    let goal_dict_list = translate_strips_conditions(translation, task.goal_list);

    // Translate global constraint
    let gc_as_list = vec![task.global_constraint.clone()];
    let global_constraint_dict_list = translate_strips_conditions(translation, &gc_as_list);

    if goal_dict_list.is_none() {
        return Ok(trivial_task(false, "Goal violates a mutex"));
    }

    let goal_dict_list = goal_dict_list.unwrap();
    // A goal literal names one value of one variable: a positive one its own
    // fact's value, a negative one the only other value of the binary variable
    // that `compute_groups` reserved for it.
    assert_eq!(goal_dict_list.len(), 1, "the goal is a single assignment");

    let goal_pairs: Vec<(usize, usize)> = goal_dict_list[0].iter().collect();

    if goal_pairs.is_empty() {
        return Ok(trivial_task(true, "Empty goal"));
    }

    let sas_goal = SASGoal::new(goal_pairs);

    assert!(
        global_constraint_dict_list.is_some()
            && global_constraint_dict_list.as_ref().unwrap().len() == 1
    );

    // Numeric init values.
    //
    // A fluent without a SAS numeric variable is skipped: it holds a constant,
    // which every expression mentioning it was folded against, so there is no
    // variable left for its initial value to name.
    let mut num_init_values: Vec<f64> = vec![0.0; translation.numeric_names.len()];

    let mut relevant_numeric: Vec<usize> = vec![];
    for fact in task.num_init {
        let Some(&var) = translation.numeric.get(&fact.fluent) else {
            continue;
        };
        let FunctionalExpression::NumericConstant(nc) = &fact.expression else {
            return Err(format!(
                "numeric init fact for {} must assign a numeric constant, got {}",
                fact.fluent, fact.expression
            ));
        };
        num_init_values[var] = nc.value.into_inner();
        if fact.fluent.ntype == 'R' {
            relevant_numeric.push(var);
        }
    }

    // Fold the constant numeric axioms into the initial values before the
    // operators are translated: `sas_operator_cost` evaluates a non-constant
    // action cost in the initial state, and normalization rewrites every
    // compound cost expression into a derived function backed by such an
    // axiom. Constant axiom effects are derived fluents and therefore never
    // collide with the `:init` facts above.
    for axiom in &numeric_axioms.constant {
        if let Some(&var) = translation.numeric.get(&axiom.effect) {
            let Some(FunctionalExpression::NumericConstant(nc)) = axiom.parts.first() else {
                unreachable!(
                    "constant numeric axiom {} must hold a folded numeric constant",
                    axiom.name
                );
            };
            num_init_values[var] = nc.value.into_inner();
        }
    }

    // Translate operators
    translation.num_vals = num_init_values;
    translation.relevant_numeric = relevant_numeric;
    let operators = translate_strips_operators(translation, task.actions);

    // Translate axioms
    let sas_axioms = translate_strips_axioms(translation, &processed_axioms);

    // Translate numeric axioms
    let const_num_axiom_effects: HashSet<PrimitiveNumericExpression> = numeric_axioms
        .constant
        .iter()
        .map(|ax| ax.effect.clone())
        .collect();
    let sas_num_axioms: Vec<SASNumericAxiom> = numeric_axioms
        .axioms
        .iter()
        .filter(|ax| {
            !const_num_axiom_effects.contains(&ax.effect)
                && !numeric_axioms.equivalent.contains_key(&ax.effect)
        })
        .filter_map(|ax| translate_numeric_axiom(ax, &translation.dictionary, &translation.numeric))
        .collect();

    // Compute axiom layers
    let mut axiom_layers: Vec<i32> = vec![-1; translation.ranges.len()];
    let mut num_axiom_layers: Vec<i32> = vec![-1; translation.numeric_names.len()];
    let mut num_axiom_layer = 0i32;

    for (&layer, layer_axioms) in &numeric_axioms.by_layer {
        let mut sorted_axioms = layer_axioms.clone();
        sorted_axioms.sort_by(|a, b| a.name.cmp(&b.name));
        for axiom in &sorted_axioms {
            if !numeric_axioms.equivalent.contains_key(&axiom.effect)
                && let Some(&var) = translation.numeric.get(&axiom.effect)
            {
                if layer == -1 {
                    num_axiom_layers[var] = -1;
                } else {
                    num_axiom_layers[var] = num_axiom_layer;
                    num_axiom_layer += 1;
                }
            }
        }
    }

    // A comparison variable's initial-state entry is its axiom default, and the
    // comparison axiom overwrites it with a verdict before anything reads it.
    // `False` is the default that says what the closure has not yet proven.
    let comp_axiom_default = ConditionValue::False.as_usize() as i32;
    init_values.extend(std::iter::repeat_n(
        comp_axiom_default,
        translation.sas_comparison_axioms.len(),
    ));

    for axiom in &translation.sas_comparison_axioms {
        axiom_layers[axiom.effect] = num_axiom_layer;
    }

    for (atom, &layer) in &axiom_layer_dict {
        assert!(layer >= 0);
        if let Some(pairs) = translation.dictionary.get(atom) {
            let (var, _val) = pairs[0];
            axiom_layers[var] = layer + num_axiom_layer + 1;
        }
    }

    // Extend axiom_layers for comparison axiom variables
    while axiom_layers.len() < translation.ranges.len() {
        axiom_layers.push(num_axiom_layer);
    }

    add_key_to_comp_axioms(
        &translation.sas_comparison_axioms.clone(),
        &mut translation.translation_key,
    );

    let variables = SASVariables::new(
        translation.ranges.clone(),
        axiom_layers,
        translation.translation_key.clone(),
        num_axiom_layer,
    );

    // Name each numeric variable after the expression it was created for. The
    // `numeric` map cannot serve here: equivalent axiom effects share a
    // variable, so it holds several names per variable.
    let num_variables: Vec<String> = translation
        .numeric_names
        .iter()
        .map(|entry| format!("{}", entry))
        .collect();
    let num_var_types: Vec<String> = translation
        .numeric_names
        .iter()
        .map(|entry| entry.ntype.to_string())
        .collect();

    let numeric_variables =
        SASNumericVariables::new(num_variables, num_axiom_layers, num_var_types);

    let mutexes: Vec<SASMutexGroup> = translation
        .mutex_key
        .iter()
        .map(|group| SASMutexGroup::new(group.clone()))
        .collect();

    let sas_init = SASInit::new(init_values, translation.num_vals.clone());

    // Look up task.metric fluent
    let sas_metric = if task.metric.1.symbol.is_empty() || task.metric.1.ntype == 'X' {
        // Unit cost or special marker
        (task.metric.0.clone(), -1i64)
    } else {
        if let Some(&idx) = translation.numeric.get(&task.metric.1) {
            (task.metric.0.clone(), idx as i64)
        } else {
            (task.metric.0.clone(), -1i64)
        }
    };

    // The global constraint is a single atom (asserted by the caller) and
    // `USE_PARTIAL_ENCODING` maps every atom to exactly one SAS fact, so the
    // translated condition holds exactly one pair. Picking an arbitrary pair
    // out of a larger map would silently install a different constraint, and
    // an empty map has no pair to pick at all.
    let global_constraint_facts = &global_constraint_dict_list.unwrap()[0];
    assert_eq!(
        global_constraint_facts.len(),
        1,
        "global constraint must translate to exactly one SAS fact, got {global_constraint_facts:?}"
    );
    let gc_pair = global_constraint_facts
        .iter()
        .next()
        .expect("length checked above");

    let mut sas_task = SASTask {
        variables,
        numeric_variables,
        mutexes,
        init: sas_init,
        goal: sas_goal,
        operators,
        axioms: sas_axioms,
        comp_axioms: std::mem::take(&mut translation.sas_comparison_axioms),
        numeric_axioms: sas_num_axioms,
        global_constraint: gc_pair,
        metric: sas_metric,
    };
    sas_task.canonicalize();
    Ok(sas_task)
}

// ============================================================
// trivial_task
// ============================================================

fn trivial_task(solvable: bool, msg: &str) -> SASTask {
    if solvable {
        info!("{}! Generating solvable task...", msg);
    } else {
        info!("{}! Generating unsolvable task...", msg);
    }
    simplify::trivial_task(solvable)
}

// ============================================================
// build_mutex_key
// ============================================================

fn build_mutex_key(
    strips_to_sas: &HashMap<Atom, Vec<(usize, usize)>>,
    groups: &[Vec<Atom>],
) -> Vec<Vec<(usize, usize)>> {
    let mut group_keys = vec![];
    for group in groups {
        let mut group_key = vec![];
        for fact in group {
            if let Some(pairs) = strips_to_sas.get(fact) {
                for &(var, val) in pairs {
                    group_key.push((var, val));
                }
            } else {
                info!("not in strips_to_sas, left out: {:?}", fact);
            }
        }
        group_keys.push(group_key);
    }
    group_keys
}

// ============================================================
// build_implied_facts
// ============================================================

fn build_implied_facts(
    strips_to_sas: &HashMap<Atom, Vec<(usize, usize)>>,
    groups: &[Vec<Atom>],
    mutex_groups: &[Vec<Atom>],
) -> HashMap<(usize, usize), Vec<(usize, usize)>> {
    // Find lonely propositions (groups of size 1)
    let mut lonely_propositions: HashMap<Atom, usize> = HashMap::new();
    for (var_no, group) in groups.iter().enumerate() {
        if group.len() == 1 {
            let lonely_prop = &group[0];
            if let Some(pairs) = strips_to_sas.get(lonely_prop) {
                assert_eq!(pairs, &[(var_no, 0)]);
                lonely_propositions.insert(lonely_prop.clone(), var_no);
            }
        }
    }

    let mut implied_facts: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();

    for mutex_group in mutex_groups {
        for prop in mutex_group {
            if let Some(&prop_var) = lonely_propositions.get(prop) {
                let prop_is_false = (prop_var, 1);
                for other_prop in mutex_group {
                    if other_prop != prop
                        && let Some(other_facts) = strips_to_sas.get(other_prop)
                    {
                        for &other_fact in other_facts {
                            implied_facts
                                .entry(other_fact)
                                .or_default()
                                .push(prop_is_false);
                        }
                    }
                }
            }
        }
    }

    implied_facts
}

// ============================================================
// Main entry point: pddl_to_sas / translate_task_from_grounded_internal
// ============================================================

/// Called from main.rs as translate_task_from_grounded_internal
/// Encodes a grounded task as SAS+.
///
/// `singleton_groups` skips invariant synthesis and gives every fact its own
/// variable: a larger encoding of the same task.
pub fn translate_task_from_grounded_internal(
    explored: &crate::instantiate::ExploreResult,
    norm_task: &NormalizableTask,
    singleton_groups: bool,
    layer_strategy: options::LayerStrategy,
) -> Result<SASTask, String> {
    let task = &norm_task.task;
    let crate::instantiate::ExploreResult {
        atoms,
        num_fluents,
        grounded_ops,
        grounded_axioms,
        numeric_axioms: num_axioms,
        reachable_action_params,
        ..
    } = explored;
    let goal = &norm_task.goal;

    fn type_rank(ntype: char) -> u8 {
        match ntype {
            'I' => 4,
            'R' => 3,
            'D' => 2,
            'C' => 1,
            _ => 0,
        }
    }

    fn merge_numeric_fluent_type(
        merged: &mut HashMap<(String, Vec<String>), PrimitiveNumericExpression>,
        pne: PrimitiveNumericExpression,
    ) {
        let key = (pne.symbol.clone(), pne.args.clone());
        match merged.get(&key) {
            Some(existing) if type_rank(existing.ntype) >= type_rank(pne.ntype) => {}
            _ => {
                merged.insert(key, pne);
            }
        }
    }

    let mut merged_num_fluents: HashMap<(String, Vec<String>), PrimitiveNumericExpression> =
        HashMap::new();
    for fluent in num_fluents {
        merge_numeric_fluent_type(&mut merged_num_fluents, fluent.clone());
    }
    merge_numeric_fluent_type(&mut merged_num_fluents, task.metric.1.clone());

    let mut num_fluents_vec: Vec<PrimitiveNumericExpression> =
        merged_num_fluents.into_values().collect();
    num_fluents_vec.sort_by(|left, right| {
        left.symbol
            .cmp(&right.symbol)
            .then_with(|| left.args.cmp(&right.args))
            .then_with(|| left.ntype.cmp(&right.ntype))
    });

    // Build goal list
    let goal_list: Vec<Condition> = match goal {
        Condition::Conjunction(conj) => conj.parts.clone(),
        other => vec![other.clone()],
    };

    let mut negative_in_goal: HashSet<Atom> = HashSet::new();
    for item in &goal_list {
        match item {
            Condition::Atom(_) => {}
            Condition::NegatedAtom(negated) => {
                negative_in_goal.insert(Atom::new(negated.predicate.clone(), negated.args.clone()));
            }
            _ => return Err(format!("Non-literal goal: {:?}", item)),
        }
    }

    // Compute fact groups
    let atoms_set: HashSet<Atom> = atoms.iter().cloned().collect();
    let fact_groups::FactGroups {
        groups,
        mutex_groups,
        translation_key,
    } = if singleton_groups {
        // Fast path: skip invariant finding / mutex discovery. Every fact
        // already gets a binary variable, so a negative goal needs nothing
        // extra. This preserves semantics but produces a less compact encoding.
        fact_groups::compute_singleton_groups(&atoms_set)
    } else {
        fact_groups::compute_groups(task, &atoms_set, reachable_action_params, &negative_in_goal)
    };

    let numeric_axioms = numeric_axiom_rules::handle_axioms(num_axioms);

    let facts = encode_facts(&groups, options::USE_PARTIAL_ENCODING);
    let numeric = encode_numeric_fluents(
        &numeric_axioms.axioms,
        &numeric_axioms.equivalent,
        &num_fluents_vec,
    );
    // The mutex groups overlap, so they get their own variables, used only to
    // test whether a condition violates a mutex.
    let mutex_facts = encode_facts(&mutex_groups, false);

    let implied_facts = if options::ADD_IMPLIED_PRECONDITIONS {
        build_implied_facts(&facts.dictionary, &groups, &mutex_groups)
    } else {
        HashMap::new()
    };
    let mutex_key = build_mutex_key(&facts.dictionary, &mutex_groups);

    let mut translation = Translation {
        ranges: facts.ranges,
        dictionary: facts.dictionary,
        numeric: numeric.variables,
        numeric_names: numeric.names,
        mutex_ranges: mutex_facts.ranges,
        mutex_dictionary: mutex_facts.dictionary,
        mutex_key,
        translation_key,
        comparison_axioms: HashMap::new(),
        sas_comparison_axioms: Vec::new(),
        implied_facts,
        num_vals: Vec::new(),
        relevant_numeric: Vec::new(),
        simplified_effect_conditions: 0,
        added_implied_preconditions: 0,
    };

    let gc = &task.global_constraint;
    assert!(
        matches!(gc, Condition::Atom(_)),
        "Global constraint must be an atom literal"
    );

    let sas_task = translate_task(
        &mut translation,
        GroundedTask {
            init: &task.init,
            num_init: &task.num_init,
            goal_list: &goal_list,
            global_constraint: gc,
            actions: grounded_ops,
            axioms: grounded_axioms.to_vec(),
            metric: &task.metric,
        },
        &numeric_axioms,
        layer_strategy,
    )?;

    info!(
        "{} effect conditions simplified",
        translation.simplified_effect_conditions
    );
    info!(
        "{} implied preconditions added",
        translation.added_implied_preconditions
    );

    // Filter unreachable facts
    if options::FILTER_UNREACHABLE_FACTS {
        let mut sas_task = sas_task;
        return match simplify::filter_unreachable_propositions(&mut sas_task) {
            Ok(()) => Ok(sas_task),
            // Naming every variant rather than using a wildcard makes a future
            // variant a compile error instead of a silent fallthrough:
            // `filter_unreachable_propositions` renames the task in place, so
            // returning `sas_task` for an unhandled error would hand back a
            // half-rewritten task.
            Err(simplify::SimplifyError::Impossible) => Ok(simplify::trivial_task(false)),
            Err(simplify::SimplifyError::TriviallySolvable) => Ok(simplify::trivial_task(true)),
        };
    }

    Ok(sas_task)
}

// ============================================================
// Helper functions
// ============================================================

/// Convert a Condition to an Atom for dictionary lookup
fn condition_to_atom(cond: &Condition) -> Option<Atom> {
    match cond {
        Condition::Atom(a) => Some(a.clone()),
        Condition::NegatedAtom(na) => {
            Some(Atom::new(format!("NOT-{}", na.predicate), na.args.clone()))
        }
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
            Some(Atom::new(name, vec![]))
        }
        _ => None,
    }
}

/// The comparison `comparator` relates `operands` by, asserted or denied.
fn make_fc_condition(
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
fn negate_condition(cond: &Condition) -> Condition {
    match cond {
        Condition::Atom(a) => Condition::NegatedAtom(a.negate()),
        Condition::NegatedAtom(na) => Condition::Atom(na.negate()),
        // Negating a comparison moves the same payload to the other variant.
        Condition::FunctionComparison(comparison) => {
            Condition::NegatedFunctionComparison(comparison.clone())
        }
        Condition::NegatedFunctionComparison(comparison) => {
            Condition::FunctionComparison(comparison.clone())
        }
        _ => cond.clone(),
    }
}

/// Extension trait for FunctionalExpression
impl FunctionalExpression {
    pub fn as_pne(&self) -> Option<&PrimitiveNumericExpression> {
        match self {
            FunctionalExpression::PrimitiveNumericExpression(pne) => Some(pne),
            _ => None,
        }
    }
}
