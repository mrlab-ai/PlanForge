/// Main translation from STRIPS/PDDL ground representation to SAS+ finite-domain representation.
use std::collections::{BTreeMap, HashMap, HashSet};

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
    count: usize,
    variables: HashMap<PrimitiveNumericExpression, usize>,
}

fn encode_numeric_fluents(
    num_axioms: &[InstantiatedNumericAxiom],
    equivalent: &HashMap<PrimitiveNumericExpression, PrimitiveNumericExpression>,
    num_fluents: &[PrimitiveNumericExpression],
) -> NumericEncoding {
    let mut variables: HashMap<PrimitiveNumericExpression, usize> = HashMap::new();
    let mut count = 0usize;

    let mut redundant = vec![];
    for axiom in num_axioms {
        if equivalent.contains_key(&axiom.effect) {
            redundant.push(axiom.effect.clone());
        } else {
            variables.insert(axiom.effect.clone(), count);
            count += 1;
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
        if !variables.contains_key(&fluent) {
            variables.insert(fluent, count);
            count += 1;
        }
    }

    NumericEncoding { count, variables }
}

// ============================================================
// translate_strips_conditions_aux
// ============================================================

/// Translates a conjunction of ground literals into the finite-domain
/// conditions that entail it, or `None` if it is unsatisfiable.
fn translate_strips_conditions_aux(
    conditions: &[Condition],
    dictionary: &mut HashMap<Atom, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_dictionary: &HashMap<PrimitiveNumericExpression, usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), Condition>,
    sas_comp_axioms: &mut Vec<SASCompareAxiom>,
    mutex_check: bool,
) -> Option<Vec<Assignment>> {
    let mut condition = Domains::default();

    for fact in conditions {
        match fact {
            Condition::FunctionComparison(_) | Condition::NegatedFunctionComparison(_) => {
                let (comparator, parts_fexpr, negated) = match fact {
                    Condition::FunctionComparison(fc) => (&fc.comparator, &fc.parts, false),
                    Condition::NegatedFunctionComparison(nfc) => {
                        (&nfc.comparator, &nfc.parts, true)
                    }
                    _ => unreachable!(),
                };

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

                let key = (comparator.clone(), parts.clone());

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
                        SASCompareAxiom::new(comparator.clone(), parts.clone(), ranges.len());

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
                    ranges.push(3);

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
                if let Some(pairs) = dictionary.get(atom) {
                    for &(var, val) in pairs {
                        if condition.get(var).is_some_and(|vals| !vals.contains(&val)) {
                            return None; // conflicting
                        }
                        condition.set_single(var, val);
                    }
                }
                // Static facts that aren't in dictionary can be ignored (they're static true)
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
    conditions: &[Condition],
    dictionary: &mut HashMap<Atom, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_dictionary: &HashMap<PrimitiveNumericExpression, usize>,
    mutex_dict: &mut HashMap<Atom, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), Condition>,
    sas_comp_axioms: &mut Vec<SASCompareAxiom>,
) -> Option<Vec<Assignment>> {
    if conditions.is_empty() {
        return Some(vec![Assignment::default()]); // Quick exit for common case
    }

    // Check if the condition violates any mutexes
    let mutex_result = translate_strips_conditions_aux(
        conditions,
        mutex_dict,
        mutex_ranges,
        numeric_dictionary,
        comp_axiom_dict,
        sas_comp_axioms,
        true,
    );
    mutex_result.as_ref()?;

    translate_strips_conditions_aux(
        conditions,
        dictionary,
        ranges,
        numeric_dictionary,
        comp_axiom_dict,
        sas_comp_axioms,
        false,
    )
}

// ============================================================
// translate_strips_operator
// ============================================================

fn translate_strips_operator(
    simplified_effect_condition_counter: &mut usize,
    added_implied_precondition_counter: &mut usize,
    operator: &PropositionalAction,
    dictionary: &mut HashMap<Atom, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_dictionary: &HashMap<PrimitiveNumericExpression, usize>,
    mutex_dict: &mut HashMap<Atom, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    implied_facts: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), Condition>,
    sas_comp_axioms: &mut Vec<SASCompareAxiom>,
    num_vals: &[f64],
    relevant_numeric_vars: &[usize],
) -> Vec<SASOperator> {
    let conditions = translate_strips_conditions(
        &operator.precondition,
        dictionary,
        ranges,
        numeric_dictionary,
        mutex_dict,
        mutex_ranges,
        comp_axiom_dict,
        sas_comp_axioms,
    );

    if conditions.is_none() {
        return vec![];
    }

    let mut sas_operators = vec![];
    for condition in conditions.unwrap() {
        if let Some(op) = translate_strips_operator_aux(
            simplified_effect_condition_counter,
            added_implied_precondition_counter,
            operator,
            dictionary,
            ranges,
            numeric_dictionary,
            mutex_dict,
            mutex_ranges,
            implied_facts,
            &condition,
            comp_axiom_dict,
            sas_comp_axioms,
            num_vals,
            relevant_numeric_vars,
        ) {
            sas_operators.push(op);
        }
    }
    sas_operators
}

// ============================================================
// negate_and_translate_condition
// ============================================================

fn negate_and_translate_condition(
    add_conds: &[Vec<Condition>],
    dictionary: &mut HashMap<Atom, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_dictionary: &HashMap<PrimitiveNumericExpression, usize>,
    mutex_dict: &mut HashMap<Atom, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), Condition>,
    sas_comp_axioms: &mut Vec<SASCompareAxiom>,
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
        let translated = translate_strips_conditions(
            &cond,
            dictionary,
            ranges,
            numeric_dictionary,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        );
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
    simplified_effect_condition_counter: &mut usize,
    added_implied_precondition_counter: &mut usize,
    operator: &PropositionalAction,
    dictionary: &mut HashMap<Atom, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_dictionary: &HashMap<PrimitiveNumericExpression, usize>,
    mutex_dict: &mut HashMap<Atom, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    implied_facts: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    condition: &Assignment,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), Condition>,
    sas_comp_axioms: &mut Vec<SASCompareAxiom>,
    num_vals: &[f64],
    relevant_numeric: &[usize],
) -> Option<SASOperator> {
    // Collect all add effects
    let mut effects_by_variable: HashMap<usize, HashMap<usize, Vec<Assignment>>> = HashMap::new();
    let mut add_conds_by_variable: HashMap<usize, Vec<Vec<Condition>>> = HashMap::new();

    for (conditions_list, fact) in &operator.add_effects {
        let eff_condition_list = translate_strips_conditions(
            conditions_list,
            dictionary,
            ranges,
            numeric_dictionary,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        );
        if eff_condition_list.is_none() {
            continue; // Impossible condition
        }
        if let Some(pairs) = dictionary.get(fact) {
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
        let eff_condition_list = translate_strips_conditions(
            conditions_list,
            dictionary,
            ranges,
            numeric_dictionary,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        );
        if eff_condition_list.is_none() {
            continue;
        }
        if let Some(pairs) = dictionary.get(fact) {
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
        let eff_condition_list = translate_strips_conditions(
            conditions_list,
            dictionary,
            ranges,
            numeric_dictionary,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        );
        if eff_condition_list.is_none() {
            continue;
        }
        if let Some(expr_pne) = assignment.expression.as_pne() {
            if let Some(&expr_var) = numeric_dictionary.get(expr_pne)
                && let Some(&fluent_var) = numeric_dictionary.get(&assignment.fluent)
            {
                ass_effects_by_variable
                    .entry(fluent_var)
                    .or_default()
                    .entry((assignment.symbol.clone(), expr_var))
                    .or_default()
                    .extend(eff_condition_list.unwrap());
            }
        } else {
            // Expression might be in numeric dictionary directly
            // Check if expression can be looked up
        }
    }

    if let Some(cost_assignment) = &operator.cost
        && let Some(expr_pne) = cost_assignment.expression.as_pne()
        && let Some(&expr_var) = numeric_dictionary.get(expr_pne)
        && let Some(&fluent_var) = numeric_dictionary.get(&cost_assignment.fluent)
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

        let no_add_effect_condition = negate_and_translate_condition(
            &add_conds,
            dictionary,
            ranges,
            numeric_dictionary,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        );

        if no_add_effect_condition.is_none() {
            continue; // Always an add effect
        }

        let none_of_those = ranges[var] - 1;
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

    build_sas_operator(
        simplified_effect_condition_counter,
        added_implied_precondition_counter,
        &operator.name,
        condition,
        &effects_by_variable,
        &ass_effects_by_variable,
        sas_operator_cost(
            &operator.name,
            operator.cost.as_ref(),
            numeric_dictionary,
            num_vals,
        ),
        ranges,
        implied_facts,
        relevant_numeric,
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
    action_name: &str,
    cost: Option<&FunctionAssignment>,
    numeric_dictionary: &HashMap<PrimitiveNumericExpression, usize>,
    num_vals: &[f64],
) -> f64 {
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
    simplified_effect_condition_counter: &mut usize,
    added_implied_precondition_counter: &mut usize,
    name: &str,
    condition: &Assignment,
    effects_by_variable: &HashMap<usize, HashMap<usize, Vec<Assignment>>>,
    ass_effects_by_variable: &HashMap<usize, HashMap<(String, usize), Vec<Assignment>>>,
    cost: f64,
    ranges: &[usize],
    implied_facts: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    relevant_numeric_variables: &[usize],
) -> Option<SASOperator> {
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
    axiom: &PropositionalAxiom,
    dictionary: &mut HashMap<Atom, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    num_dict: &HashMap<PrimitiveNumericExpression, usize>,
    mutex_dict: &mut HashMap<Atom, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), Condition>,
    sas_comp_axioms: &mut Vec<SASCompareAxiom>,
) -> Vec<SASAxiom> {
    let conditions = translate_strips_conditions(
        &axiom.condition,
        dictionary,
        ranges,
        num_dict,
        mutex_dict,
        mutex_ranges,
        comp_axiom_dict,
        sas_comp_axioms,
    );
    if conditions.is_none() {
        return vec![];
    }

    let effect = match &axiom.effect {
        Condition::NegatedAtom(natom) => {
            let pos_atom = Atom::new(natom.predicate.clone(), natom.args.clone());
            if let Some(pairs) = dictionary.get(&pos_atom) {
                let (var, _) = pairs[0];
                (var, ranges[var] - 1)
            } else {
                return vec![];
            }
        }
        Condition::Atom(atom) => {
            if let Some(pairs) = dictionary.get(atom) {
                pairs[0]
            } else {
                return vec![];
            }
        }
        _ => return vec![],
    };

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
    simplified_effect_condition_counter: &mut usize,
    added_implied_precondition_counter: &mut usize,
    actions: &[PropositionalAction],
    strips_to_sas: &mut HashMap<Atom, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_strips_to_sas: &HashMap<PrimitiveNumericExpression, usize>,
    mutex_dict: &mut HashMap<Atom, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    implied_facts: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), Condition>,
    sas_comp_axioms: &mut Vec<SASCompareAxiom>,
    num_vals: &[f64],
    relevant_numeric_vars: &[usize],
) -> Vec<SASOperator> {
    let mut result = vec![];
    for action in actions {
        let sas_ops = translate_strips_operator(
            simplified_effect_condition_counter,
            added_implied_precondition_counter,
            action,
            strips_to_sas,
            ranges,
            numeric_strips_to_sas,
            mutex_dict,
            mutex_ranges,
            implied_facts,
            comp_axiom_dict,
            sas_comp_axioms,
            num_vals,
            relevant_numeric_vars,
        );
        result.extend(sas_ops);
    }
    result
}

// ============================================================
// translate_strips_axioms
// ============================================================

fn translate_strips_axioms(
    axioms: &[PropositionalAxiom],
    strips_to_sas: &mut HashMap<Atom, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    num_dict: &HashMap<PrimitiveNumericExpression, usize>,
    mutex_dict: &mut HashMap<Atom, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), Condition>,
    sas_comp_axioms: &mut Vec<SASCompareAxiom>,
) -> Vec<SASAxiom> {
    let mut result = vec![];
    for axiom in axioms {
        let sas_axioms = translate_strips_axiom(
            axiom,
            strips_to_sas,
            ranges,
            num_dict,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        );
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
            "<none of those>".to_string(),
        ]);
    }
}

// ============================================================
// translate_task
// ============================================================

fn translate_task(
    simplified_effect_condition_counter: &mut usize,
    added_implied_precondition_counter: &mut usize,
    strips_to_sas: &mut HashMap<Atom, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    translation_key: &mut Vec<Vec<String>>,
    numeric_strips_to_sas: &HashMap<PrimitiveNumericExpression, usize>,
    num_count: usize,
    mutex_dict: &mut HashMap<Atom, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    mutex_key: &[Vec<(usize, usize)>],
    init: &[Atom],
    num_init: &[FunctionAssignment],
    goal_list: &[Condition],
    global_constraint: &Condition,
    actions: &[PropositionalAction],
    axioms: Vec<PropositionalAxiom>,
    num_axioms: &[InstantiatedNumericAxiom],
    num_axioms_by_layer: &BTreeMap<i32, Vec<InstantiatedNumericAxiom>>,
    num_axiom_map: &HashMap<PrimitiveNumericExpression, PrimitiveNumericExpression>,
    const_num_axioms: &HashSet<InstantiatedNumericAxiom>,
    metric: &(String, PrimitiveNumericExpression),
    implied_facts: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    init_constant_predicates: &[Atom],
    init_constant_numerics: &[FunctionAssignment],
) -> Result<SASTask, String> {
    // Process axioms
    let (processed_axioms, axiom_init, axiom_layer_dict) =
        axiom_rules::handle_axioms(actions, axioms, goal_list, global_constraint);

    // Extend init with axiom init atoms
    let mut full_init: Vec<Atom> = init.to_vec();
    full_init.extend(axiom_init);

    // Initialize init_values: Closed World Assumption
    let mut init_values: Vec<i32> = ranges.iter().map(|&r| (r as i32) - 1).collect();
    for fact in &full_init {
        if let Some(pairs) = strips_to_sas.get(fact) {
            for &(var, val) in pairs {
                let curr_val = init_values[var];
                if curr_val != (ranges[var] as i32 - 1) && curr_val != val as i32 {
                    return Err(format!("Inconsistent init facts! [fact = {:?}]", fact));
                }
                init_values[var] = val as i32;
            }
        }
    }

    // Comparison axioms tracking
    let mut comp_axiom_dict: HashMap<(String, Vec<usize>), Condition> = HashMap::new();
    let mut sas_comp_axioms: Vec<SASCompareAxiom> = vec![];

    // Translate goal
    let goal_dict_list = translate_strips_conditions(
        goal_list,
        strips_to_sas,
        ranges,
        numeric_strips_to_sas,
        mutex_dict,
        mutex_ranges,
        &mut comp_axiom_dict,
        &mut sas_comp_axioms,
    );

    // Translate global constraint
    let gc_as_list = vec![global_constraint.clone()];
    let global_constraint_dict_list = translate_strips_conditions(
        &gc_as_list,
        strips_to_sas,
        ranges,
        numeric_strips_to_sas,
        mutex_dict,
        mutex_ranges,
        &mut comp_axiom_dict,
        &mut sas_comp_axioms,
    );

    if goal_dict_list.is_none() {
        return Ok(trivial_task(false, "Goal violates a mutex"));
    }

    let goal_dict_list = goal_dict_list.unwrap();
    assert!(goal_dict_list.len() == 1, "Negative goal not supported");

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
    // Fluents without a SAS numeric variable are not dropped here: the caller
    // splits `num_init` on exactly the same predicate and hands those facts to
    // `SASTask` as `init_constant_numerics`, so skipping them is the
    // complementary half of that split.
    let mut num_init_values: Vec<f64> = vec![0.0; num_count];

    let mut relevant_numeric: Vec<usize> = vec![];
    for fact in num_init {
        let Some(&var) = numeric_strips_to_sas.get(&fact.fluent) else {
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
    for axiom in const_num_axioms {
        if let Some(&var) = numeric_strips_to_sas.get(&axiom.effect) {
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
    let operators = translate_strips_operators(
        simplified_effect_condition_counter,
        added_implied_precondition_counter,
        actions,
        strips_to_sas,
        ranges,
        numeric_strips_to_sas,
        mutex_dict,
        mutex_ranges,
        implied_facts,
        &mut comp_axiom_dict,
        &mut sas_comp_axioms,
        &num_init_values,
        &relevant_numeric,
    );

    // Translate axioms
    let sas_axioms = translate_strips_axioms(
        &processed_axioms,
        strips_to_sas,
        ranges,
        numeric_strips_to_sas,
        mutex_dict,
        mutex_ranges,
        &mut comp_axiom_dict,
        &mut sas_comp_axioms,
    );

    // Translate numeric axioms
    let const_num_axiom_effects: HashSet<PrimitiveNumericExpression> = const_num_axioms
        .iter()
        .map(|ax| ax.effect.clone())
        .collect();
    let sas_num_axioms: Vec<SASNumericAxiom> = num_axioms
        .iter()
        .filter(|ax| {
            !const_num_axiom_effects.contains(&ax.effect) && !num_axiom_map.contains_key(&ax.effect)
        })
        .filter_map(|ax| translate_numeric_axiom(ax, strips_to_sas, numeric_strips_to_sas))
        .collect();

    // Compute axiom layers
    let mut axiom_layers: Vec<i32> = vec![-1; ranges.len()];
    let mut num_axiom_layers: Vec<i32> = vec![-1; num_count];
    let mut num_axiom_layer = 0i32;

    for (&layer, layer_axioms) in num_axioms_by_layer {
        let mut sorted_axioms = layer_axioms.clone();
        sorted_axioms.sort_by(|a, b| a.name.cmp(&b.name));
        for axiom in &sorted_axioms {
            if !num_axiom_map.contains_key(&axiom.effect)
                && let Some(&var) = numeric_strips_to_sas.get(&axiom.effect)
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

    // Extend init with comparison axiom init values
    let comp_axiom_init: Vec<i32> = vec![2; sas_comp_axioms.len()]; // init to "none of those" (value 2)
    init_values.extend(comp_axiom_init);

    for axiom in &sas_comp_axioms {
        axiom_layers[axiom.effect] = num_axiom_layer;
    }

    for (atom, &layer) in &axiom_layer_dict {
        assert!(layer >= 0);
        if let Some(pairs) = strips_to_sas.get(atom) {
            let (var, _val) = pairs[0];
            axiom_layers[var] = layer + num_axiom_layer + 1;
        }
    }

    // Extend axiom_layers for comparison axiom variables
    while axiom_layers.len() < ranges.len() {
        axiom_layers.push(num_axiom_layer);
    }

    add_key_to_comp_axioms(&sas_comp_axioms, translation_key);

    let variables = SASVariables::new(
        ranges.clone(),
        axiom_layers,
        translation_key.clone(),
        num_axiom_layer,
    );

    // Build numeric variable names
    let mut num_variables: Vec<String> = vec![String::new(); num_count];
    let mut num_var_types: Vec<String> = vec!["U".to_string(); num_count];
    for (entry, &idx) in numeric_strips_to_sas {
        num_variables[idx] = format!("{}", entry);
        num_var_types[idx] = entry.ntype.to_string();
    }

    let numeric_variables =
        SASNumericVariables::new(num_variables, num_axiom_layers, num_var_types);

    let mutexes: Vec<SASMutexGroup> = mutex_key
        .iter()
        .map(|group| SASMutexGroup::new(group.clone()))
        .collect();

    let sas_init = SASInit::new(init_values, num_init_values);

    // Look up metric fluent
    let sas_metric = if metric.1.symbol.is_empty() || metric.1.ntype == 'X' {
        // Unit cost or special marker
        (metric.0.clone(), -1i64)
    } else {
        if let Some(&idx) = numeric_strips_to_sas.get(&metric.1) {
            (metric.0.clone(), idx as i64)
        } else {
            (metric.0.clone(), -1i64)
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

    Ok(SASTask::new(
        variables,
        numeric_variables,
        mutexes,
        sas_init,
        sas_goal,
        operators,
        sas_axioms,
        sas_comp_axioms,
        sas_num_axioms,
        gc_pair,
        sas_metric,
        init_constant_predicates.to_vec(),
        init_constant_numerics.to_vec(),
    ))
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
pub fn translate_task_from_grounded_internal(
    atoms: &[Atom],
    grounded_ops: &[PropositionalAction],
    _dom: &super::pddl_parser::lisp_parser::SExpr,
    _prob: &super::pddl_parser::lisp_parser::SExpr,
    num_fluents: &[PrimitiveNumericExpression],
    num_axioms: &[InstantiatedNumericAxiom],
    py_groups: Option<Vec<Vec<String>>>,
    grounded_axioms: &[PropositionalAxiom],
    reachable_action_params: &HashMap<String, Vec<Vec<String>>>,
    goal: &Condition,
    norm_task: &NormalizableTask,
) -> Result<SASTask, String> {
    let task = &norm_task.task;

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
    let num_fluents_set: HashSet<PrimitiveNumericExpression> =
        num_fluents_vec.iter().cloned().collect();

    // Compute fact groups
    let atoms_set: HashSet<Atom> = atoms.iter().cloned().collect();
    let fact_groups::FactGroups {
        groups,
        mutex_groups,
        mut translation_key,
    } = if py_groups.is_some() {
        // Fast path: skip invariant finding / mutex discovery.
        // This preserves semantics but produces a less compact encoding.
        fact_groups::compute_singleton_groups(&atoms_set)
    } else {
        fact_groups::compute_groups(task, &atoms_set, &Some(reachable_action_params.clone()))
    };

    let numeric_axiom_rules::NumericAxioms {
        axioms: processed_num_axioms,
        by_layer: num_axioms_by_layer,
        equivalent: num_axiom_map,
        constant: const_num_axioms,
        ..
    } = numeric_axiom_rules::handle_axioms(num_axioms);

    let FactEncoding {
        mut ranges,
        dictionary: mut strips_to_sas,
    } = encode_facts(&groups, options::USE_PARTIAL_ENCODING);
    let NumericEncoding {
        count: num_count,
        variables: numeric_strips_to_sas,
    } = encode_numeric_fluents(&processed_num_axioms, &num_axiom_map, &num_fluents_vec);

    // The mutex groups overlap, so they get their own variables, used only to
    // test whether a condition violates a mutex.
    let FactEncoding {
        ranges: mut mutex_ranges,
        dictionary: mut mutex_dict,
    } = encode_facts(&mutex_groups, false);

    // Build implied facts
    let implied_facts = if options::ADD_IMPLIED_PRECONDITIONS {
        build_implied_facts(&strips_to_sas, &groups, &mutex_groups)
    } else {
        HashMap::new()
    };

    // Build mutex key
    let mutex_key = build_mutex_key(&strips_to_sas, &mutex_groups);

    // Build goal list
    let goal_list: Vec<Condition> = match goal {
        Condition::Conjunction(conj) => conj.parts.clone(),
        other => vec![other.clone()],
    };

    for item in &goal_list {
        match item {
            Condition::Atom(_) | Condition::NegatedAtom(_) => {}
            _ => return Err(format!("Non-literal goal: {:?}", item)),
        }
    }

    let gc = &task.global_constraint;
    assert!(
        matches!(gc, Condition::Atom(_)),
        "Global constraint must be an atom literal"
    );

    let mut simplified_effect_condition_counter: usize = 0;
    let mut added_implied_precondition_counter: usize = 0;

    // Translate the task
    let sas_task = translate_task(
        &mut simplified_effect_condition_counter,
        &mut added_implied_precondition_counter,
        &mut strips_to_sas,
        &mut ranges,
        &mut translation_key,
        &numeric_strips_to_sas,
        num_count,
        &mut mutex_dict,
        &mut mutex_ranges,
        &mutex_key,
        &task.init,
        &task.num_init,
        &goal_list,
        gc,
        grounded_ops,
        grounded_axioms.to_vec(),
        &processed_num_axioms,
        &num_axioms_by_layer,
        &num_axiom_map,
        &const_num_axioms,
        &task.metric,
        &implied_facts,
        &task
            .init
            .iter()
            .filter(|a| !atoms_set.contains(a))
            .cloned()
            .collect::<Vec<_>>(),
        &task
            .num_init
            .iter()
            .filter(|a| !num_fluents_set.contains(&a.fluent))
            .cloned()
            .collect::<Vec<_>>(),
    )?;

    info!(
        "{} effect conditions simplified",
        simplified_effect_condition_counter
    );
    info!(
        "{} implied preconditions added",
        added_implied_precondition_counter
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
// dump_statistics
// ============================================================

pub fn dump_statistics(sas_task: &SASTask) {
    info!("Translator variables: {}", sas_task.variables.ranges.len());
    info!(
        "Translator derived variables: {}",
        sas_task
            .variables
            .axiom_layers
            .iter()
            .filter(|&&l| l >= 0)
            .count()
    );
    info!(
        "Translator facts: {}",
        sas_task.variables.ranges.iter().sum::<usize>()
    );
    info!("Translator goal facts: {}", sas_task.goal.pairs.len());
    info!("Translator mutex groups: {}", sas_task.mutexes.len());
    info!(
        "Translator total mutex groups size: {}",
        sas_task
            .mutexes
            .iter()
            .map(|m| m.get_encoding_size())
            .sum::<usize>()
    );
    info!("Translator operators: {}", sas_task.operators.len());
    info!("Translator axioms: {}", sas_task.axioms.len());
    info!("Translator task size: {}", sas_task.get_encoding_size());
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
        Condition::FunctionComparison(fc) => {
            let name = format!(
                "__fc_{}_{}",
                fc.comparator,
                fc.parts
                    .iter()
                    .map(|p| format!("{}", p))
                    .collect::<Vec<_>>()
                    .join("_")
            );
            Some(Atom::new(name, vec![]))
        }
        Condition::NegatedFunctionComparison(nfc) => {
            let name = format!(
                "__nfc_{}_{}",
                nfc.comparator,
                nfc.parts
                    .iter()
                    .map(|p| format!("{}", p))
                    .collect::<Vec<_>>()
                    .join("_")
            );
            Some(Atom::new(name, vec![]))
        }
        _ => None,
    }
}

/// Create a FunctionComparison or NegatedFunctionComparison condition
fn make_fc_condition(comparator: &str, parts: &[FunctionalExpression], negated: bool) -> Condition {
    if negated {
        Condition::NegatedFunctionComparison(NegatedFunctionComparison::new(
            comparator.to_string(),
            parts.to_vec(),
        ))
    } else {
        Condition::FunctionComparison(FunctionComparison::new(
            comparator.to_string(),
            parts.to_vec(),
        ))
    }
}

/// Negate a condition
fn negate_condition(cond: &Condition) -> Condition {
    match cond {
        Condition::Atom(a) => Condition::NegatedAtom(a.negate()),
        Condition::NegatedAtom(na) => Condition::Atom(na.negate()),
        Condition::FunctionComparison(fc) => Condition::NegatedFunctionComparison(fc.negate()),
        Condition::NegatedFunctionComparison(nfc) => Condition::FunctionComparison(nfc.negate()),
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
