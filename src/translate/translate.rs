use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::translate::build_model::Atom;
use crate::translate::instantiate;
use crate::translate::normalize;
use crate::translate::numeric_axiom_rules::{InstantiatedNumericAxiom, PrimitiveNumericExpression};
use crate::translate::pddl;
use crate::translate::pddl_parser::PddlTask;
use crate::translate::simplify;
use crate::translate::to_sas;
use crate::translate::options;
use crate::translate::sas::{CompareAxiom, SASAxiom, SASOperator, SASTask};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone)]
pub struct TranslateConfig {
    pub simplify: bool,
}

static SIMPLIFIED_EFFECT_CONDITION_COUNTER: AtomicUsize = AtomicUsize::new(0);
static ADDED_IMPLIED_PRECONDITION_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StripsAtom {
    pub predicate: String,
    pub args: Vec<String>,
    pub negated: bool,
}

impl StripsAtom {
    pub fn positive(&self) -> Self {
        Self {
            predicate: self.predicate.clone(),
            args: self.args.clone(),
            negated: false,
        }
    }

    pub fn negate(&self) -> Self {
        Self {
            predicate: self.predicate.clone(),
            args: self.args.clone(),
            negated: !self.negated,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionComparison {
    pub comparator: String,
    pub parts: Vec<String>,
    pub negated: bool,
}

impl FunctionComparison {
    pub fn negate(&self) -> Self {
        Self {
            comparator: self.comparator.clone(),
            parts: self.parts.clone(),
            negated: !self.negated,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum StripsFact {
    Atom(StripsAtom),
    Comparison(FunctionComparison),
}

impl StripsFact {
    pub fn is_negated(&self) -> bool {
        match self {
            StripsFact::Atom(atom) => atom.negated,
            StripsFact::Comparison(comp) => comp.negated,
        }
    }

    pub fn negate(&self) -> Self {
        match self {
            StripsFact::Atom(atom) => StripsFact::Atom(atom.negate()),
            StripsFact::Comparison(comp) => StripsFact::Comparison(comp.negate()),
        }
    }

    pub fn positive_atom(&self) -> Option<StripsAtom> {
        match self {
            StripsFact::Atom(atom) => Some(atom.positive()),
            _ => None,
        }
    }
}

impl Default for TranslateConfig {
    fn default() -> Self {
        Self { simplify: true }
    }
}

pub fn translate_from_files(domain: &Path, problem: &Path) -> Result<SASTask, String> {
    translate_from_files_with_config(domain, problem, &TranslateConfig::default())
}

pub fn translate_from_files_with_config(
    domain: &Path,
    problem: &Path,
    config: &TranslateConfig,
) -> Result<SASTask, String> {
    let task = PddlTask::from_files(domain, problem).map_err(|err| err.to_string())?;
    let dom = pddl::Domain::from_sexprs(&task.domain_forms)
        .ok_or_else(|| "failed to parse domain PDDL".to_string())?;
    let prob = pddl::Problem::from_sexprs(&task.problem_forms)
        .ok_or_else(|| "failed to parse problem PDDL".to_string())?;
    translate_from_ast(&dom, &prob, config)
}

pub fn translate_from_ast(
    dom: &pddl::Domain,
    prob: &pddl::Problem,
    config: &TranslateConfig,
) -> Result<SASTask, String> {
    let mut norm_task = normalize::NormalizableTask::from_ast(dom, prob);
    norm_task.add_global_constraints();
    normalize::normalize(&mut norm_task)?;

    let exploration = instantiate::explore_normalized(&norm_task)
        .map_err(|err| err.to_string())?;

    let py_groups: Option<Vec<Vec<String>>> = None;
    let mut sas_task = to_sas::build_sas(
        &exploration.grounded_ops,
        dom,
        prob,
        &exploration.numeric_axioms,
        py_groups,
        &exploration.grounded_axioms,
        &norm_task.goal,
        &norm_task,
    )
    .map_err(|err| err.to_string())?;

    if config.simplify {
        match simplify::filter_unreachable_propositions(&mut sas_task) {
            Ok(_) => {}
            Err(simplify::SimplifyError::Impossible) => {
                sas_task = simplify::trivial_task(false);
            }
            Err(simplify::SimplifyError::TriviallySolvable) => {
                sas_task = simplify::trivial_task(true);
            }
        }
    }

    Ok(sas_task)
}

fn format_pne(pne: &PrimitiveNumericExpression) -> String {
    if pne.args.is_empty() {
        if pne.name.ends_with(')') {
            pne.name.clone()
        } else {
            format!("{}()", pne.name)
        }
    } else {
        format!("{}({})", pne.name, pne.args.join(", "))
    }
}

pub fn strips_to_sas_dictionary(
    groups: &[Vec<Atom>],
    num_axioms: &[InstantiatedNumericAxiom],
    num_axiom_map: &HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom>,
    num_fluents: &HashSet<String>,
    assert_partial: bool,
    include_numeric: bool,
) -> (
    Vec<usize>,
    HashMap<Atom, Vec<(usize, usize)>>,
    usize,
    HashMap<String, usize>,
) {
    let mut dictionary: HashMap<Atom, Vec<(usize, usize)>> = HashMap::new();
    for (var_no, group) in groups.iter().enumerate() {
        for (val_no, atom) in group.iter().enumerate() {
            dictionary
                .entry(atom.clone())
                .or_default()
                .push((var_no, val_no));
        }
    }
    if assert_partial {
        for pairs in dictionary.values() {
            assert!(pairs.len() == 1);
        }
    }

    let ranges: Vec<usize> = groups.iter().map(|g| g.len() + 1).collect();

    let mut numeric_dictionary: HashMap<String, usize> = HashMap::new();
    let mut num_count = 0;

    if include_numeric {
        let mut redundant_axioms: Vec<PrimitiveNumericExpression> = Vec::new();
        for axiom in num_axioms {
            if num_axiom_map.contains_key(&axiom.effect) {
                redundant_axioms.push(axiom.effect.clone());
            } else {
                let key = format_pne(&axiom.effect);
                numeric_dictionary.insert(key, num_count);
                num_count += 1;
            }
        }

        for axiom_effect in redundant_axioms {
            if let Some(mapped) = num_axiom_map.get(&axiom_effect) {
                let key = format_pne(&axiom_effect);
                let mapped_key = format_pne(&mapped.effect);
                if let Some(idx) = numeric_dictionary.get(&mapped_key) {
                    numeric_dictionary.insert(key, *idx);
                }
            }
        }

        let mut fluent_list: Vec<String> = num_fluents.iter().cloned().collect();
        fluent_list.sort();
        for fluent in fluent_list {
            if !numeric_dictionary.contains_key(&fluent) {
                numeric_dictionary.insert(fluent, num_count);
                num_count += 1;
            }
        }
    }

    (ranges, dictionary, num_count, numeric_dictionary)
}

pub fn build_mutex_key(
    strips_to_sas: &HashMap<Atom, Vec<(usize, usize)>>,
    groups: &[Vec<Atom>],
) -> Vec<Vec<(usize, usize)>> {
    let mut group_keys = Vec::new();
    for group in groups {
        let mut group_key = Vec::new();
        for fact in group {
            if let Some(entries) = strips_to_sas.get(fact) {
                for (var, val) in entries {
                    group_key.push((*var, *val));
                }
            }
        }
        group_keys.push(group_key);
    }
    group_keys
}

pub fn build_implied_facts(
    strips_to_sas: &HashMap<Atom, Vec<(usize, usize)>>,
    groups: &[Vec<Atom>],
    mutex_groups: &[Vec<Atom>],
) -> HashMap<(usize, usize), Vec<(usize, usize)>> {
    let mut lonely_propositions: HashMap<Atom, usize> = HashMap::new();
    for (var_no, group) in groups.iter().enumerate() {
        if group.len() == 1 {
            let lonely_prop = group[0].clone();
            lonely_propositions.insert(lonely_prop, var_no);
        }
    }

    let mut implied: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for mutex_group in mutex_groups {
        for prop in mutex_group {
            if let Some(prop_var) = lonely_propositions.get(prop) {
                let prop_is_false = (*prop_var, 1);
                for other_prop in mutex_group {
                    if other_prop != prop {
                        if let Some(other_facts) = strips_to_sas.get(other_prop) {
                            for other_fact in other_facts {
                                implied
                                    .entry(*other_fact)
                                    .or_default()
                                    .push(prop_is_false);
                            }
                        }
                    }
                }
            }
        }
    }

    implied
}

pub fn dump_statistics(task: &SASTask) {
    let derived_vars = task.axiom_layers.iter().filter(|layer| **layer >= 0).count();
    let total_facts: usize = task.ranges.iter().sum();
    let mutex_groups = task.mutex_groups.len();
    let mutex_size: usize = task.mutex_groups.iter().map(|g| g.len()).sum();

    println!("Translator variables: {}", task.ranges.len());
    println!("Translator derived variables: {}", derived_vars);
    println!("Translator facts: {}", total_facts);
    println!("Translator goal facts: {}", task.goal.len());
    println!("Translator mutex groups: {}", mutex_groups);
    println!("Translator total mutex groups size: {}", mutex_size);
    println!("Translator operators: {}", task.operators.len());
    println!("Translator axioms: {}", task.axioms.len());
    println!("Translator task size: {}", task.variables.len());
}

pub fn translate_strips_conditions_aux(
    conditions: &[StripsFact],
    dictionary: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_dictionary: &HashMap<String, usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), StripsFact>,
    sas_comp_axioms: &mut Vec<CompareAxiom>,
    mutex_check: bool,
) -> Option<Vec<HashMap<usize, usize>>> {
    let mut condition: HashMap<usize, HashSet<usize>> = HashMap::new();

    for fact in conditions {
        match fact {
            StripsFact::Comparison(comp) => {
                if !dictionary.contains_key(fact) {
                    for part in &comp.parts {
                        assert!(numeric_dictionary.contains_key(part));
                    }
                    let parts: Vec<usize> = comp
                        .parts
                        .iter()
                        .map(|p| numeric_dictionary[p])
                        .collect();
                    let key = (comp.comparator.clone(), parts.clone());
                    let mut fact_to_use = StripsFact::Comparison(comp.clone());
                    if let Some(pos_fact) = comp_axiom_dict.get(&key) {
                        if comp.negated {
                            fact_to_use = pos_fact.negate();
                        } else {
                            fact_to_use = pos_fact.clone();
                        }
                    } else {
                        let axiom = CompareAxiom {
                            comp: comp.comparator.clone(),
                            parts: parts.clone(),
                            effect_var: ranges.len(),
                        };
                        let (pos_fact, neg_fact) = if comp.negated {
                            (fact_to_use.negate(), fact_to_use)
                        } else {
                            (fact_to_use.clone(), fact_to_use.negate())
                        };
                        if !mutex_check {
                            sas_comp_axioms.push(axiom);
                            comp_axiom_dict.insert(key, pos_fact.clone());
                        }
                        dictionary
                            .entry(pos_fact)
                            .or_default()
                            .push((ranges.len(), 0));
                        dictionary
                            .entry(neg_fact)
                            .or_default()
                            .push((ranges.len(), 1));
                        ranges.push(3);
                        fact_to_use = if comp.negated {
                            StripsFact::Comparison(comp.negate())
                        } else {
                            StripsFact::Comparison(comp.clone())
                        };
                    }

                    if let Some(entry) = dictionary.get(&fact_to_use).and_then(|v| v.first()) {
                        let (var, val) = *entry;
                        if let Some(existing) = condition.get(&var) {
                            if !existing.contains(&val) {
                                return None;
                            }
                        }
                        condition.insert(var, HashSet::from([val]));
                    }
                } else if let Some(entry) = dictionary.get(fact).and_then(|v| v.first()) {
                    let (var, val) = *entry;
                    if let Some(existing) = condition.get(&var) {
                        if !existing.contains(&val) {
                            return None;
                        }
                    }
                    condition.insert(var, HashSet::from([val]));
                }
            }
            StripsFact::Atom(atom) => {
                if atom.negated {
                    continue;
                }
                if let Some(entries) = dictionary.get(fact) {
                    for (var, val) in entries {
                        if let Some(existing) = condition.get(var) {
                            if !existing.contains(val) {
                                return None;
                            }
                        }
                        condition.insert(*var, HashSet::from([*val]));
                    }
                }
            }
        }
    }

    let number_of_values = |vals: &HashSet<usize>| vals.len();

    for fact in conditions {
        if matches!(fact, StripsFact::Comparison(_)) {
            continue;
        }
        if fact.is_negated() {
            let mut done = false;
            let mut new_condition: HashMap<usize, HashSet<usize>> = HashMap::new();
            if let Some(atom) = fact.positive_atom() {
                let positive_fact = StripsFact::Atom(atom);
                if let Some(entries) = dictionary.get(&positive_fact) {
                    for (var, val) in entries {
                        let mut poss_vals: HashSet<usize> =
                            (0..ranges[*var]).collect();
                        poss_vals.remove(val);

                        if !condition.contains_key(var) {
                            if new_condition.contains_key(var) {
                                continue;
                            }
                            new_condition.insert(*var, poss_vals);
                        } else if let Some(existing) = condition.get_mut(var) {
                            done = true;
                            existing.retain(|v| poss_vals.contains(v));
                            if existing.is_empty() {
                                return None;
                            }
                        }
                    }
                }
            }

            if !done && !new_condition.is_empty() {
                let mut candidates: Vec<(usize, HashSet<usize>)> = new_condition
                    .into_iter()
                    .collect();
                candidates.sort_by_key(|(_, vals)| number_of_values(vals));
                if let Some((var, vals)) = candidates.into_iter().next() {
                    condition.insert(var, vals);
                }
            }
        }
    }

    let mut sorted_conds: Vec<(usize, HashSet<usize>)> = condition.into_iter().collect();
    sorted_conds.sort_by_key(|(_, vals)| number_of_values(vals));

    let mut flat_conds: Vec<HashMap<usize, usize>> = vec![HashMap::new()];
    for (var, vals) in sorted_conds {
        if vals.len() == 1 {
            let val = *vals.iter().next().unwrap();
            for cond in &mut flat_conds {
                cond.insert(var, val);
            }
        } else {
            let mut new_conds = Vec::new();
            for cond in &flat_conds {
                for val in &vals {
                    let mut new_cond = cond.clone();
                    new_cond.insert(var, *val);
                    new_conds.push(new_cond);
                }
            }
            flat_conds = new_conds;
        }
    }

    Some(flat_conds)
}

pub fn translate_strips_conditions(
    conditions: &[StripsFact],
    dictionary: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_dictionary: &HashMap<String, usize>,
    mutex_dict: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), StripsFact>,
    sas_comp_axioms: &mut Vec<CompareAxiom>,
) -> Option<Vec<HashMap<usize, usize>>> {
    if conditions.is_empty() {
        return Some(vec![HashMap::new()]);
    }

    if translate_strips_conditions_aux(
        conditions,
        mutex_dict,
        mutex_ranges,
        numeric_dictionary,
        comp_axiom_dict,
        sas_comp_axioms,
        true,
    )
    .is_none()
    {
        return None;
    }

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

#[derive(Clone, Debug)]
pub struct AssignmentEffect {
    pub fluent: String,
    pub symbol: String,
    pub expression: String,
}

#[derive(Clone, Debug)]
pub struct StripsOperator {
    pub name: String,
    pub precondition: Vec<StripsFact>,
    pub add_effects: Vec<(Vec<StripsFact>, StripsFact)>,
    pub del_effects: Vec<(Vec<StripsFact>, StripsFact)>,
    pub assign_effects: Vec<(Vec<StripsFact>, AssignmentEffect)>,
    pub cost: f64,
}

#[derive(Clone, Debug)]
pub struct StripsAxiom {
    pub condition: Vec<StripsFact>,
    pub effect: StripsAtom,
}

pub fn negate_and_translate_condition(
    condition: &[Vec<StripsFact>],
    dictionary: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_dict: &HashMap<String, usize>,
    mutex_dict: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), StripsFact>,
    sas_comp_axioms: &mut Vec<CompareAxiom>,
) -> Option<Vec<HashMap<usize, usize>>> {
    if condition.iter().any(|clause| clause.is_empty()) {
        return None;
    }

    let mut combinations: Vec<Vec<StripsFact>> = vec![Vec::new()];
    for clause in condition {
        let mut next = Vec::new();
        for combo in &combinations {
            for lit in clause {
                let mut new_combo = combo.clone();
                new_combo.push(lit.clone());
                next.push(new_combo);
            }
        }
        combinations = next;
    }

    let mut negation = Vec::new();
    for combination in combinations {
        let negated: Vec<StripsFact> = combination
            .iter()
            .map(|lit| lit.negate())
            .collect();
        if let Some(cond) = translate_strips_conditions(
            &negated,
            dictionary,
            ranges,
            numeric_dict,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        ) {
            negation.extend(cond);
        }
    }
    if negation.is_empty() {
        None
    } else {
        Some(negation)
    }
}

pub fn translate_strips_operator(
    operator: &StripsOperator,
    dictionary: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_dictionary: &HashMap<String, usize>,
    mutex_dict: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    implied_facts: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), StripsFact>,
    sas_comp_axioms: &mut Vec<CompareAxiom>,
    num_vals: usize,
    relevant_numeric_vars: &HashSet<usize>,
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
    let mut sas_operators = Vec::new();
    if let Some(condition_list) = conditions {
        for condition in condition_list {
            if let Some(op) = translate_strips_operator_aux(
                operator,
                dictionary,
                ranges,
                numeric_dictionary,
                mutex_dict,
                mutex_ranges,
                implied_facts,
                condition,
                comp_axiom_dict,
                sas_comp_axioms,
                num_vals,
                relevant_numeric_vars,
            ) {
                sas_operators.push(op);
            }
        }
    }
    sas_operators
}

pub fn translate_strips_operator_aux(
    operator: &StripsOperator,
    dictionary: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_dictionary: &HashMap<String, usize>,
    mutex_dict: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    implied_facts: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    mut condition: HashMap<usize, usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), StripsFact>,
    sas_comp_axioms: &mut Vec<CompareAxiom>,
    _num_vals: usize,
    relevant_numeric: &HashSet<usize>,
) -> Option<SASOperator> {
    let mut effects_by_variable: HashMap<usize, HashMap<usize, Vec<HashMap<usize, usize>>>> =
        HashMap::new();
    let mut add_conds_by_variable: HashMap<usize, Vec<Vec<StripsFact>>> = HashMap::new();

    for (conds, fact) in &operator.add_effects {
        let eff_condition_list = translate_strips_conditions(
            conds,
            dictionary,
            ranges,
            numeric_dictionary,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        )?;
        if let Some(entries) = dictionary.get(fact) {
            for (var, val) in entries {
                effects_by_variable
                    .entry(*var)
                    .or_default()
                    .entry(*val)
                    .or_default()
                    .extend(eff_condition_list.clone());
                add_conds_by_variable
                    .entry(*var)
                    .or_default()
                    .push(conds.clone());
            }
        }
    }

    let mut del_effects_by_variable: HashMap<usize, HashMap<usize, Vec<HashMap<usize, usize>>>> =
        HashMap::new();
    for (conds, fact) in &operator.del_effects {
        let eff_condition_list = translate_strips_conditions(
            conds,
            dictionary,
            ranges,
            numeric_dictionary,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        )?;
        if let Some(entries) = dictionary.get(fact) {
            for (var, val) in entries {
                del_effects_by_variable
                    .entry(*var)
                    .or_default()
                    .entry(*val)
                    .or_default()
                    .extend(eff_condition_list.clone());
            }
        }
    }

    let mut ass_effects_by_variable: HashMap<usize, HashMap<(String, usize), Vec<HashMap<usize, usize>>>> =
        HashMap::new();
    for (conds, assignment) in &operator.assign_effects {
        let eff_condition_list = translate_strips_conditions(
            conds,
            dictionary,
            ranges,
            numeric_dictionary,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        )?;
        if let Some(expr_idx) = numeric_dictionary.get(&assignment.expression) {
            if let Some(fluent_idx) = numeric_dictionary.get(&assignment.fluent) {
                ass_effects_by_variable
                    .entry(*fluent_idx)
                    .or_default()
                    .entry((assignment.symbol.clone(), *expr_idx))
                    .or_default()
                    .extend(eff_condition_list);
            }
        }
    }

    for (var, del_effects) in del_effects_by_variable {
        let no_add_effect_condition = negate_and_translate_condition(
            add_conds_by_variable.get(&var).map(|v| v.as_slice()).unwrap_or(&[]),
            dictionary,
            ranges,
            numeric_dictionary,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        );
        if no_add_effect_condition.is_none() {
            continue;
        }
        let no_add_effect_condition = no_add_effect_condition.unwrap();
        let none_of_those = ranges[var] - 1;
        for (val, conds) in del_effects {
            for mut cond in conds {
                if let Some(existing) = cond.get(&var) {
                    if *existing != val {
                        continue;
                    }
                }
                cond.insert(var, val);
                for no_add_cond in &no_add_effect_condition {
                    let mut new_cond = cond.clone();
                    let mut ok = true;
                    for (cvar, cval) in no_add_cond {
                        if let Some(existing) = new_cond.get(cvar) {
                            if *existing != *cval {
                                ok = false;
                                break;
                            }
                        }
                        new_cond.insert(*cvar, *cval);
                    }
                    if ok {
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
        &operator.name,
        &mut condition,
        effects_by_variable,
        ass_effects_by_variable,
        operator.cost,
        ranges,
        implied_facts,
        relevant_numeric,
    )
}

pub fn build_sas_operator(
    name: &str,
    condition: &mut HashMap<usize, usize>,
    effects_by_variable: HashMap<usize, HashMap<usize, Vec<HashMap<usize, usize>>>>,
    ass_effects_by_variable: HashMap<usize, HashMap<(String, usize), Vec<HashMap<usize, usize>>>>,
    deprecated_cost: f64,
    ranges: &[usize],
    implied_facts: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    relevant_numeric_variables: &HashSet<usize>,
) -> Option<SASOperator> {
    let add_implied = options::get()
        .map(|opts| opts.add_implied_preconditions)
        .unwrap_or(false);
    let mut implied_precondition: HashSet<(usize, usize)> = HashSet::new();
    if add_implied {
        for (var, val) in condition.iter() {
            if let Some(implied) = implied_facts.get(&(*var, *val)) {
                implied_precondition.extend(implied);
            }
        }
    }

    let mut prevail_and_pre = condition.clone();
    let mut pre_post = Vec::new();
    let mut num_pre_post = Vec::new();

    for (var, effects) in effects_by_variable {
        let orig_pre = *condition.get(&var).unwrap_or(&usize::MAX);
        let mut added_effect = false;
        for (post, eff_conditions) in effects {
            let mut pre = orig_pre;
            if pre == post {
                continue;
            }
            let mut eff_condition_lists: Vec<Vec<(usize, usize)>> = eff_conditions
                .iter()
                .map(|eff_cond| {
                    let mut v: Vec<(usize, usize)> = eff_cond.iter().map(|(k, v)| (*k, *v)).collect();
                    v.sort();
                    v
                })
                .collect();
            if ranges[var] == 2 {
                if prune_stupid_effect_conditions(var, post, &mut eff_condition_lists) {
                    SIMPLIFIED_EFFECT_CONDITION_COUNTER.fetch_add(1, Ordering::Relaxed);
                }
                if add_implied && pre == usize::MAX && implied_precondition.contains(&(var, 1 - post)) {
                    ADDED_IMPLIED_PRECONDITION_COUNTER.fetch_add(1, Ordering::Relaxed);
                    pre = 1 - post;
                }
            }
            for eff_condition in eff_condition_lists {
                let mut filtered = Vec::new();
                let mut contradicts = false;
                for (variable, value) in eff_condition {
                    if let Some(prev) = prevail_and_pre.get(&variable) {
                        if *prev != value {
                            contradicts = true;
                            break;
                        }
                    } else {
                        filtered.push((variable, value));
                    }
                }
                if contradicts {
                    continue;
                }
                pre_post.push((var, if pre == usize::MAX { 0 } else { pre }, post, filtered));
                added_effect = true;
            }
        }
        if added_effect {
            condition.remove(&var);
        }
    }

    for (numvar, effects) in ass_effects_by_variable {
        for ((ass_op, post_var), eff_conditions) in effects {
            let eff_condition_lists: Vec<Vec<(usize, usize)>> = eff_conditions
                .iter()
                .map(|eff_cond| {
                    let mut v: Vec<(usize, usize)> = eff_cond.iter().map(|(k, v)| (*k, *v)).collect();
                    v.sort();
                    v
                })
                .collect();
            for eff_condition in eff_condition_lists {
                num_pre_post.push((numvar, ass_op.clone(), post_var, eff_condition));
            }
        }
    }

    if pre_post.is_empty() {
        let mut irrelevant = true;
        for effect in &num_pre_post {
            if relevant_numeric_variables.contains(&effect.0) {
                irrelevant = false;
                break;
            }
        }
        if irrelevant {
            return None;
        }
    }

    let prevail = condition.iter().map(|(k, v)| (*k, *v)).collect();
    Some(SASOperator {
        name: name.to_string(),
        prevails: prevail,
        effects: pre_post,
        numeric_effects: num_pre_post,
        cost: deprecated_cost,
    })
}

pub fn prune_stupid_effect_conditions(
    var: usize,
    val: usize,
    conditions: &mut Vec<Vec<(usize, usize)>>,
) -> bool {
    if conditions == &vec![Vec::new()] {
        return false;
    }
    let dual_fact = (var, 1 - val);
    let mut simplified = false;
    for condition in conditions.iter_mut() {
        while let Some(pos) = condition.iter().position(|v| *v == dual_fact) {
            condition.remove(pos);
            simplified = true;
        }
        if condition.is_empty() {
            conditions.clear();
            conditions.push(Vec::new());
            simplified = true;
            break;
        }
    }
    simplified
}

pub fn translate_strips_axiom(
    axiom: &StripsAxiom,
    dictionary: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    num_dict: &HashMap<String, usize>,
    mutex_dict: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), StripsFact>,
    sas_comp_axioms: &mut Vec<CompareAxiom>,
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
    let mut axioms = Vec::new();
    if let Some(cond_list) = conditions {
        let effect = if axiom.effect.negated {
            let positive = StripsFact::Atom(axiom.effect.positive());
            if let Some(entries) = dictionary.get(&positive) {
                let (var, _) = entries[0];
                (var, ranges[var] - 1)
            } else {
                return axioms;
            }
        } else if let Some(entries) = dictionary.get(&StripsFact::Atom(axiom.effect.clone())) {
            entries[0]
        } else {
            return axioms;
        };
        for condition in cond_list {
            axioms.push(SASAxiom {
                condition: condition.iter().map(|(k, v)| (*k, *v)).collect(),
                effect,
            });
        }
    }
    axioms
}

pub fn translate_numeric_axiom(
    axiom: &InstantiatedNumericAxiom,
    prop_dictionary: &HashMap<StripsFact, Vec<(usize, usize)>>,
    num_dictionary: &HashMap<String, usize>,
) -> Option<crate::translate::sas::NumericAxiom> {
    let effect = num_dictionary.get(&format_pne(&axiom.effect)).copied()?;
    let mut parts = Vec::new();
    for part in &axiom.parts {
        match part {
            crate::translate::numeric_axiom_rules::NumericPart::Primitive(pne) => {
                parts.push(*num_dictionary.get(&format_pne(pne))?);
            }
            crate::translate::numeric_axiom_rules::NumericPart::Axiom(sub) => {
                let key = format_pne(&sub.effect);
                if let Some(idx) = num_dictionary.get(&key) {
                    parts.push(*idx);
                } else if let Some(entries) = prop_dictionary.get(&StripsFact::Atom(StripsAtom {
                    predicate: key,
                    args: Vec::new(),
                    negated: false,
                })) {
                    parts.push(entries[0].0);
                } else {
                    return None;
                }
            }
            crate::translate::numeric_axiom_rules::NumericPart::Constant(constant) => {
                let key = constant.0.to_string();
                parts.push(*num_dictionary.get(&key)?);
            }
        }
    }

    Some(crate::translate::sas::NumericAxiom {
        op: axiom.op.clone().unwrap_or_default(),
        parts,
        effect,
    })
}

pub fn translate_strips_operators(
    actions: &[StripsOperator],
    strips_to_sas: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    numeric_strips_to_sas: &HashMap<String, usize>,
    mutex_dict: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    implied_facts: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), StripsFact>,
    sas_comp_axioms: &mut Vec<CompareAxiom>,
    num_vals: usize,
    relevant_numeric_vars: &HashSet<usize>,
) -> Vec<SASOperator> {
    let mut result = Vec::new();
    for action in actions {
        let mut ops = translate_strips_operator(
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
        result.append(&mut ops);
    }
    result
}

pub fn translate_strips_axioms(
    axioms: &[StripsAxiom],
    strips_to_sas: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    num_dict: &HashMap<String, usize>,
    mutex_dict: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    comp_axiom_dict: &mut HashMap<(String, Vec<usize>), StripsFact>,
    sas_comp_axioms: &mut Vec<CompareAxiom>,
) -> Vec<SASAxiom> {
    let mut result = Vec::new();
    for axiom in axioms {
        let mut sas_axioms = translate_strips_axiom(
            axiom,
            strips_to_sas,
            ranges,
            num_dict,
            mutex_dict,
            mutex_ranges,
            comp_axiom_dict,
            sas_comp_axioms,
        );
        result.append(&mut sas_axioms);
    }
    result
}

pub fn add_key_to_comp_axioms(
    comparison_axioms: &mut Vec<CompareAxiom>,
    translation_key: &mut Vec<Vec<String>>,
) {
    for axiom in comparison_axioms {
        let mut value_list = Vec::new();
        value_list.push(format!("{} {:?}", axiom.comp, axiom.parts));
        value_list.push(format!("not({} {:?})", axiom.comp, axiom.parts));
        value_list.push("<none of those>".to_string());
        translation_key.push(value_list);
    }
}

pub fn dump_task(
    init: &[StripsFact],
    goals: &[StripsFact],
    actions: &[StripsOperator],
    axioms: &[StripsAxiom],
    axiom_layer_dict: &HashMap<String, usize>,
) -> Result<(), String> {
    let mut out = String::new();
    out.push_str("Initial state\n");
    for atom in init {
        out.push_str(&format!("{:?}\n", atom));
    }
    out.push_str("\nGoals\n");
    for goal in goals {
        out.push_str(&format!("{:?}\n", goal));
    }
    for action in actions {
        out.push_str("\nAction\n");
        out.push_str(&format!("{:?}\n", action));
    }
    for axiom in axioms {
        out.push_str("\nAxiom\n");
        out.push_str(&format!("{:?}\n", axiom));
    }
    out.push_str("\nAxiom layers\n");
    for (atom, layer) in axiom_layer_dict {
        out.push_str(&format!("{}: layer {}\n", atom, layer));
    }
    std::fs::write("output.dump", out)
        .map_err(|err| format!("failed to write output.dump: {}", err))
}

pub fn trivial_task(solvable: bool) -> SASTask {
    simplify::trivial_task(solvable)
}

pub fn solvable_sas_task(_msg: &str) -> SASTask {
    simplify::trivial_task(true)
}

pub fn unsolvable_sas_task(_msg: &str) -> SASTask {
    simplify::trivial_task(false)
}

pub fn pddl_to_sas(dom: &pddl::Domain, prob: &pddl::Problem) -> Result<SASTask, String> {
    translate_from_ast(dom, prob, &TranslateConfig::default())
}

pub fn translate_task(
    strips_to_sas: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    ranges: &mut Vec<usize>,
    translation_key: &mut Vec<Vec<String>>,
    numeric_strips_to_sas: &HashMap<String, usize>,
    num_count: usize,
    mutex_dict: &mut HashMap<StripsFact, Vec<(usize, usize)>>,
    mutex_ranges: &mut Vec<usize>,
    mutex_key: &[Vec<(usize, usize)>],
    init: &[StripsFact],
    num_init: &HashMap<String, f64>,
    goal_list: &[StripsFact],
    global_constraint: &StripsFact,
    actions: &[StripsOperator],
    axioms: &[StripsAxiom],
    num_axioms: &[InstantiatedNumericAxiom],
    _num_axioms_by_layer: &HashMap<i32, Vec<InstantiatedNumericAxiom>>,
    _num_axiom_map: &HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom>,
    _const_num_axioms: &[InstantiatedNumericAxiom],
    metric: (String, isize),
    implied_facts: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    _init_constant_predicates: &[StripsFact],
    _init_constant_numerics: &HashMap<String, f64>,
) -> SASTask {
    let mut init_values: Vec<i32> = ranges.iter().map(|r| (r - 1) as i32).collect();
    for fact in init {
        if let Some(pairs) = strips_to_sas.get(fact) {
            for (var, val) in pairs {
                init_values[*var] = *val as i32;
            }
        }
    }

    let mut comp_axiom_dict: HashMap<(String, Vec<usize>), StripsFact> = HashMap::new();
    let mut sas_comp_axioms: Vec<CompareAxiom> = Vec::new();

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
    if goal_dict_list.is_none() {
        return unsolvable_sas_task("Goal violates a mutex");
    }
    let goal_dict_list = goal_dict_list.unwrap();
    let goal_pairs: Vec<(usize, usize)> = goal_dict_list
        .get(0)
        .map(|d| d.iter().map(|(k, v)| (*k, *v)).collect())
        .unwrap_or_default();
    if goal_pairs.is_empty() {
        return solvable_sas_task("Empty goal");
    }

    let global_constraint_dict_list = translate_strips_conditions(
        std::slice::from_ref(global_constraint),
        strips_to_sas,
        ranges,
        numeric_strips_to_sas,
        mutex_dict,
        mutex_ranges,
        &mut comp_axiom_dict,
        &mut sas_comp_axioms,
    )
    .unwrap_or_default();
    let global_constraint_pair = global_constraint_dict_list
        .get(0)
        .and_then(|d| d.iter().next())
        .map(|(k, v)| (*k, *v));

    add_key_to_comp_axioms(&mut sas_comp_axioms, translation_key);

    let mut num_init_values = vec![0.0; num_count];
    for (name, value) in num_init {
        if let Some(idx) = numeric_strips_to_sas.get(name) {
            if *idx < num_init_values.len() {
                num_init_values[*idx] = *value;
            }
        }
    }

    let relevant_numeric_vars: HashSet<usize> = numeric_strips_to_sas.values().copied().collect();

    let operators = translate_strips_operators(
        actions,
        strips_to_sas,
        ranges,
        numeric_strips_to_sas,
        mutex_dict,
        mutex_ranges,
        implied_facts,
        &mut comp_axiom_dict,
        &mut sas_comp_axioms,
        num_count,
        &relevant_numeric_vars,
    );
    let axioms_out = translate_strips_axioms(
        axioms,
        strips_to_sas,
        ranges,
        numeric_strips_to_sas,
        mutex_dict,
        mutex_ranges,
        &mut comp_axiom_dict,
        &mut sas_comp_axioms,
    );

    let numeric_axioms: Vec<crate::translate::sas::NumericAxiom> = num_axioms
        .iter()
        .filter_map(|ax| translate_numeric_axiom(ax, strips_to_sas, numeric_strips_to_sas))
        .collect();

    let variables = translation_key
        .iter()
        .map(|values| crate::translate::sas::Variable {
            value_names: values.clone(),
        })
        .collect();

    SASTask {
        variables,
        operators,
        numeric_variables: Vec::new(),
        numeric_axioms,
        comparison_axioms: sas_comp_axioms
            .iter()
            .map(|ax| crate::translate::sas::CompareAxiom {
                comp: ax.comp.clone(),
                parts: ax.parts.clone(),
                effect_var: ax.effect_var,
            })
            .collect(),
        axioms: axioms_out,
        numeric_init: num_init_values,
        mutex_groups: mutex_key.to_vec(),
        ranges: ranges.clone(),
        axiom_layers: vec![-1; ranges.len()],
        init: init_values,
        goal: goal_pairs,
        translation_key: translation_key.clone(),
        canonical_variables: Vec::new(),
        canonical_operators: Vec::new(),
        canonical_metric: None,
        metric: metric,
        global_constraint: global_constraint_pair,
        comp_axiom_layer: -1,
    }
}
