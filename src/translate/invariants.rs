use itertools::Itertools;
/// Port of invariants.py
/// Invariant parts and invariant checking for mutex group computation.
use std::collections::{HashMap, HashSet};

use super::constraints::{Assignment, ConstraintSystem, NegativeClause};
use super::pddl::actions::Action;
use super::pddl::conditions::*;
use super::tools;

/// Python: def invert_list(alist)
fn invert_list(alist: &[String]) -> HashMap<String, Vec<usize>> {
    let mut result: HashMap<String, Vec<usize>> = HashMap::new();
    for (pos, arg) in alist.iter().enumerate() {
        result.entry(arg.clone()).or_default().push(pos);
    }
    result
}

/// Python: def instantiate_factored_mapping(pairs)
fn instantiate_factored_mapping(pairs: &[(Vec<usize>, Vec<i32>)]) -> Vec<Vec<(usize, i32)>> {
    let part_mappings: Vec<Vec<Vec<(usize, i32)>>> = pairs
        .iter()
        .map(|(preimg, img)| {
            // Generate all permutations of img and zip with preimg
            let perms: Vec<Vec<i32>> = img.iter().cloned().permutations(img.len()).collect();
            perms
                .into_iter()
                .map(|perm_img| {
                    preimg
                        .iter()
                        .cloned()
                        .zip(perm_img.into_iter())
                        .collect::<Vec<(usize, i32)>>()
                })
                .collect()
        })
        .collect();

    // Cartesian product: concatenate lists
    let as_vec_vecs: Vec<Vec<Vec<(usize, i32)>>> = part_mappings;
    tools::cartesian_product(&as_vec_vecs)
}

/// Python: def find_unique_variables(action, invariant)
fn find_unique_variables(action: &Action, invariant: &Invariant) -> Vec<String> {
    let mut params: HashSet<String> = action.parameters.iter().map(|p| p.name.clone()).collect();
    for eff in &action.effects {
        for p in &eff.parameters {
            params.insert(p.name.clone());
        }
    }
    let mut inv_vars = vec![];
    let mut counter = 0;
    for _ in 0..invariant.arity() {
        loop {
            let new_name = format!("?v{}", counter);
            counter += 1;
            if !params.contains(&new_name) {
                inv_vars.push(new_name);
                break;
            }
        }
    }
    inv_vars
}

/// Python: def get_literals(condition)
fn get_literals(condition: &Condition) -> Vec<&Condition> {
    match condition {
        Condition::Atom(_) | Condition::NegatedAtom(_) => vec![condition],
        Condition::Conjunction(conj) => conj
            .parts
            .iter()
            .filter(|p| matches!(p, Condition::Atom(_) | Condition::NegatedAtom(_)))
            .collect(),
        _ => vec![],
    }
}

/// Helper: get predicate and args from a literal condition
fn literal_info(cond: &Condition) -> Option<(bool, &str, &[String])> {
    match cond {
        Condition::Atom(a) => Some((false, &a.predicate, &a.args)),
        Condition::NegatedAtom(na) => Some((true, &na.predicate, &na.args)),
        _ => None,
    }
}

/// Python: def ensure_conjunction_sat(system, *parts)
pub fn ensure_conjunction_sat(system: &mut ConstraintSystem, parts: &[&[&Condition]]) {
    let mut pos: HashMap<String, Vec<&Condition>> = HashMap::new();
    let mut neg: HashMap<String, Vec<&Condition>> = HashMap::new();

    for part in parts {
        for literal in *part {
            if let Some((negated, predicate, args)) = literal_info(literal) {
                if predicate == "=" {
                    if args.len() == 2 {
                        if negated {
                            let n = NegativeClause::new(vec![(args[0].clone(), args[1].clone())]);
                            system.add_negative_clause(n);
                        } else {
                            let a = Assignment::new(vec![(args[0].clone(), args[1].clone())]);
                            system.add_assignment_disjunction(vec![a]);
                        }
                    }
                } else if negated {
                    neg.entry(predicate.to_string()).or_default().push(literal);
                } else {
                    pos.entry(predicate.to_string()).or_default().push(literal);
                }
            }
        }
    }

    for (pred, posatoms) in &pos {
        if let Some(negatoms) = neg.get(pred) {
            for posatom in posatoms {
                for negatom in negatoms {
                    if let (Some((_, _, pos_args)), Some((_, _, neg_args))) =
                        (literal_info(posatom), literal_info(negatom))
                    {
                        let parts: Vec<(String, String)> = neg_args
                            .iter()
                            .zip(pos_args.iter())
                            .map(|(a, b)| (a.clone(), b.clone()))
                            .collect();
                        if !parts.is_empty() {
                            system.add_negative_clause(NegativeClause::new(parts));
                        }
                    }
                }
            }
        }
    }
}

/// Python: def ensure_cover(system, literal, invariant, inv_vars)
fn ensure_cover(
    system: &mut ConstraintSystem,
    literal: &Condition,
    invariant: &Invariant,
    inv_vars: &[String],
) {
    let a = invariant.get_covering_assignments(inv_vars, literal);
    assert_eq!(a.len(), 1);
    system.add_assignment_disjunction(a);
}

/// Python: def ensure_inequality(system, literal1, literal2)
fn ensure_inequality(system: &mut ConstraintSystem, literal1: &Condition, literal2: &Condition) {
    if let (Some((_, pred1, args1)), Some((_, pred2, args2))) =
        (literal_info(literal1), literal_info(literal2))
    {
        if pred1 == pred2 && !args1.is_empty() {
            let parts: Vec<(String, String)> = args1
                .iter()
                .zip(args2.iter())
                .map(|(a, b)| (a.clone(), b.clone()))
                .collect();
            system.add_negative_clause(NegativeClause::new(parts));
        }
    }
}

/// Python: class InvariantPart(object)
#[derive(Debug, Clone, Eq)]
pub struct InvariantPart {
    pub predicate: String,
    pub order: Vec<usize>, // mapping from invariant var positions to predicate arg positions
    pub omitted_pos: i32,  // position of the "counted" variable, -1 if none
}

impl InvariantPart {
    pub fn new(predicate: String, order: Vec<usize>, omitted_pos: i32) -> Self {
        InvariantPart {
            predicate,
            order,
            omitted_pos,
        }
    }

    /// Python: def arity(self)
    pub fn arity(&self) -> usize {
        self.order.len()
    }

    /// Python: def get_assignment(self, parameters, literal)
    pub fn get_assignment(&self, parameters: &[String], literal: &Condition) -> Assignment {
        if let Some((_, _, args)) = literal_info(literal) {
            let equalities: Vec<(String, String)> = parameters
                .iter()
                .zip(self.order.iter())
                .map(|(param, &argpos)| (param.clone(), args[argpos].clone()))
                .collect();
            Assignment::new(equalities)
        } else {
            Assignment::new(vec![])
        }
    }

    /// Python: def get_parameters(self, literal)
    pub fn get_parameters(&self, literal: &Condition) -> Vec<String> {
        if let Some((_, _, args)) = literal_info(literal) {
            self.order.iter().map(|&pos| args[pos].clone()).collect()
        } else {
            vec![]
        }
    }

    /// Python: def instantiate(self, parameters)
    pub fn instantiate(&self, parameters: &[String]) -> Atom {
        let num_args = self.order.len() + if self.omitted_pos != -1 { 1 } else { 0 };
        let mut args = vec!["?X".to_string(); num_args];
        for (param, &argpos) in parameters.iter().zip(self.order.iter()) {
            args[argpos] = param.clone();
        }
        Atom {
            predicate: self.predicate.clone(),
            args,
        }
    }

    /// Python: def possible_mappings(self, own_literal, other_literal)
    pub fn possible_mappings(
        &self,
        own_literal: &Condition,
        other_literal: &Condition,
    ) -> Vec<Vec<(usize, i32)>> {
        let (_, _, other_args) = match literal_info(other_literal) {
            Some(info) => info,
            None => return vec![],
        };

        let allowed_omissions_init = other_args.len() as i32 - self.order.len() as i32;
        if allowed_omissions_init != 0 && allowed_omissions_init != 1 {
            return vec![];
        }
        let mut allowed_omissions = allowed_omissions_init;

        let own_parameters = self.get_parameters(own_literal);
        let arg_to_ordered_pos = invert_list(&own_parameters);
        let other_args_vec: Vec<String> = other_args.to_vec();
        let other_arg_to_pos = invert_list(&other_args_vec);

        let mut factored_mapping: Vec<(Vec<usize>, Vec<i32>)> = vec![];

        for (key, other_positions) in &other_arg_to_pos {
            let own_positions = arg_to_ordered_pos.get(key).cloned().unwrap_or_default();
            let len_diff = own_positions.len() as i32 - other_positions.len() as i32;
            if len_diff >= 1 || len_diff <= -2 || (len_diff == -1 && allowed_omissions == 0) {
                return vec![];
            }
            if len_diff != 0 {
                let mut own_pos_extended = own_positions.clone();
                own_pos_extended.push(usize::MAX); // sentinel for -1
                let own_pos_i32: Vec<i32> = own_pos_extended
                    .iter()
                    .map(|&p| if p == usize::MAX { -1 } else { p as i32 })
                    .collect();
                allowed_omissions = 0;
                factored_mapping.push((other_positions.clone(), own_pos_i32));
            } else {
                let own_pos_i32: Vec<i32> = own_positions.iter().map(|&p| p as i32).collect();
                factored_mapping.push((other_positions.clone(), own_pos_i32));
            }
        }

        instantiate_factored_mapping(&factored_mapping)
    }

    /// Python: def possible_matches(self, own_literal, other_literal)
    pub fn possible_matches(
        &self,
        own_literal: &Condition,
        other_literal: &Condition,
    ) -> Vec<InvariantPart> {
        let (_, other_pred, _) = match literal_info(other_literal) {
            Some(info) => info,
            None => return vec![],
        };

        let mut result = vec![];
        for mapping in self.possible_mappings(own_literal, other_literal) {
            let mut new_order = vec![0usize; self.order.len()];
            let mut omitted: i32 = -1;
            for (key, value) in &mapping {
                if *value == -1 {
                    omitted = *key as i32;
                } else {
                    new_order[*value as usize] = *key;
                }
            }
            result.push(InvariantPart::new(
                other_pred.to_string(),
                new_order,
                omitted,
            ));
        }
        result
    }

    /// Python: def matches(self, other, own_literal, other_literal)
    pub fn matches(
        &self,
        other: &InvariantPart,
        own_literal: &Condition,
        other_literal: &Condition,
    ) -> bool {
        self.get_parameters(own_literal) == other.get_parameters(other_literal)
    }
}

impl PartialEq for InvariantPart {
    fn eq(&self, other: &Self) -> bool {
        self.predicate == other.predicate && self.order == other.order
    }
}

impl std::hash::Hash for InvariantPart {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.predicate.hash(state);
        self.order.hash(state);
    }
}

impl PartialOrd for InvariantPart {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InvariantPart {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.predicate
            .cmp(&other.predicate)
            .then(self.order.cmp(&other.order))
    }
}

/// Python: class Invariant(object)
#[derive(Debug, Clone)]
pub struct Invariant {
    pub parts: HashSet<InvariantPart>,
    pub predicates: HashSet<String>,
    pub predicate_to_part: HashMap<String, InvariantPart>,
}

impl Invariant {
    pub fn new(parts: impl IntoIterator<Item = InvariantPart>) -> Self {
        let parts_set: HashSet<InvariantPart> = parts.into_iter().collect();
        let predicates: HashSet<String> = parts_set.iter().map(|p| p.predicate.clone()).collect();
        let predicate_to_part: HashMap<String, InvariantPart> = parts_set
            .iter()
            .map(|p| (p.predicate.clone(), p.clone()))
            .collect();
        assert_eq!(parts_set.len(), predicates.len());
        Invariant {
            parts: parts_set,
            predicates,
            predicate_to_part,
        }
    }

    /// Python: def arity(self)
    pub fn arity(&self) -> usize {
        self.parts.iter().next().unwrap().arity()
    }

    /// Python: def get_parameters(self, atom)
    pub fn get_parameters(&self, atom: &Condition) -> Vec<String> {
        if let Some((_, pred, _)) = literal_info(atom) {
            if let Some(part) = self.predicate_to_part.get(pred) {
                return part.get_parameters(atom);
            }
        }
        vec![]
    }

    /// Python: def instantiate(self, parameters)
    pub fn instantiate(&self, parameters: &[String]) -> Vec<Atom> {
        self.parts
            .iter()
            .map(|part| part.instantiate(parameters))
            .collect()
    }

    /// Python: def get_covering_assignments(self, parameters, atom)
    pub fn get_covering_assignments(
        &self,
        parameters: &[String],
        atom: &Condition,
    ) -> Vec<Assignment> {
        if let Some((_, pred, _)) = literal_info(atom) {
            if let Some(part) = self.predicate_to_part.get(pred) {
                return vec![part.get_assignment(parameters, atom)];
            }
        }
        vec![]
    }

    /// Python: def check_balance(self, balance_checker, enqueue_func)
    pub fn check_balance(
        &self,
        balance_checker: &BalanceChecker,
        enqueue_func: &mut dyn FnMut(Invariant),
    ) -> bool {
        let mut actions_to_check: HashSet<usize> = HashSet::new();
        for part in &self.parts {
            if let Some(indices) = balance_checker.get_threats(&part.predicate) {
                actions_to_check.extend(indices);
            }
        }
        for &action_idx in &actions_to_check {
            let heavy_action = balance_checker.get_heavy_action(action_idx);
            if self.operator_too_heavy(heavy_action) {
                return false;
            }
            let action = &balance_checker.actions[action_idx];
            if self.operator_unbalanced(action, enqueue_func) {
                return false;
            }
        }
        true
    }

    /// Python: def operator_too_heavy(self, h_action)
    pub fn operator_too_heavy(&self, h_action: &Action) -> bool {
        let add_effects: Vec<&super::pddl::effects::Effect> = h_action
            .effects
            .iter()
            .filter(|eff| {
                if let Some((negated, pred, _)) = literal_info(&eff.peffect) {
                    !negated && self.predicate_to_part.contains_key(pred)
                } else {
                    false
                }
            })
            .collect();

        let inv_vars = find_unique_variables(h_action, self);

        if add_effects.len() <= 1 {
            return false;
        }

        for combo in add_effects.iter().combinations(2) {
            let eff1 = combo[0];
            let eff2 = combo[1];
            let mut system = ConstraintSystem::new();
            ensure_inequality(&mut system, &eff1.peffect, &eff2.peffect);
            ensure_cover(&mut system, &eff1.peffect, self, &inv_vars);
            ensure_cover(&mut system, &eff2.peffect, self, &inv_vars);

            let precond_lits = get_literals(&h_action.precondition);
            let eff1_cond_lits = get_literals(&eff1.condition);
            let eff2_cond_lits = get_literals(&eff2.condition);
            let eff1_neg = negate_literal(&eff1.peffect);
            let eff2_neg = negate_literal(&eff2.peffect);
            let eff1_neg_slice = [&eff1_neg];
            let eff2_neg_slice = [&eff2_neg];

            let parts: Vec<&[&Condition]> = vec![
                &precond_lits[..],
                &eff1_cond_lits[..],
                &eff2_cond_lits[..],
                &eff1_neg_slice[..],
                &eff2_neg_slice[..],
            ];
            ensure_conjunction_sat(&mut system, &parts);

            if system.is_solvable() {
                return true;
            }
        }
        false
    }

    /// Python: def operator_unbalanced(self, action, enqueue_func)
    pub fn operator_unbalanced(
        &self,
        action: &Action,
        enqueue_func: &mut dyn FnMut(Invariant),
    ) -> bool {
        let inv_vars = find_unique_variables(action, self);
        let relevant_effs: Vec<&super::pddl::effects::Effect> = action
            .effects
            .iter()
            .filter(|eff| {
                if let Some((_, pred, _)) = literal_info(&eff.peffect) {
                    self.predicate_to_part.contains_key(pred)
                } else {
                    false
                }
            })
            .collect();

        let add_effects: Vec<&&super::pddl::effects::Effect> = relevant_effs
            .iter()
            .filter(|eff| {
                if let Some((negated, _, _)) = literal_info(&eff.peffect) {
                    !negated
                } else {
                    false
                }
            })
            .collect();

        let del_effects: Vec<&&super::pddl::effects::Effect> = relevant_effs
            .iter()
            .filter(|eff| {
                if let Some((negated, _, _)) = literal_info(&eff.peffect) {
                    negated
                } else {
                    false
                }
            })
            .collect();

        for eff in &add_effects {
            if self.add_effect_unbalanced(action, eff, &del_effects, &inv_vars, enqueue_func) {
                return true;
            }
        }
        false
    }

    /// Python: def minimal_covering_renamings(self, action, add_effect, inv_vars)
    fn minimal_covering_renamings(
        &self,
        action: &Action,
        add_effect: &super::pddl::effects::Effect,
        inv_vars: &[String],
    ) -> Vec<ConstraintSystem> {
        let assigs = self.get_covering_assignments(inv_vars, &add_effect.peffect);

        let params: Vec<String> = action.parameters.iter().map(|p| p.name.clone()).collect();
        let mut minimal_renamings = vec![];

        for assignment in &assigs {
            let mut system = ConstraintSystem::new();
            system.add_assignment(assignment.clone());
            let mapping = assignment.get_mapping();
            if params.len() > 1 {
                for combo in params.iter().combinations(2) {
                    let n1 = combo[0];
                    let n2 = combo[1];
                    let mapped_n1 = mapping.get(n1).unwrap_or(n1);
                    let mapped_n2 = mapping.get(n2).unwrap_or(n2);
                    if mapped_n1 != mapped_n2 {
                        let neg = NegativeClause::new(vec![(n1.clone(), n2.clone())]);
                        system.add_negative_clause(neg);
                    }
                }
            }
            minimal_renamings.push(system);
        }
        minimal_renamings
    }

    /// Python: def add_effect_unbalanced(self, action, add_effect, del_effects, inv_vars, enqueue_func)
    fn add_effect_unbalanced(
        &self,
        action: &Action,
        add_effect: &super::pddl::effects::Effect,
        del_effects: &[&&super::pddl::effects::Effect],
        inv_vars: &[String],
        enqueue_func: &mut dyn FnMut(Invariant),
    ) -> bool {
        let mut minimal_renamings = self.minimal_covering_renamings(action, add_effect, inv_vars);

        let mut lhs_by_pred: HashMap<String, Vec<&Condition>> = HashMap::new();
        let precond_lits = get_literals(&action.precondition);
        let add_cond_lits = get_literals(&add_effect.condition);
        let add_neg = negate_literal(&add_effect.peffect);

        for lit in precond_lits
            .iter()
            .chain(add_cond_lits.iter())
            .chain(std::iter::once(&&add_neg))
        {
            if let Some((_, pred, _)) = literal_info(lit) {
                lhs_by_pred.entry(pred.to_string()).or_default().push(lit);
            }
        }

        for del_effect in del_effects {
            minimal_renamings = self.unbalanced_renamings(
                del_effect,
                add_effect,
                inv_vars,
                &lhs_by_pred,
                minimal_renamings,
            );
            if minimal_renamings.is_empty() {
                return false;
            }
        }

        self.refine_candidate(add_effect, action, enqueue_func);
        true
    }

    /// Python: def refine_candidate(self, add_effect, action, enqueue_func)
    fn refine_candidate(
        &self,
        add_effect: &super::pddl::effects::Effect,
        action: &Action,
        enqueue_func: &mut dyn FnMut(Invariant),
    ) {
        if let Some((_, add_pred, _)) = literal_info(&add_effect.peffect) {
            if let Some(part) = self.predicate_to_part.get(add_pred) {
                for del_eff in &action.effects {
                    if let Some((negated, del_pred, _)) = literal_info(&del_eff.peffect) {
                        if negated && !self.predicate_to_part.contains_key(del_pred) {
                            for match_part in
                                part.possible_matches(&add_effect.peffect, &del_eff.peffect)
                            {
                                let mut new_parts: HashSet<InvariantPart> = self.parts.clone();
                                new_parts.insert(match_part);
                                enqueue_func(Invariant::new(new_parts));
                            }
                        }
                    }
                }
            }
        }
    }

    /// Python: def unbalanced_renamings(self, del_effect, add_effect, inv_vars, lhs_by_pred, unbalanced_renamings)
    fn unbalanced_renamings(
        &self,
        del_effect: &super::pddl::effects::Effect,
        add_effect: &super::pddl::effects::Effect,
        inv_vars: &[String],
        lhs_by_pred: &HashMap<String, Vec<&Condition>>,
        unbalanced_renamings: Vec<ConstraintSystem>,
    ) -> Vec<ConstraintSystem> {
        let mut system = ConstraintSystem::new();
        ensure_inequality(&mut system, &add_effect.peffect, &del_effect.peffect);
        ensure_cover(&mut system, &del_effect.peffect, self, inv_vars);

        // Check constants
        let mut check_constants = false;
        let mut constant_test_system = ConstraintSystem::new();

        if !system.combinatorial_assignments.is_empty()
            && !system.combinatorial_assignments[0].is_empty()
        {
            for (a, b) in &system.combinatorial_assignments[0][0].equalities {
                if !b.starts_with('?') {
                    check_constants = true;
                    let neg = NegativeClause::new(vec![(a.clone(), b.clone())]);
                    constant_test_system.add_negative_clause(neg);
                }
            }
        }

        ensure_inequality(&mut system, &add_effect.peffect, &del_effect.peffect);

        let mut still_unbalanced = vec![];
        for renaming in unbalanced_renamings {
            if check_constants {
                let new_sys = constant_test_system.combine(&renaming);
                if new_sys.is_solvable() {
                    still_unbalanced.push(renaming);
                    continue;
                }
            }
            let mut new_sys = system.combine(&renaming);
            if self.lhs_satisfiable(&renaming, lhs_by_pred) {
                if let Some(implies_system) = self.imply_del_effect(del_effect, lhs_by_pred) {
                    new_sys = new_sys.combine(&implies_system);
                } else {
                    still_unbalanced.push(renaming);
                    continue;
                }
            }
            if !new_sys.is_solvable() {
                still_unbalanced.push(renaming);
            }
        }
        still_unbalanced
    }

    /// Python: def lhs_satisfiable(self, renaming, lhs_by_pred)
    fn lhs_satisfiable(
        &self,
        renaming: &ConstraintSystem,
        lhs_by_pred: &HashMap<String, Vec<&Condition>>,
    ) -> bool {
        let mut system = renaming.copy();
        let all_lits: Vec<&Condition> = lhs_by_pred
            .values()
            .flat_map(|v| v.iter().copied())
            .collect();
        let all_lits_ref: Vec<&[&Condition]> = vec![&all_lits[..]];
        ensure_conjunction_sat(&mut system, &all_lits_ref);
        system.is_solvable()
    }

    /// Python: def imply_del_effect(self, del_effect, lhs_by_pred)
    fn imply_del_effect(
        &self,
        del_effect: &super::pddl::effects::Effect,
        lhs_by_pred: &HashMap<String, Vec<&Condition>>,
    ) -> Option<ConstraintSystem> {
        let mut implies_system = ConstraintSystem::new();

        let del_cond_lits = get_literals(&del_effect.condition);
        let del_neg = negate_literal(&del_effect.peffect);
        let all_lits: Vec<&Condition> = del_cond_lits
            .into_iter()
            .chain(std::iter::once(&del_neg as &Condition))
            .collect();

        for literal in &all_lits {
            if let Some((negated, pred, args)) = literal_info(literal) {
                let matches = lhs_by_pred.get(pred).cloned().unwrap_or_default();
                let mut poss_assignments = vec![];
                for m in &matches {
                    if let Some((m_negated, _, m_args)) = literal_info(m) {
                        if m_negated != negated {
                            continue;
                        }
                        let equalities: Vec<(String, String)> = args
                            .iter()
                            .zip(m_args.iter())
                            .map(|(a, b)| (a.clone(), b.clone()))
                            .collect();
                        poss_assignments.push(Assignment::new(equalities));
                    }
                }
                if poss_assignments.is_empty() {
                    return None;
                }
                implies_system.add_assignment_disjunction(poss_assignments);
            }
        }

        Some(implies_system)
    }
}

impl PartialEq for Invariant {
    fn eq(&self, other: &Self) -> bool {
        self.parts == other.parts
    }
}

impl Eq for Invariant {}

impl std::hash::Hash for Invariant {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash as sorted parts for consistency
        let mut sorted: Vec<&InvariantPart> = self.parts.iter().collect();
        sorted.sort();
        for part in sorted {
            part.hash(state);
        }
    }
}

impl PartialOrd for Invariant {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Invariant {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let mut self_parts: Vec<&InvariantPart> = self.parts.iter().collect();
        let mut other_parts: Vec<&InvariantPart> = other.parts.iter().collect();
        self_parts.sort();
        other_parts.sort();
        self_parts.cmp(&other_parts)
    }
}

/// Helper to negate a literal condition
fn negate_literal(cond: &Condition) -> Condition {
    match cond {
        Condition::Atom(a) => {
            Condition::NegatedAtom(NegatedAtom::new(a.predicate.clone(), a.args.clone()))
        }
        Condition::NegatedAtom(na) => {
            Condition::Atom(Atom::new(na.predicate.clone(), na.args.clone()))
        }
        _ => cond.clone(),
    }
}

/// Python: class BalanceChecker from invariant_finder.py
/// Placed here to be accessible from Invariant methods.
pub struct BalanceChecker {
    pub predicates_to_add_action_indices: HashMap<String, HashSet<usize>>,
    pub action_to_heavy_action: HashMap<usize, Action>,
    pub actions: Vec<Action>,
}

impl BalanceChecker {
    pub fn get_threats(&self, predicate: &str) -> Option<&HashSet<usize>> {
        self.predicates_to_add_action_indices.get(predicate)
    }

    pub fn get_heavy_action(&self, action_idx: usize) -> &Action {
        self.action_to_heavy_action
            .get(&action_idx)
            .unwrap_or(&self.actions[action_idx])
    }
}
