use itertools::Itertools;
/// Invariant parts and invariant checking for mutex group computation.
use std::collections::{HashMap, HashSet};

use super::constraints::{Assignment, ConstraintSystem, NegativeClause, class_of};
use super::pddl::actions::Action;
use super::pddl::conditions::*;
use super::pddl::effects::Effect;
use super::tools;
use super::tools::OrderedSet;

fn invert_list(alist: &[String]) -> HashMap<String, Vec<usize>> {
    let mut result: HashMap<String, Vec<usize>> = HashMap::new();
    for (pos, arg) in alist.iter().enumerate() {
        result.entry(arg.clone()).or_default().push(pos);
    }
    result
}

/// Every way of picking, for each pair, one bijection from its preimage to a
/// permutation of its image, concatenated into a single mapping.
fn instantiate_factored_mapping(pairs: &[(Vec<usize>, Vec<i32>)]) -> Vec<Vec<(usize, i32)>> {
    let part_mappings: Vec<Vec<Vec<(usize, i32)>>> = pairs
        .iter()
        .map(|(preimg, img)| {
            img.iter()
                .copied()
                .permutations(img.len())
                .map(|permuted| preimg.iter().copied().zip(permuted).collect())
                .collect()
        })
        .collect();
    tools::cartesian_product(&part_mappings)
}

/// One name per invariant parameter, none of them clashing with a variable the
/// action already uses.
fn find_unique_variables(action: &Action, invariant: &Invariant) -> Vec<String> {
    let taken: HashSet<&str> = action
        .parameters
        .iter()
        .chain(action.effects.iter().flat_map(|eff| eff.parameters.iter()))
        .map(|param| param.name.as_str())
        .collect();
    (0..)
        .map(|index| format!("?v{index}"))
        .filter(|name| !taken.contains(name.as_str()))
        .take(invariant.arity())
        .collect()
}

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

/// Whether the condition is negated, its predicate and its arguments, or `None`
/// if it is not a literal. Effects of numeric actions need not be literals, so
/// the invariant analysis filters rather than asserts on them.
fn literal_info(cond: &Condition) -> Option<(bool, &str, &[String])> {
    match cond {
        Condition::Atom(a) => Some((false, &a.predicate, &a.args)),
        Condition::NegatedAtom(na) => Some((true, &na.predicate, &na.args)),
        _ => None,
    }
}

/// The arguments of a condition already known to be a literal.
fn literal_args(cond: &Condition) -> &[String] {
    match cond {
        Condition::Atom(a) => &a.args,
        Condition::NegatedAtom(na) => &na.args,
        other => panic!("expected a literal, got {other}"),
    }
}

/// Constrains `system` so that it is only solvable if the conjunction of all
/// parts is satisfiable: `(= x y)` and `(not (= x y))` become an equality and an
/// inequality, and a predicate occurring both positively and negatively demands
/// that the two occurrences differ in some argument.
pub fn ensure_conjunction_sat(system: &mut ConstraintSystem, parts: &[&[&Condition]]) {
    let mut pos: HashMap<&str, Vec<&[String]>> = HashMap::new();
    let mut neg: HashMap<&str, Vec<&[String]>> = HashMap::new();

    for literal in parts.iter().copied().flatten() {
        let Some((negated, predicate, args)) = literal_info(literal) else {
            continue;
        };
        if predicate == "=" {
            assert_eq!(args.len(), 2, "an (in)equality relates two terms");
            let pair = vec![(args[0].clone(), args[1].clone())];
            if negated {
                system.add_negative_clause(NegativeClause::new(pair));
            } else {
                system.add_assignment(Assignment::new(pair));
            }
        } else if negated {
            neg.entry(predicate).or_default().push(args);
        } else {
            pos.entry(predicate).or_default().push(args);
        }
    }

    for (predicate, pos_occurrences) in &pos {
        for neg_args in neg.get(predicate).into_iter().flatten() {
            for pos_args in pos_occurrences {
                let differ: Vec<(String, String)> = neg_args
                    .iter()
                    .cloned()
                    .zip(pos_args.iter().cloned())
                    .collect();
                if !differ.is_empty() {
                    system.add_negative_clause(NegativeClause::new(differ));
                }
            }
        }
    }
}

fn ensure_cover(
    system: &mut ConstraintSystem,
    literal: &Condition,
    invariant: &Invariant,
    inv_vars: &[String],
) {
    system.add_assignment(invariant.cover_equivalence_conjunction(inv_vars, literal));
}

fn ensure_inequality(system: &mut ConstraintSystem, literal1: &Condition, literal2: &Condition) {
    if let (Some((_, pred1, args1)), Some((_, pred2, args2))) =
        (literal_info(literal1), literal_info(literal2))
        && pred1 == pred2
        && !args1.is_empty()
    {
        let parts: Vec<(String, String)> = args1
            .iter()
            .zip(args2.iter())
            .map(|(a, b)| (a.clone(), b.clone()))
            .collect();
        system.add_negative_clause(NegativeClause::new(parts));
    }
}

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

    pub fn arity(&self) -> usize {
        self.order.len()
    }

    pub fn get_assignment(&self, parameters: &[String], literal: &Condition) -> Assignment {
        let args = literal_args(literal);
        Assignment::new(
            parameters
                .iter()
                .zip(self.order.iter())
                .map(|(param, &argpos)| (param.clone(), args[argpos].clone()))
                .collect(),
        )
    }

    pub fn get_parameters(&self, literal: &Condition) -> Vec<String> {
        let args = literal_args(literal);
        self.order.iter().map(|&pos| args[pos].clone()).collect()
    }

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

    pub fn possible_mappings(
        &self,
        own_literal: &Condition,
        other_literal: &Condition,
    ) -> Vec<Vec<(usize, i32)>> {
        let other_args = literal_args(other_literal);

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

    pub fn possible_matches(
        &self,
        own_literal: &Condition,
        other_literal: &Condition,
    ) -> Vec<InvariantPart> {
        let (_, other_pred, _) =
            literal_info(other_literal).expect("matches are only sought between literals");

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

    pub fn arity(&self) -> usize {
        self.parts.iter().next().unwrap().arity()
    }

    pub fn get_parameters(&self, atom: &Condition) -> Vec<String> {
        self.part_for(atom).get_parameters(atom)
    }

    /// The part covering `atom`'s predicate. There is at most one, which
    /// `Invariant::new` asserts by comparing the number of parts with the
    /// number of distinct predicates; several parts would make covering a
    /// disjunction rather than a single equality conjunction.
    fn part_for(&self, atom: &Condition) -> &InvariantPart {
        let (_, predicate, _) =
            literal_info(atom).expect("invariant parts are only matched against literals");
        self.predicate_to_part
            .get(predicate)
            .expect("the invariant has a part for the atom's predicate")
    }

    /// The equality conjunction that makes the invariant cover `atom`: every
    /// invariant parameter is equated with the corresponding argument of the
    /// atom. Whether the literal is negated does not matter.
    pub fn cover_equivalence_conjunction(
        &self,
        parameters: &[String],
        atom: &Condition,
    ) -> Assignment {
        self.part_for(atom).get_assignment(parameters, atom)
    }

    pub fn check_balance(
        &self,
        balance_checker: &BalanceChecker,
        enqueue_func: &mut dyn FnMut(Invariant),
    ) -> bool {
        // The actions are collected in a fixed order, and each is checked at
        // most once. Which action's check fails first decides which refined
        // candidates are enqueued, so drawing them from a hash set would make
        // the invariants found -- and with them the SAS variables -- depend on
        // the run. Mainline Fast Downward avoids a set here for the same reason
        // (issue879), and then draws from the collected actions at random to
        // fail early on average; we keep the collection order instead, which is
        // reproducible without a pseudo-random generator to agree on.
        let mut parts: Vec<&InvariantPart> = self.parts.iter().collect();
        parts.sort();
        let mut actions_to_check: OrderedSet<usize> = OrderedSet::default();
        for part in parts {
            for &action_idx in balance_checker.get_threats(&part.predicate) {
                actions_to_check.insert(action_idx);
            }
        }
        for action_idx in actions_to_check.into_vec() {
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

    pub fn operator_too_heavy(&self, h_action: &Action) -> bool {
        let add_effects: Vec<&Effect> = h_action
            .effects
            .iter()
            .filter(|eff| self.covers(&eff.peffect) == Some(false))
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

            let precond_literals = get_literals(&h_action.precondition);
            let eff1_cond_literals = get_literals(&eff1.condition);
            let eff2_cond_literals = get_literals(&eff2.condition);
            let eff1_neg = negate_literal(&eff1.peffect);
            let eff2_neg = negate_literal(&eff2.peffect);
            let eff1_neg_slice = [&eff1_neg];
            let eff2_neg_slice = [&eff2_neg];

            let parts: Vec<&[&Condition]> = vec![
                &precond_literals[..],
                &eff1_cond_literals[..],
                &eff2_cond_literals[..],
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

    pub fn operator_unbalanced(
        &self,
        action: &Action,
        enqueue_func: &mut dyn FnMut(Invariant),
    ) -> bool {
        let inv_vars = find_unique_variables(action, self);
        let (add_effects, del_effects): (Vec<&Effect>, Vec<&Effect>) = action
            .effects
            .iter()
            .filter(|eff| self.covers(&eff.peffect).is_some())
            .partition(|eff| self.covers(&eff.peffect) == Some(false));

        add_effects.iter().any(|add_effect| {
            self.add_effect_unbalanced(action, add_effect, &del_effects, &inv_vars, enqueue_func)
        })
    }

    /// Whether the effect is negated, or `None` if the invariant has no part for
    /// it -- either because it is not a literal, which a numeric effect need not
    /// be, or because its predicate does not occur in the invariant.
    fn covers(&self, peffect: &Condition) -> Option<bool> {
        let (negated, predicate, _) = literal_info(peffect)?;
        self.predicate_to_part
            .contains_key(predicate)
            .then_some(negated)
    }

    /// Whether `add_effect` can threaten the invariant in an application of
    /// `action` that no delete effect of the same action balances. If so, the
    /// refined candidates are enqueued.
    fn add_effect_unbalanced(
        &self,
        action: &Action,
        add_effect: &Effect,
        del_effects: &[&Effect],
        inv_vars: &[String],
        enqueue_func: &mut dyn FnMut(Invariant),
    ) -> bool {
        // What must hold for the action to be applicable and to actually
        // produce the add effect, indexed by predicate. The atom must not
        // already be true, or the effect produces nothing.
        let mut produced_by_pred: HashMap<&str, Vec<&Condition>> = HashMap::new();
        let precond_literals = get_literals(&action.precondition);
        let add_cond_literals = get_literals(&add_effect.condition);
        let add_neg = negate_literal(&add_effect.peffect);
        for literal in precond_literals
            .iter()
            .chain(add_cond_literals.iter())
            .chain(std::iter::once(&&add_neg))
        {
            let (_, predicate, _) = literal_info(literal).expect("get_literals yields literals");
            produced_by_pred.entry(predicate).or_default().push(literal);
        }

        // Equating every invariant parameter with its value in the add effect
        // is exactly the case in which the add effect threatens the invariant.
        let mut add_cover = self.cover_equivalence_conjunction(inv_vars, &add_effect.peffect);

        // The add effect has to be balanced in *every* threatening application,
        // so a solution may not restrict the action parameters or the effect's
        // quantified variables beyond what the threat itself forces: it may
        // neither equate two of them nor bind one to an object.
        let params: Vec<&str> = action
            .parameters
            .iter()
            .chain(add_effect.parameters.iter())
            .map(|param| param.name.as_str())
            .collect();
        let mut param_system = ConstraintSystem::new();
        let representative = add_cover
            .representative()
            .expect("a cover equates each invariant parameter with one effect argument");
        for &param in &params {
            if class_of(representative, param).starts_with('?') {
                param_system.add_not_constant(param.to_string());
            }
        }
        for (&n1, &n2) in params.iter().tuple_combinations() {
            if class_of(representative, n1) != class_of(representative, n2) {
                param_system.add_negative_clause(NegativeClause::new(vec![(
                    n1.to_string(),
                    n2.to_string(),
                )]));
            }
        }

        for del_effect in del_effects {
            if self.balances(
                del_effect,
                add_effect,
                &produced_by_pred,
                &add_cover,
                &param_system,
                inv_vars,
            ) {
                return false;
            }
        }

        self.refine_candidate(add_effect, action, enqueue_func);
        true
    }

    /// Whether `del_effect` is guaranteed to consume the atom that `add_effect`
    /// produces, in every application in which the add effect threatens the
    /// invariant.
    ///
    /// `add_cover` fixes the invariant parameters to the add effect's
    /// arguments, and `param_system` keeps the action parameters and the add
    /// effect's quantified variables unrestricted; both only depend on the add
    /// effect and are therefore computed once by the caller.
    fn balances(
        &self,
        del_effect: &Effect,
        add_effect: &Effect,
        produced_by_pred: &HashMap<&str, Vec<&Condition>>,
        add_cover: &Assignment,
        param_system: &ConstraintSystem,
        inv_vars: &[String],
    ) -> bool {
        let Some(balance_system) = self.balance_system(add_effect, del_effect, produced_by_pred)
        else {
            // No production by the add effect can imply a consumption.
            return false;
        };

        let mut system = ConstraintSystem::new();
        system.add_assignment(add_cover.clone());
        ensure_cover(&mut system, &del_effect.peffect, self, inv_vars);
        system.extend(&balance_system);
        system.extend(param_system);
        system.is_solvable()
    }

    /// A system that is solvable if the conjunction in `produced_by_pred`
    /// implies the consumption of the delete effect's atom, and the produced
    /// and consumed atoms differ -- under add-after-delete semantics, deleting
    /// the very atom the action adds balances nothing.
    ///
    /// `None` if no instantiation can make the implication hold.
    fn balance_system(
        &self,
        add_effect: &Effect,
        del_effect: &Effect,
        produced_by_pred: &HashMap<&str, Vec<&Condition>>,
    ) -> Option<ConstraintSystem> {
        let mut system = ConstraintSystem::new();
        let del_cond_literals = get_literals(&del_effect.condition);
        let del_neg = negate_literal(&del_effect.peffect);
        for literal in del_cond_literals.iter().chain(std::iter::once(&&del_neg)) {
            let (negated, predicate, args) =
                literal_info(literal).expect("get_literals yields literals");
            // The ways in which one of the literals that hold on production
            // implies this literal: they must agree on every argument.
            let possibilities: Vec<Assignment> = produced_by_pred
                .get(predicate)
                .map_or(&[][..], Vec::as_slice)
                .iter()
                .filter_map(|candidate| {
                    let (candidate_negated, _, candidate_args) =
                        literal_info(candidate).expect("get_literals yields literals");
                    (candidate_negated == negated).then(|| {
                        Assignment::new(
                            args.iter()
                                .cloned()
                                .zip(candidate_args.iter().cloned())
                                .collect(),
                        )
                    })
                })
                .collect();
            if possibilities.is_empty() {
                return None;
            }
            system.add_assignment_disjunction(possibilities);
        }

        ensure_inequality(&mut system, &add_effect.peffect, &del_effect.peffect);
        Some(system)
    }

    fn refine_candidate(
        &self,
        add_effect: &Effect,
        action: &Action,
        enqueue_func: &mut dyn FnMut(Invariant),
    ) {
        let part = self.part_for(&add_effect.peffect);
        for del_eff in &action.effects {
            if let Some((negated, del_pred, _)) = literal_info(&del_eff.peffect)
                && negated
                && !self.predicate_to_part.contains_key(del_pred)
            {
                for match_part in part.possible_matches(&add_effect.peffect, &del_eff.peffect) {
                    let mut new_parts: HashSet<InvariantPart> = self.parts.clone();
                    new_parts.insert(match_part);
                    enqueue_func(Invariant::new(new_parts));
                }
            }
        }
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

/// Placed here to be accessible from Invariant methods.
pub struct BalanceChecker {
    /// The actions that add each predicate, by action index, in increasing
    /// order. A set would do as far as the contents go, but the order in which
    /// the balance check considers the actions has to be reproducible.
    pub predicates_to_add_action_indices: HashMap<String, Vec<usize>>,
    pub action_to_heavy_action: HashMap<usize, Action>,
    pub actions: Vec<Action>,
}

impl BalanceChecker {
    pub fn get_threats(&self, predicate: &str) -> &[usize] {
        self.predicates_to_add_action_indices
            .get(predicate)
            .map_or(&[], Vec::as_slice)
    }

    /// The variant of `actions[action_idx]` with every parameterized effect
    /// duplicated.
    ///
    /// `action_to_heavy_action` only holds the actions that needed such a
    /// variant; an action whose effects are all parameter-free is its own heavy
    /// action, which is why the miss returns the action itself rather than a
    /// substitute. Both entries are built from the same inequality-augmented
    /// action, and an out-of-range index still panics on the slice.
    pub fn get_heavy_action(&self, action_idx: usize) -> &Action {
        self.action_to_heavy_action
            .get(&action_idx)
            .unwrap_or(&self.actions[action_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pddl::pddl_types::TypedObject;

    fn atom(predicate: &str, args: &[&str]) -> Condition {
        Condition::Atom(Atom::new(
            predicate.to_string(),
            args.iter().map(|arg| arg.to_string()).collect(),
        ))
    }

    /// The invariant "no object satisfies both P and Q".
    fn no_object_is_p_and_q() -> Invariant {
        Invariant::new([
            InvariantPart::new("P".to_string(), vec![0], -1),
            InvariantPart::new("Q".to_string(), vec![0], -1),
        ])
    }

    fn action(parameters: &[&str], precondition: Condition, effects: Vec<Effect>) -> Action {
        Action::new(
            "a".to_string(),
            parameters
                .iter()
                .map(|name| TypedObject::new(name, "object"))
                .collect(),
            parameters.len(),
            precondition,
            effects,
            None,
        )
    }

    fn is_unbalanced(action: &Action) -> bool {
        no_object_is_p_and_q().operator_unbalanced(action, &mut |_: Invariant| {})
    }

    /// `a(?p)` adds `P(?p)` but deletes `Q(?p)` only where `R(?p)` holds, while
    /// the precondition merely forces `R` on the object `c`. Every other binding
    /// of `?p` leaves both `P(?p)` and `Q(?p)` true. The delete effect balances
    /// the add effect only for `?p = c`, and a balance that needs a specific
    /// object is no balance at all.
    #[test]
    fn a_delete_condition_may_not_bind_an_action_parameter_to_an_object() {
        let action = action(
            &["?p"],
            Condition::Conjunction(Conjunction::new(vec![
                atom("R", &["c"]),
                atom("Q", &["?p"]),
            ])),
            vec![
                Effect::new(vec![], Condition::Truth, atom("P", &["?p"])),
                Effect::new(
                    vec![],
                    atom("R", &["?p"]),
                    negate_literal(&atom("Q", &["?p"])),
                ),
            ],
        );
        assert!(is_unbalanced(&action));
    }

    /// `a(?p)` deletes `Q(?p)` but adds `P(?y)` for every `?y` satisfying the
    /// effect condition, so any `?y` other than `?p` breaks the invariant. The
    /// quantified variable of an add effect is as free as an action parameter,
    /// so a balance may not equate it with one either.
    #[test]
    fn a_quantified_add_effect_variable_may_not_be_equated_with_an_action_parameter() {
        let action = action(
            &["?p"],
            atom("Q", &["?p"]),
            vec![
                Effect::new(
                    vec![TypedObject::new("?y", "object")],
                    Condition::Conjunction(Conjunction::new(vec![
                        atom("S", &["?y"]),
                        atom("Q", &["?y"]),
                    ])),
                    atom("P", &["?y"]),
                ),
                Effect::new(
                    vec![],
                    Condition::Truth,
                    negate_literal(&atom("Q", &["?p"])),
                ),
            ],
        );
        assert!(is_unbalanced(&action));
    }
}
