use crate::translate::invariants::{ActionView, EffectView, Invariant, InvariantPart, Literal};
use crate::translate::normalize::{NormalizableTask, TaskAction};
use crate::translate::pddl::Condition;
use crate::translate::pddl_parser::SExpr;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

// Port of python/translate/invariant_finder.py

fn literal_from_effect(effect: &SExpr) -> Option<Literal> {
    match effect {
        SExpr::Atom(a) => Some(Literal {
            predicate: a.clone(),
            args: Vec::new(),
            negated: false,
        }),
        SExpr::List(items) if !items.is_empty() => {
            if let SExpr::Atom(op) = &items[0] {
                let op_l = op.as_str();
                if op_l == "not" && items.len() == 2 {
                    if let SExpr::List(inner) = &items[1] {
                        if inner.is_empty() {
                            return None;
                        }
                        if let SExpr::Atom(pred) = &inner[0] {
                            let args = inner[1..]
                                .iter()
                                .filter_map(|x| match x {
                                    SExpr::Atom(a) => Some(a.clone()),
                                    _ => None,
                                })
                                .collect();
                            return Some(Literal {
                                predicate: pred.clone(),
                                args,
                                negated: true,
                            });
                        }
                    }
                    return None;
                }
                if matches!(
                    op_l,
                    "assign" | "increase" | "decrease" | "scale-up" | "scale-down" | "="
                ) {
                    return None;
                }
                let pred = op.clone();
                let args = items[1..]
                    .iter()
                    .filter_map(|x| match x {
                        SExpr::Atom(a) => Some(a.clone()),
                        _ => None,
                    })
                    .collect();
                return Some(Literal {
                    predicate: pred,
                    args,
                    negated: false,
                });
            }
            None
        }
        _ => None,
    }
}

fn literal_from_sexpr(sexpr: &SExpr) -> Option<Literal> {
    literal_from_effect(sexpr)
}

fn rename_vars_sexpr(sexpr: &SExpr, mapping: &HashMap<String, String>) -> SExpr {
    match sexpr {
        SExpr::Atom(a) => {
            if a.starts_with('?') {
                if let Some(new) = mapping.get(a) {
                    SExpr::Atom(new.clone())
                } else {
                    SExpr::Atom(a.clone())
                }
            } else {
                SExpr::Atom(a.clone())
            }
        }
        SExpr::List(items) => SExpr::List(
            items
                .iter()
                .map(|i| rename_vars_sexpr(i, mapping))
                .collect(),
        ),
    }
}

fn rename_vars_condition(cond: &Condition, mapping: &HashMap<String, String>) -> Condition {
    match cond {
        Condition::Atom(pred, args) => Condition::Atom(
            pred.clone(),
            args.iter()
                .map(|a| mapping.get(a).cloned().unwrap_or_else(|| a.clone()))
                .collect(),
        ),
        Condition::Not(inner) => Condition::Not(Box::new(rename_vars_condition(inner, mapping))),
        Condition::And(parts) => Condition::And(
            parts
                .iter()
                .map(|p| rename_vars_condition(p, mapping))
                .collect(),
        ),
        Condition::Or(parts) => Condition::Or(
            parts
                .iter()
                .map(|p| rename_vars_condition(p, mapping))
                .collect(),
        ),
        Condition::Forall(params, inner) => {
            let new_params: Vec<(String, Option<String>)> = params
                .iter()
                .map(|(n, t)| {
                    (
                        mapping.get(n).cloned().unwrap_or_else(|| n.clone()),
                        t.clone(),
                    )
                })
                .collect();
            Condition::Forall(new_params, Box::new(rename_vars_condition(inner, mapping)))
        }
        Condition::Exists(params, inner) => {
            let new_params: Vec<(String, Option<String>)> = params
                .iter()
                .map(|(n, t)| {
                    (
                        mapping.get(n).cloned().unwrap_or_else(|| n.clone()),
                        t.clone(),
                    )
                })
                .collect();
            Condition::Exists(new_params, Box::new(rename_vars_condition(inner, mapping)))
        }
        Condition::Comparison(op, left, right) => Condition::Comparison(
            op.clone(),
            rename_vars_sexpr(left, mapping),
            rename_vars_sexpr(right, mapping),
        ),
        Condition::True => Condition::True,
    }
}

fn rename_vars_literal(lit: &Literal, mapping: &HashMap<String, String>) -> Literal {
    Literal {
        predicate: lit.predicate.clone(),
        args: lit
            .args
            .iter()
            .map(|a| mapping.get(a).cloned().unwrap_or_else(|| a.clone()))
            .collect(),
        negated: lit.negated,
    }
}

fn action_view_from_task_action(action: &TaskAction) -> ActionView {
    let parameters = action
        .parameters
        .iter()
        .map(|(n, _)| n.clone())
        .collect::<Vec<_>>();
    let mut effects = Vec::new();
    for eff in &action.effects {
        if let Some(peffect) = literal_from_effect(&eff.effect) {
            let params = eff
                .parameters
                .iter()
                .map(|(n, _)| n.clone())
                .collect::<Vec<_>>();
            effects.push(EffectView {
                parameters: params,
                condition: eff.condition.clone(),
                peffect,
            });
        }
    }
    ActionView {
        name: action.name.clone(),
        parameters,
        precondition: action.precondition.clone(),
        effects,
    }
}

fn duplicate_universal_effects(action: &ActionView) -> ActionView {
    let mut effects = Vec::new();
    let mut used_names: HashSet<String> = action.parameters.iter().cloned().collect();
    for eff in &action.effects {
        for p in &eff.parameters {
            used_names.insert(p.clone());
        }
    }
    let mut counter = 0usize;
    for eff in &action.effects {
        effects.push(eff.clone());
        if eff.parameters.is_empty() {
            continue;
        }
        let mut mapping: HashMap<String, String> = HashMap::new();
        let mut new_params = Vec::new();
        for p in &eff.parameters {
            loop {
                let candidate = format!("?u{}", counter);
                counter += 1;
                if !used_names.contains(&candidate) {
                    used_names.insert(candidate.clone());
                    mapping.insert(p.clone(), candidate.clone());
                    new_params.push(candidate);
                    break;
                }
            }
        }
        let new_condition = rename_vars_condition(&eff.condition, &mapping);
        let new_peffect = rename_vars_literal(&eff.peffect, &mapping);
        effects.push(EffectView {
            parameters: new_params,
            condition: new_condition,
            peffect: new_peffect,
        });
    }
    ActionView {
        name: action.name.clone(),
        parameters: action.parameters.clone(),
        precondition: action.precondition.clone(),
        effects,
    }
}

fn add_inequality_preconds(
    action: &ActionView,
    reachable_action_params: &Option<HashMap<String, Vec<Vec<String>>>>,
) -> ActionView {
    let Some(reachable) = reachable_action_params else {
        return action.clone();
    };
    if action.parameters.len() < 2 {
        return action.clone();
    }
    let params = match reachable.get(&action.name) {
        Some(p) => p,
        None => return action.clone(),
    };
    let mut inequal_pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..action.parameters.len() {
        for j in (i + 1)..action.parameters.len() {
            let mut any_equal = false;
            for tuple in params {
                if i < tuple.len() && j < tuple.len() && tuple[i] == tuple[j] {
                    any_equal = true;
                    break;
                }
            }
            if !any_equal {
                inequal_pairs.push((i, j));
            }
        }
    }
    if inequal_pairs.is_empty() {
        return action.clone();
    }
    let mut parts = vec![action.precondition.clone()];
    for (i, j) in inequal_pairs {
        let p1 = action.parameters[i].clone();
        let p2 = action.parameters[j].clone();
        parts.push(Condition::Not(Box::new(Condition::Atom(
            "=".to_string(),
            vec![p1, p2],
        ))));
    }
    ActionView {
        name: action.name.clone(),
        parameters: action.parameters.clone(),
        precondition: Condition::And(parts),
        effects: action.effects.clone(),
    }
}

struct BalanceChecker {
    actions: Vec<ActionView>,
    heavy_actions: Vec<ActionView>,
    predicates_to_add_actions: HashMap<String, HashSet<usize>>,
}

impl BalanceChecker {
    fn new(
        task: &NormalizableTask,
        reachable_action_params: Option<HashMap<String, Vec<Vec<String>>>>,
    ) -> Self {
        let mut predicates_to_add_actions: HashMap<String, HashSet<usize>> = HashMap::new();
        let mut actions = Vec::new();
        let mut heavy_actions = Vec::new();
        for act in &task.actions {
            let base = action_view_from_task_action(act);
            let action = add_inequality_preconds(&base, &reachable_action_params);
            let heavy_action = duplicate_universal_effects(&action);
            for eff in &action.effects {
                if !eff.peffect.negated {
                    predicates_to_add_actions
                        .entry(eff.peffect.predicate.clone())
                        .or_default()
                        .insert(actions.len());
                }
            }
            actions.push(action);
            heavy_actions.push(heavy_action);
        }
        BalanceChecker {
            actions,
            heavy_actions,
            predicates_to_add_actions,
        }
    }

    fn get_threats(&self, predicate: &str) -> HashSet<usize> {
        self.predicates_to_add_actions
            .get(predicate)
            .cloned()
            .unwrap_or_default()
    }

    fn get_action(&self, idx: usize) -> &ActionView {
        &self.actions[idx]
    }

    fn get_heavy_action(&self, idx: usize) -> &ActionView {
        &self.heavy_actions[idx]
    }
}

// Collect fluent predicates: those that occur in add/del effects of any action
fn get_fluent_predicates(task: &NormalizableTask) -> HashSet<String> {
    let mut fluents: HashSet<String> = HashSet::new();
    for action in &task.actions {
        for eff in &action.effects {
            if let Some(lit) = literal_from_effect(&eff.effect) {
                fluents.insert(lit.predicate);
            }
        }
    }
    fluents
}

// Build initial invariants analogous to python: for each fluent predicate produce
// InvariantPart for omitted_pos in [-1] + all positions and wrap into Invariant
fn get_initial_invariants(task: &NormalizableTask) -> Vec<Invariant> {
    let mut out: Vec<Invariant> = Vec::new();
    let fluent_names = get_fluent_predicates(task);
    for (pred, params) in &task.predicates {
        if !fluent_names.contains(pred) {
            continue;
        }
        let arity = params.len();
        let all_args: Vec<usize> = (0..arity).collect();
        let mut omitted_positions: Vec<i32> = vec![-1];
        for i in &all_args {
            omitted_positions.push(*i as i32);
        }
        for omitted in omitted_positions {
            let order: Vec<usize> = all_args
                .iter()
                .filter(|&&i| i as i32 != omitted)
                .cloned()
                .collect();
            let part = InvariantPart::new(pred.clone(), order, omitted);
            out.push(Invariant::new(vec![part]));
        }
    }
    out
}

fn check_balance(
    candidate: &Invariant,
    balance_checker: &BalanceChecker,
    enqueue_func: &mut impl FnMut(Invariant),
) -> bool {
    let mut actions_to_check: HashSet<usize> = HashSet::new();
    for part in &candidate.parts {
        actions_to_check.extend(balance_checker.get_threats(&part.predicate));
    }
    for action_idx in actions_to_check {
        let heavy_action = balance_checker.get_heavy_action(action_idx);
        if candidate.operator_too_heavy(heavy_action) {
            return false;
        }
        let action = balance_checker.get_action(action_idx);
        if candidate.operator_unbalanced(action, enqueue_func) {
            return false;
        }
    }
    true
}

fn find_invariants(
    task: &NormalizableTask,
    reachable_action_params: Option<HashMap<String, Vec<Vec<String>>>>,
) -> Vec<Invariant> {
    let limit_candidates = 100000usize;
    let max_time_secs = 300u64;

    let mut candidates: VecDeque<Invariant> = VecDeque::new();
    let initial = get_initial_invariants(task);
    for inv in initial.into_iter().take(limit_candidates) {
        candidates.push_back(inv);
    }

    let mut seen: HashSet<Invariant> = HashSet::new();
    for c in candidates.iter() {
        seen.insert(c.clone());
    }

    let balance_checker = BalanceChecker::new(task, reachable_action_params);
    let mut found: Vec<Invariant> = Vec::new();
    let start = Instant::now();

    while let Some(candidate) = candidates.pop_front() {
        if start.elapsed().as_secs() > max_time_secs {
            break;
        }
        let mut pending: Vec<Invariant> = Vec::new();
        {
            let mut enqueue_func = |inv: Invariant| {
                if seen.len() < limit_candidates && !seen.contains(&inv) {
                    seen.insert(inv.clone());
                    pending.push(inv);
                }
            };
            if check_balance(&candidate, &balance_checker, &mut enqueue_func) {
                found.push(candidate);
            }
        }
        for inv in pending {
            candidates.push_back(inv);
        }
    }
    found
}

fn useful_groups(invariants: &[Invariant], initial_facts: &[SExpr]) -> Vec<Vec<String>> {
    let mut predicate_to_invariants: HashMap<String, Vec<Invariant>> = HashMap::new();
    for inv in invariants {
        for pred in &inv.predicates {
            predicate_to_invariants
                .entry(pred.clone())
                .or_default()
                .push(inv.clone());
        }
    }

    let mut nonempty_groups: HashSet<(Invariant, Vec<String>)> = HashSet::new();
    let mut overcrowded_groups: HashSet<(Invariant, Vec<String>)> = HashSet::new();

    for sexpr in initial_facts {
        let Some(atom) = literal_from_sexpr(sexpr) else {
            continue;
        };
        if atom.predicate == "=" {
            continue;
        }
        if let Some(invariants_for_pred) = predicate_to_invariants.get(&atom.predicate) {
            for inv in invariants_for_pred {
                let params = inv.get_parameters_for_atom(&atom);
                let key = (inv.clone(), params.clone());
                if !nonempty_groups.contains(&key) {
                    nonempty_groups.insert(key);
                } else {
                    overcrowded_groups.insert(key);
                }
            }
        }
    }

    let useful: Vec<(Invariant, Vec<String>)> = nonempty_groups
        .difference(&overcrowded_groups)
        .cloned()
        .collect();
    let mut result = Vec::new();
    for (inv, params) in useful {
        result.push(inv.instantiate(&params));
    }
    result
}

pub fn get_groups(
    task: &NormalizableTask,
    reachable_action_params: Option<HashMap<String, Vec<Vec<String>>>>,
) -> Vec<Vec<String>> {
    let mut invariants = find_invariants(task, reachable_action_params);
    let inv_key = |inv: &Invariant| {
        let mut parts = inv.parts.clone();
        parts.sort_by(|x, y| {
            let pc = x.predicate.cmp(&y.predicate);
            if pc != std::cmp::Ordering::Equal {
                pc
            } else {
                x.order.cmp(&y.order)
            }
        });
        let mut key = String::new();
        for part in parts {
            key.push_str(&part.predicate);
            key.push(':');
            for pos in &part.order {
                key.push_str(&format!("{}", pos));
                key.push(',');
            }
            key.push(';');
        }
        key
    };
    invariants.sort_by(|a, b| inv_key(a).cmp(&inv_key(b)));
    useful_groups(&invariants, &task.init)
}

// Stub matching Python API: compute reachable action params for the task.
// Full implementation would analyze the task to determine which parameter
// tuples are reachable for each action. For now return None to indicate
// we don't provide additional inequality preconditions.
pub fn get_reachable_action_params(
    _task: &NormalizableTask,
) -> Option<HashMap<String, Vec<Vec<String>>>> {
    None
}
