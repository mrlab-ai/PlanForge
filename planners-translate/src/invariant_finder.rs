use itertools::Itertools;
/// Port of invariant_finder.py
/// Finds mutex invariants among ground atoms.
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;
use tracing::info;

use super::invariants::{BalanceChecker, Invariant, InvariantPart};
use super::options;
use super::pddl::actions::Action;
use super::pddl::conditions::*;
use super::pddl::tasks::Task;

/// Python: class BalanceChecker.__init__(self, task, reachable_action_params)
fn build_balance_checker(
    task: &Task,
    reachable_action_params: &Option<HashMap<String, Vec<Vec<String>>>>,
) -> BalanceChecker {
    let mut predicates_to_add_action_indices: HashMap<String, HashSet<usize>> = HashMap::new();
    let mut action_to_heavy_action: HashMap<usize, Action> = HashMap::new();
    let mut actions: Vec<Action> = vec![];

    for (idx, act) in task.actions.iter().enumerate() {
        let action = add_inequality_preconds(act, reachable_action_params);
        let mut too_heavy_effects = vec![];
        let mut create_heavy_act = false;

        for eff in &action.effects {
            too_heavy_effects.push(eff.clone());
            if !eff.parameters.is_empty() {
                create_heavy_act = true;
                too_heavy_effects.push(eff.clone());
            }
            // Check if it's an add effect (Atom, not negated)
            if let Condition::Atom(a) = &eff.peffect {
                predicates_to_add_action_indices
                    .entry(a.predicate.clone())
                    .or_default()
                    .insert(idx);
            }
        }

        if create_heavy_act {
            let heavy_act = Action {
                name: action.name.clone(),
                parameters: action.parameters.clone(),
                num_external_parameters: action.num_external_parameters,
                precondition: action.precondition.clone(),
                effects: too_heavy_effects,
                cost: action.cost.clone(),
                assign_effects: action.assign_effects.clone(),
            };
            action_to_heavy_action.insert(idx, heavy_act);
        }

        actions.push(action);
    }

    BalanceChecker {
        predicates_to_add_action_indices,
        action_to_heavy_action,
        actions,
    }
}

/// Python: def add_inequality_preconds(action, reachable_action_params)
fn add_inequality_preconds(
    action: &Action,
    reachable_action_params: &Option<HashMap<String, Vec<Vec<String>>>>,
) -> Action {
    let rap = match reachable_action_params {
        Some(r) => r,
        None => return action.clone(),
    };

    if action.parameters.len() < 2 {
        return action.clone();
    }

    let mut inequal_params = vec![];
    for combo in (0..action.parameters.len()).combinations(2) {
        let pos1 = combo[0];
        let pos2 = combo[1];
        if let Some(params_list) = rap.get(&action.name) {
            let mut all_different = true;
            for params in params_list {
                if params[pos1] == params[pos2] {
                    all_different = false;
                    break;
                }
            }
            if all_different {
                inequal_params.push((pos1, pos2));
            }
        }
    }

    if !inequal_params.is_empty() {
        let mut precond_parts = vec![action.precondition.clone()];
        for (pos1, pos2) in inequal_params {
            let param1 = action.parameters[pos1].name.clone();
            let param2 = action.parameters[pos2].name.clone();
            let new_cond =
                Condition::NegatedAtom(NegatedAtom::new("=".to_string(), vec![param1, param2]));
            precond_parts.push(new_cond);
        }
        // Simplified conjunction (Python calls .simplified())
        let precond = if precond_parts.len() == 1 {
            precond_parts.pop().unwrap()
        } else {
            Condition::Conjunction(Conjunction::new(precond_parts))
        };
        Action {
            name: action.name.clone(),
            parameters: action.parameters.clone(),
            num_external_parameters: action.num_external_parameters,
            precondition: precond,
            effects: action.effects.clone(),
            cost: action.cost.clone(),
            assign_effects: action.assign_effects.clone(),
        }
    } else {
        action.clone()
    }
}

/// Python: def get_fluents(task)
fn get_fluents(task: &Task) -> HashSet<String> {
    let mut fluent_names = HashSet::new();
    for action in &task.actions {
        for eff in &action.effects {
            match &eff.peffect {
                Condition::Atom(a) => {
                    fluent_names.insert(a.predicate.clone());
                }
                Condition::NegatedAtom(na) => {
                    fluent_names.insert(na.predicate.clone());
                }
                _ => {}
            }
        }
    }
    fluent_names
}

/// Python: def get_initial_invariants(task)
fn get_initial_invariants(task: &Task) -> Vec<Invariant> {
    let fluent_names = get_fluents(task);
    let mut result = vec![];
    for pred in &task.predicates {
        if !fluent_names.contains(&pred.name) {
            continue;
        }
        let all_args: Vec<usize> = (0..pred.arguments.len()).collect();
        // Try with omitted_arg = -1 (no omitted position)
        {
            let order = all_args.clone();
            let part = InvariantPart::new(pred.name.clone(), order, -1);
            result.push(Invariant::new(vec![part]));
        }
        // Try omitting each arg position
        for &omitted_arg in &all_args {
            let order: Vec<usize> = all_args
                .iter()
                .filter(|&&i| i != omitted_arg)
                .cloned()
                .collect();
            let part = InvariantPart::new(pred.name.clone(), order, omitted_arg as i32);
            result.push(Invariant::new(vec![part]));
        }
    }
    result
}

/// Python: def find_invariants(task, reachable_action_params)
fn find_invariants(
    task: &Task,
    reachable_action_params: &Option<HashMap<String, Vec<Vec<String>>>>,
) -> Vec<Invariant> {
    let limit = options::INVARIANT_GENERATION_MAX_CANDIDATES;
    let initial = get_initial_invariants(task);
    let mut candidates: VecDeque<Invariant> = initial.into_iter().take(limit).collect();
    info!("{} initial candidates", candidates.len());
    let mut seen_candidates: HashSet<Invariant> = candidates.iter().cloned().collect();

    let balance_checker = build_balance_checker(task, reachable_action_params);

    let start_time = Instant::now();
    let mut result = vec![];

    while let Some(candidate) = candidates.pop_front() {
        if start_time.elapsed().as_secs() > options::INVARIANT_GENERATION_MAX_TIME {
            info!("Time limit reached, aborting invariant generation");
            return result;
        }

        let mut enqueue_func = |invariant: Invariant| {
            if seen_candidates.len() < limit && !seen_candidates.contains(&invariant) {
                seen_candidates.insert(invariant.clone());
                candidates.push_back(invariant);
            }
        };

        if candidate.check_balance(&balance_checker, &mut enqueue_func) {
            result.push(candidate);
        }
    }

    result
}

/// Python: def useful_groups(invariants, initial_facts)
fn useful_groups(invariants: &[Invariant], initial_facts: &[Atom]) -> Vec<Vec<Atom>> {
    let mut predicate_to_invariants: HashMap<String, Vec<&Invariant>> = HashMap::new();
    for inv in invariants {
        for pred in &inv.predicates {
            predicate_to_invariants
                .entry(pred.clone())
                .or_default()
                .push(inv);
        }
    }

    let mut nonempty_groups: HashSet<(usize, Vec<String>)> = HashSet::new();
    let mut overcrowded_groups: HashSet<(usize, Vec<String>)> = HashSet::new();

    // Map invariants to indices for hashing
    let inv_to_idx: HashMap<*const Invariant, usize> = invariants
        .iter()
        .enumerate()
        .map(|(i, inv)| (inv as *const Invariant, i))
        .collect();

    for atom in initial_facts {
        let atom_cond = Condition::Atom(atom.clone());
        if let Some(inv_list) = predicate_to_invariants.get(&atom.predicate) {
            for inv in inv_list {
                let params = inv.get_parameters(&atom_cond);
                let inv_idx = inv_to_idx[&(*inv as *const Invariant)];
                let group_key = (inv_idx, params);
                if !nonempty_groups.contains(&group_key) {
                    nonempty_groups.insert(group_key);
                } else {
                    overcrowded_groups.insert(group_key);
                }
            }
        }
    }

    let useful = &nonempty_groups - &overcrowded_groups;
    let mut groups: Vec<Vec<Atom>> = vec![];
    for (inv_idx, parameters) in useful {
        let inv = &invariants[inv_idx];
        let mut parts: Vec<&InvariantPart> = inv.parts.iter().collect();
        parts.sort();
        let group: Vec<Atom> = parts
            .iter()
            .map(|part| part.instantiate(&parameters))
            .collect();
        groups.push(group);
    }
    groups
}

/// Python: def get_groups(task, reachable_action_params)
/// Main entry point: finds groups of mutex atoms.
pub fn get_groups(
    task: &Task,
    reachable_action_params: &Option<HashMap<String, Vec<Vec<String>>>>,
) -> Vec<Vec<Atom>> {
    info!("Finding invariants...");
    let mut invariants = find_invariants(task, reachable_action_params);
    invariants.sort();
    info!("Found {} invariants", invariants.len());

    info!("Checking invariant weight...");
    let groups = useful_groups(&invariants, &task.init);
    info!("Found {} useful groups", groups.len());
    groups
}
