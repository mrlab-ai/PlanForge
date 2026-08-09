use itertools::Itertools;
/// Finds mutex invariants among ground atoms.
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;
use tracing::info;

use super::invariants::{BalanceChecker, Invariant, InvariantPart};
use super::options;
use super::pddl::actions::Action;
use super::pddl::conditions::*;
use super::pddl::tasks::Task;

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

/// Turns the invariants into the mutex groups that are useful as SAS
/// variables: those exactly one of whose facts holds initially. No fact means
/// the group would be empty, more than one means the initial state already
/// violates the invariant.
fn useful_groups(invariants: &[Invariant], initial_facts: &[Atom]) -> Vec<Vec<Atom>> {
    let mut invariants_by_predicate: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, invariant) in invariants.iter().enumerate() {
        for predicate in &invariant.predicates {
            invariants_by_predicate
                .entry(predicate)
                .or_default()
                .push(index);
        }
    }

    let mut nonempty: HashSet<(usize, Vec<String>)> = HashSet::new();
    let mut overcrowded: HashSet<(usize, Vec<String>)> = HashSet::new();
    for atom in initial_facts {
        let Some(candidates) = invariants_by_predicate.get(atom.predicate.as_str()) else {
            continue;
        };
        let atom_condition = Condition::Atom(atom.clone());
        for &index in candidates {
            let group = (index, invariants[index].get_parameters(&atom_condition));
            if nonempty.contains(&group) {
                overcrowded.insert(group);
            } else {
                nonempty.insert(group);
            }
        }
    }

    nonempty
        .difference(&overcrowded)
        .map(|(index, parameters)| {
            let mut parts: Vec<&InvariantPart> = invariants[*index].parts.iter().collect();
            parts.sort();
            parts
                .iter()
                .map(|part| part.instantiate(parameters))
                .collect()
        })
        .collect()
}

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
