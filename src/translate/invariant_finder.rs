use crate::translate::invariants::{Invariant, InvariantPart};
use crate::translate::pddl_ast::{Condition, Domain, Problem};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

// Port of python/translate/invariant_finder.py (essential flow):
// - construct initial candidates (Invariant with one part per fluent predicate)
// - run balance checks with BalanceChecker and a candidate queue
// - produce useful_groups by instantiating found invariants over init facts

struct BalanceChecker {
    #[allow(dead_code)]
    predicates_to_add_actions: HashMap<String, HashSet<String>>, // predicate -> set of action names
}

impl BalanceChecker {
    fn new(
        domain: &Domain,
        _reachable_action_params: Option<HashMap<String, Vec<Vec<String>>>>,
    ) -> Self {
        let mut predicates_to_add_actions: HashMap<String, HashSet<String>> = HashMap::new();
        for act in &domain.actions {
            if let Some(eff_s) = &act.effect {
                // parse SExpr into Effect
                let eff = crate::translate::pddl_ast::sexpr_to_effect(eff_s);
                match eff {
                    crate::translate::pddl_ast::Effect::Add(pred, _args) => {
                        predicates_to_add_actions
                            .entry(pred.clone())
                            .or_default()
                            .insert(act.name.clone());
                    }
                    crate::translate::pddl_ast::Effect::And(v) => {
                        for sub in v {
                            match sub {
                                crate::translate::pddl_ast::Effect::Add(pred, _args) => {
                                    predicates_to_add_actions
                                        .entry(pred.clone())
                                        .or_default()
                                        .insert(act.name.clone());
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        BalanceChecker {
            predicates_to_add_actions,
        }
    }

    #[allow(dead_code)]
    fn get_threats(&self, predicate: &str) -> HashSet<String> {
        self.predicates_to_add_actions
            .get(predicate)
            .cloned()
            .unwrap_or_default()
    }
}

// Collect fluent predicates: those that occur in add/del effects of any action
fn get_fluent_predicates(domain: &Domain) -> HashSet<String> {
    let mut fluents: HashSet<String> = HashSet::new();
    for act in &domain.actions {
        if let Some(eff_s) = &act.effect {
            let eff = crate::translate::pddl_ast::sexpr_to_effect(eff_s);
            fn collect(eff: &crate::translate::pddl_ast::Effect, fluents: &mut HashSet<String>) {
                match eff {
                    crate::translate::pddl_ast::Effect::Add(name, _)
                    | crate::translate::pddl_ast::Effect::Del(name, _) => {
                        fluents.insert(name.clone());
                    }
                    crate::translate::pddl_ast::Effect::And(list) => {
                        for sub in list {
                            collect(sub, fluents);
                        }
                    }
                    _ => {}
                }
            }
            collect(&eff, &mut fluents);
        }
    }
    fluents
}

// Build initial invariants analogous to python: for each fluent predicate produce
// InvariantPart for omitted_pos in [-1] + all positions and wrap into Invariant
fn get_initial_invariants(domain: &Domain) -> Vec<Invariant> {
    let mut out: Vec<Invariant> = Vec::new();
    let fluent_names = get_fluent_predicates(domain);
    for (pred, params) in &domain.predicates {
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

// Generate unique variable names for an invariant relative to an action's parameters
fn find_unique_variables(
    action: &crate::translate::pddl_ast::Action,
    invariant: &Invariant,
) -> Vec<String> {
    let mut params: HashSet<String> = HashSet::new();
    for (p, _t) in &action.parameters {
        params.insert(p.clone());
    }
    // also collect any names in the effect SExpr if possible (conservative: none)
    let mut inv_vars: Vec<String> = Vec::new();
    let mut counter: usize = 0;
    for _ in 0..invariant.parts[0].arity() {
        loop {
            let candidate = format!("?v{}", counter);
            counter += 1;
            if !params.contains(&candidate) {
                inv_vars.push(candidate);
                break;
            }
        }
    }
    inv_vars
}

pub fn get_groups(domain: &Domain, problem: &Problem) -> Vec<Vec<String>> {
    // Candidate generation with limit and timing from options; we use conservative defaults
    let limit_candidates = 100000usize;
    let max_time_secs = 300u64;

    let mut candidates: VecDeque<Invariant> = VecDeque::new();
    let initial = get_initial_invariants(domain);
    for inv in initial.into_iter().take(limit_candidates) {
        candidates.push_back(inv);
    }

    let mut seen: HashSet<Invariant> = HashSet::new();
    for c in candidates.iter() {
        seen.insert(c.clone());
    }

    let _balance_checker = BalanceChecker::new(domain, None);

    let start = Instant::now();
    let mut found: Vec<Invariant> = Vec::new();
    while let Some(candidate) = candidates.pop_front() {
        if start.elapsed().as_secs() > max_time_secs {
            break;
        }
        // perform a conservative 'operator_too_heavy' check: if any action can add two
        // facts from the invariant such that constraints allow them simultaneously,
        // reject the candidate.
        let mut reject = false;
        for act in &domain.actions {
            // parse effect SExpr into structured Effect
            if let Some(eff_s) = &act.effect {
                let eff = crate::translate::pddl_ast::sexpr_to_effect(eff_s);
                // collect add effects whose predicate is part of the candidate
                let mut add_effects: Vec<(String, Vec<String>)> = Vec::new();
                match eff {
                    crate::translate::pddl_ast::Effect::Add(ref name, ref args) => {
                        if candidate.predicates.contains(name) {
                            add_effects.push((name.clone(), args.clone()));
                        }
                    }
                    crate::translate::pddl_ast::Effect::And(ref vec_eff) => {
                        for sub in vec_eff {
                            if let crate::translate::pddl_ast::Effect::Add(ref name, ref args) = sub
                            {
                                if candidate.predicates.contains(name) {
                                    add_effects.push((name.clone(), args.clone()));
                                }
                            }
                        }
                    }
                    _ => {}
                }
                if add_effects.len() > 1 {
                    // for each pair, build a constraint system and test solvability
                    for i in 0..add_effects.len() {
                        for j in (i + 1)..add_effects.len() {
                            let (ref n1, ref a1) = add_effects[i];
                            let (ref n2, ref a2) = add_effects[j];
                            // build peffect as Condition::Atom
                            let c1 = Condition::Atom(n1.clone(), a1.clone());
                            let c2 = Condition::Atom(n2.clone(), a2.clone());
                            let mut system = crate::translate::constraints::ConstraintSystem::new();
                            // ensure inequality and covers
                            crate::translate::invariants::ensure_inequality(&mut system, &c1, &c2);
                            // inv_vars: generate unique variables for invariant arity
                            let inv_vars = find_unique_variables(act, &candidate);
                            crate::translate::invariants::ensure_cover(
                                &mut system,
                                &c1,
                                &candidate,
                                &inv_vars,
                            );
                            crate::translate::invariants::ensure_cover(
                                &mut system,
                                &c2,
                                &candidate,
                                &inv_vars,
                            );
                            // Note: ensure_conjunction_sat is a simplified no-op currently
                            if system.is_solvable() {
                                reject = true;
                                break;
                            }
                        }
                        if reject {
                            break;
                        }
                    }
                }
            }
            if reject {
                break;
            }
        }
        if !reject {
            found.push(candidate);
        }
    }

    // Build useful_groups: map each found invariant to instantiated groups present in problem.init
    // First, collect initial positive atoms from problem.init (string form)
    let mut init_atoms: Vec<String> = Vec::new();
    for sexpr in &problem.init {
        if let crate::translate::pddl_parser::SExpr::List(list) = sexpr {
            if list.is_empty() {
                continue;
            }
            if let crate::translate::pddl_parser::SExpr::Atom(pred) = &list[0] {
                if pred == "=" {
                    continue;
                }
                let args: Vec<String> = list[1..]
                    .iter()
                    .filter_map(|x| match x {
                        crate::translate::pddl_parser::SExpr::Atom(a) => Some(a.clone()),
                        _ => None,
                    })
                    .collect();
                let atom = format!("{}({})", pred, args.join(", "));
                init_atoms.push(atom);
            }
        }
    }

    // For each invariant, for each init atom with predicate in invariant.predicates, build group key (invariant, params)
    let mut predicate_to_invariants: HashMap<String, Vec<Invariant>> = HashMap::new();
    for inv in &found {
        for part in &inv.parts {
            predicate_to_invariants
                .entry(part.predicate.clone())
                .or_default()
                .push(inv.clone());
        }
    }

    let mut nonempty_groups: HashSet<(Invariant, Vec<String>)> = HashSet::new();
    let mut overcrowded: HashSet<(Invariant, Vec<String>)> = HashSet::new();
    for atom in &init_atoms {
        // parse "pred(a, b, ...)"
        if let Some(open) = atom.find('(') {
            if let Some(close) = atom.rfind(')') {
                let pred = &atom[..open];
                let args_str = &atom[open + 1..close];
                let args: Vec<String> = args_str.split(',').map(|s| s.trim().to_string()).collect();
                if pred == "=" {
                    continue;
                }
                if let Some(invs) = predicate_to_invariants.get(pred) {
                    for inv in invs {
                        // compute parameters via invariant.get_parameters_for_atom: we need a Condition::Atom
                        let cond = Condition::Atom(pred.to_string(), args.clone());
                        let params = inv.get_parameters_for_atom(&cond);
                        let key = (inv.clone(), params.clone());
                        if !nonempty_groups.contains(&key) {
                            nonempty_groups.insert(key);
                        } else {
                            overcrowded.insert(key);
                        }
                    }
                }
            }
        }
    }

    let useful: Vec<(Invariant, Vec<String>)> =
        nonempty_groups.difference(&overcrowded).cloned().collect();
    // instantiate groups: for each (invariant, parameters) produce vector of strings via invariant.instantiate
    let mut groups_out: Vec<Vec<String>> = Vec::new();
    for (inv, params) in useful {
        let inst = inv.instantiate(&params);
        groups_out.push(inst);
    }

    groups_out
}

// Stub matching Python API: compute reachable action params for the task.
// Full implementation would analyze the task to determine which parameter
// tuples are reachable for each action. For now return None to indicate
// we don't provide additional inequality preconditions.
pub fn get_reachable_action_params(_domain: &Domain) -> Option<HashMap<String, Vec<Vec<String>>>> {
    None
}
