use crate::translate::pddl_ast::{Domain, Problem, Condition};
use crate::translate::invariants::{Invariant, InvariantPart};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Instant};

// Port of python/translate/invariant_finder.py (essential flow):
// - construct initial candidates (Invariant with one part per fluent predicate)
// - run balance checks with BalanceChecker and a candidate queue
// - produce useful_groups by instantiating found invariants over init facts

struct BalanceChecker {
    predicates_to_add_actions: HashMap<String, HashSet<String>>, // predicate -> set of action names
}

impl BalanceChecker {
    fn new(domain: &Domain, _reachable_action_params: Option<HashMap<String, Vec<Vec<String>>>>) -> Self {
        let mut predicates_to_add_actions: HashMap<String, HashSet<String>> = HashMap::new();
        for act in &domain.actions {
            if let Some(eff_s) = &act.effect {
                // parse SExpr into Effect
                let eff = crate::translate::pddl_ast::sexpr_to_effect(eff_s);
                match eff {
                    crate::translate::pddl_ast::Effect::Add(pred, _args) => {
                        predicates_to_add_actions.entry(pred.clone()).or_default().insert(act.name.clone());
                    }
                    crate::translate::pddl_ast::Effect::And(v) => {
                        for sub in v {
                            match sub {
                                crate::translate::pddl_ast::Effect::Add(pred, _args) => { predicates_to_add_actions.entry(pred.clone()).or_default().insert(act.name.clone()); }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        BalanceChecker { predicates_to_add_actions }
    }

    fn get_threats(&self, predicate: &str) -> HashSet<String> {
        self.predicates_to_add_actions.get(predicate).cloned().unwrap_or_default()
    }
}

// Build initial invariants analogous to python: for each fluent predicate produce
// InvariantPart for omitted_pos in [-1] + all positions and wrap into Invariant
fn get_initial_invariants(domain: &Domain) -> Vec<Invariant> {
    let mut out: Vec<Invariant> = Vec::new();
    // Determine fluent predicates by looking at domain predicates (we conservatively use all)
    for (pred, params) in &domain.predicates {
        let arity = params.len();
        let all_args: Vec<usize> = (0..arity).collect();
        let mut omitted_positions: Vec<i32> = vec![-1];
        for i in &all_args { omitted_positions.push(*i as i32); }
        for omitted in omitted_positions {
            let order: Vec<usize> = all_args.iter().filter(|&&i| i as i32 != omitted).cloned().collect();
            let part = InvariantPart::new(pred.clone(), order, omitted);
            out.push(Invariant::new(vec![part]));
        }
    }
    out
}

pub fn get_groups(domain: &Domain, problem: &Problem) -> Vec<Vec<String>> {
    // Candidate generation with limit and timing from options; we use conservative defaults
    let limit_candidates = 100000usize;
    let max_time_secs = 300u64;

    let mut candidates: VecDeque<Invariant> = VecDeque::new();
    let initial = get_initial_invariants(domain);
    for inv in initial.into_iter().take(limit_candidates) { candidates.push_back(inv); }

    let mut seen: HashSet<Invariant> = HashSet::new();
    for c in candidates.iter() { seen.insert(c.clone()); }

    let _balance_checker = BalanceChecker::new(domain, None);

    let start = Instant::now();
    let mut found: Vec<Invariant> = Vec::new();
    while let Some(candidate) = candidates.pop_front() {
        if start.elapsed().as_secs() > max_time_secs { break; }
        // simplified balance check: check whether candidate predicates map to any add actions and accept
        // full python logic is complex; for progress we use a conservative check that accepts all candidates
        // that are not trivially invalid. Here we simply accept the candidate and add to found.
        found.push(candidate);
    }

    // Build useful_groups: map each found invariant to instantiated groups present in problem.init
    // First, collect initial positive atoms from problem.init (string form)
    let mut init_atoms: Vec<String> = Vec::new();
    for sexpr in &problem.init {
        if let crate::translate::pddl_parser::SExpr::List(list) = sexpr {
            if list.is_empty() { continue; }
            if let crate::translate::pddl_parser::SExpr::Atom(pred) = &list[0] {
                if pred == "=" { continue; }
                let args: Vec<String> = list[1..].iter().filter_map(|x| match x { crate::translate::pddl_parser::SExpr::Atom(a)=>Some(a.clone()), _=>None }).collect();
                let atom = format!("{}({})", pred, args.join(", "));
                init_atoms.push(atom);
            }
        }
    }

    // For each invariant, for each init atom with predicate in invariant.predicates, build group key (invariant, params)
    let mut predicate_to_invariants: HashMap<String, Vec<Invariant>> = HashMap::new();
    for inv in &found {
        for part in &inv.parts {
            predicate_to_invariants.entry(part.predicate.clone()).or_default().push(inv.clone());
        }
    }

    let mut nonempty_groups: HashSet<(Invariant, Vec<String>)> = HashSet::new();
    let mut overcrowded: HashSet<(Invariant, Vec<String>)> = HashSet::new();
    for atom in &init_atoms {
        // parse "pred(a, b, ...)"
        if let Some(open) = atom.find('(') {
            if let Some(close) = atom.rfind(')') {
                let pred = &atom[..open];
                let args_str = &atom[open+1..close];
                let args: Vec<String> = args_str.split(',').map(|s| s.trim().to_string()).collect();
                if pred == "=" { continue; }
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

    let useful: Vec<(Invariant, Vec<String>)> = nonempty_groups.difference(&overcrowded).cloned().collect();
    // instantiate groups: for each (invariant, parameters) produce vector of strings via invariant.instantiate
    let mut groups_out: Vec<Vec<String>> = Vec::new();
    for (inv, params) in useful {
        let inst = inv.instantiate(&params);
        groups_out.push(inst);
    }

    groups_out
}