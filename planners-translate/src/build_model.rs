use super::pddl_to_prolog::{Fact, PrologProgram, RuleType};
/// Port of build_model.py
/// Forward-chaining model builder for grounding.
use std::collections::{HashMap, HashSet};

/// An atom in the model: predicate + arguments.
/// Arguments can be integers (variable positions) or strings (constants/variables).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Arg {
    Pos(usize),  // refers to position in effect
    Str(String), // constant or unbound variable
}

/// An internal atom representation for the model builder.
#[derive(Debug, Clone)]
struct InternalAtom {
    predicate: String,
    args: Vec<Arg>,
}

/// Converts variables to position numbers as in Python's `variables_to_numbers`.
fn variables_to_numbers(
    effect: &[String],
    conditions: &[Vec<String>],
) -> (InternalAtom, Vec<InternalAtom>) {
    let mut rename_map: HashMap<String, usize> = HashMap::new();
    let mut new_effect_args = Vec::new();

    for (i, arg) in effect[1..].iter().enumerate() {
        if arg.starts_with('?') {
            rename_map.insert(arg.clone(), i);
            new_effect_args.push(Arg::Pos(i));
        } else {
            new_effect_args.push(Arg::Str(arg.clone()));
        }
    }

    let new_effect = InternalAtom {
        predicate: effect[0].clone(),
        args: new_effect_args,
    };

    let new_conditions: Vec<InternalAtom> = conditions
        .iter()
        .map(|cond| {
            let args = cond[1..]
                .iter()
                .map(|arg| {
                    if let Some(&pos) = rename_map.get(arg) {
                        Arg::Pos(pos)
                    } else {
                        Arg::Str(arg.clone())
                    }
                })
                .collect();
            InternalAtom {
                predicate: cond[0].clone(),
                args,
            }
        })
        .collect();

    (new_effect, new_conditions)
}

/// A ground atom in the model.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GroundAtom {
    predicate: String,
    args: Vec<String>,
}

impl GroundAtom {
    fn new(predicate: String, args: Vec<String>) -> Self {
        GroundAtom { predicate, args }
    }

    fn to_fact(&self) -> Fact {
        let mut atom = vec![self.predicate.clone()];
        atom.extend(self.args.clone());
        Fact::new(atom)
    }
}

// ============== Build Rules ==============

trait BuildRule {
    fn prepare_effect(&self, new_atom: &GroundAtom, cond_index: usize) -> Vec<Option<String>>;
    fn update_index(&mut self, new_atom: &GroundAtom, cond_index: usize);
    fn fire(
        &self,
        new_atom: &GroundAtom,
        cond_index: usize,
        enqueue: &mut dyn FnMut(&str, &[Option<String>]),
    );
    fn conditions(&self) -> &[InternalAtom];
}

fn prepare_effect_impl(
    effect: &InternalAtom,
    conditions: &[InternalAtom],
    new_atom: &GroundAtom,
    cond_index: usize,
) -> Vec<Option<String>> {
    let mut effect_args: Vec<Option<String>> = effect
        .args
        .iter()
        .map(|a| match a {
            Arg::Str(s) => Some(s.clone()),
            Arg::Pos(_) => None,
        })
        .collect();

    let cond = &conditions[cond_index];
    for (var_no, obj) in cond.args.iter().zip(new_atom.args.iter()) {
        if let Arg::Pos(pos) = var_no {
            effect_args[*pos] = Some(obj.clone());
        }
    }
    effect_args
}

// ============== JoinRule ==============

struct JoinRule {
    effect: InternalAtom,
    conditions: Vec<InternalAtom>,
    common_var_positions: [Vec<usize>; 2],
    atoms_by_key: [HashMap<Vec<String>, Vec<GroundAtom>>; 2],
}

impl JoinRule {
    fn new(effect: InternalAtom, conditions: Vec<InternalAtom>) -> Self {
        assert_eq!(conditions.len(), 2);

        let left_args = &conditions[0].args;
        let right_args = &conditions[1].args;

        let left_vars: HashSet<usize> = left_args
            .iter()
            .filter_map(|a| match a {
                Arg::Pos(p) => Some(*p),
                _ => None,
            })
            .collect();
        let right_vars: HashSet<usize> = right_args
            .iter()
            .filter_map(|a| match a {
                Arg::Pos(p) => Some(*p),
                _ => None,
            })
            .collect();

        let mut common_vars: Vec<usize> = left_vars.intersection(&right_vars).cloned().collect();
        common_vars.sort();

        let left_positions: Vec<usize> = common_vars
            .iter()
            .map(|var| {
                left_args
                    .iter()
                    .position(|a| matches!(a, Arg::Pos(p) if p == var))
                    .unwrap()
            })
            .collect();
        let right_positions: Vec<usize> = common_vars
            .iter()
            .map(|var| {
                right_args
                    .iter()
                    .position(|a| matches!(a, Arg::Pos(p) if p == var))
                    .unwrap()
            })
            .collect();

        JoinRule {
            effect,
            conditions,
            common_var_positions: [left_positions, right_positions],
            atoms_by_key: [HashMap::new(), HashMap::new()],
        }
    }
}

impl BuildRule for JoinRule {
    fn prepare_effect(&self, new_atom: &GroundAtom, cond_index: usize) -> Vec<Option<String>> {
        prepare_effect_impl(&self.effect, &self.conditions, new_atom, cond_index)
    }

    fn update_index(&mut self, new_atom: &GroundAtom, cond_index: usize) {
        let key: Vec<String> = self.common_var_positions[cond_index]
            .iter()
            .map(|&pos| new_atom.args[pos].clone())
            .collect();
        self.atoms_by_key[cond_index]
            .entry(key)
            .or_default()
            .push(new_atom.clone());
    }

    fn fire(
        &self,
        new_atom: &GroundAtom,
        cond_index: usize,
        enqueue: &mut dyn FnMut(&str, &[Option<String>]),
    ) {
        let effect_args = self.prepare_effect(new_atom, cond_index);
        let key: Vec<String> = self.common_var_positions[cond_index]
            .iter()
            .map(|&pos| new_atom.args[pos].clone())
            .collect();
        let other_cond_index = 1 - cond_index;
        let other_cond = &self.conditions[other_cond_index];

        if let Some(atoms) = self.atoms_by_key[other_cond_index].get(&key) {
            for atom in atoms {
                // Reset effect args to partially filled state
                let mut ea = effect_args.clone();
                for (var_no, obj) in other_cond.args.iter().zip(atom.args.iter()) {
                    if let Arg::Pos(pos) = var_no {
                        ea[*pos] = Some(obj.clone());
                    }
                }
                enqueue(&self.effect.predicate, &ea);
            }
        }
    }

    fn conditions(&self) -> &[InternalAtom] {
        &self.conditions
    }
}

// ============== ProductRule ==============

struct ProductRule {
    effect: InternalAtom,
    conditions: Vec<InternalAtom>,
    atoms_by_index: Vec<Vec<GroundAtom>>,
    empty_atom_list_no: usize,
}

impl ProductRule {
    fn new(effect: InternalAtom, conditions: Vec<InternalAtom>) -> Self {
        let n = conditions.len();
        ProductRule {
            effect,
            atoms_by_index: vec![vec![]; n],
            empty_atom_list_no: n,
            conditions,
        }
    }

    fn get_bindings(atom: &GroundAtom, cond: &InternalAtom) -> Vec<(usize, String)> {
        cond.args
            .iter()
            .zip(atom.args.iter())
            .filter_map(|(var_no, obj)| {
                if let Arg::Pos(pos) = var_no {
                    Some((*pos, obj.clone()))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl BuildRule for ProductRule {
    fn prepare_effect(&self, new_atom: &GroundAtom, cond_index: usize) -> Vec<Option<String>> {
        prepare_effect_impl(&self.effect, &self.conditions, new_atom, cond_index)
    }

    fn update_index(&mut self, new_atom: &GroundAtom, cond_index: usize) {
        if self.atoms_by_index[cond_index].is_empty() {
            self.empty_atom_list_no -= 1;
        }
        self.atoms_by_index[cond_index].push(new_atom.clone());
    }

    fn fire(
        &self,
        new_atom: &GroundAtom,
        cond_index: usize,
        enqueue: &mut dyn FnMut(&str, &[Option<String>]),
    ) {
        if self.empty_atom_list_no > 0 {
            return;
        }

        // Build bindings factors from all other conditions
        let mut bindings_factors: Vec<Vec<Vec<(usize, String)>>> = vec![];
        for (pos, cond) in self.conditions.iter().enumerate() {
            if pos == cond_index {
                continue;
            }
            let atoms = &self.atoms_by_index[pos];
            let factor: Vec<Vec<(usize, String)>> = atoms
                .iter()
                .map(|atom| Self::get_bindings(atom, cond))
                .collect();
            bindings_factors.push(factor);
        }

        let eff_args = self.prepare_effect(new_atom, cond_index);

        // Compute cartesian product of bindings_factors
        let mut products: Vec<Vec<Vec<(usize, String)>>> = vec![vec![]];
        for factor in &bindings_factors {
            let mut new_products = vec![];
            for existing in &products {
                for bindings in factor {
                    let mut combined = existing.clone();
                    combined.push(bindings.clone());
                    new_products.push(combined);
                }
            }
            products = new_products;
        }

        for bindings_list in &products {
            let mut ea = eff_args.clone();
            for bindings in bindings_list {
                for (var_no, obj) in bindings {
                    ea[*var_no] = Some(obj.clone());
                }
            }
            enqueue(&self.effect.predicate, &ea);
        }
    }

    fn conditions(&self) -> &[InternalAtom] {
        &self.conditions
    }
}

// ============== ProjectRule ==============

struct ProjectRule {
    effect: InternalAtom,
    conditions: Vec<InternalAtom>,
}

impl ProjectRule {
    fn new(effect: InternalAtom, conditions: Vec<InternalAtom>) -> Self {
        assert_eq!(conditions.len(), 1);
        ProjectRule { effect, conditions }
    }
}

impl BuildRule for ProjectRule {
    fn prepare_effect(&self, new_atom: &GroundAtom, cond_index: usize) -> Vec<Option<String>> {
        prepare_effect_impl(&self.effect, &self.conditions, new_atom, cond_index)
    }

    fn update_index(&mut self, _new_atom: &GroundAtom, _cond_index: usize) {
        // No index needed for projection
    }

    fn fire(
        &self,
        new_atom: &GroundAtom,
        cond_index: usize,
        enqueue: &mut dyn FnMut(&str, &[Option<String>]),
    ) {
        let effect_args = self.prepare_effect(new_atom, cond_index);
        enqueue(&self.effect.predicate, &effect_args);
    }

    fn conditions(&self) -> &[InternalAtom] {
        &self.conditions
    }
}

// ============== Unifier ==============

/// A node in the unification trie.
enum Generator {
    Leaf(LeafGenerator),
    Match(MatchGenerator),
}

struct LeafGenerator {
    matches: Vec<(usize, usize)>, // (rule_index, cond_index)
}

struct MatchGenerator {
    index: usize,
    matches: Vec<(usize, usize)>,
    match_generator: HashMap<String, Box<Generator>>,
    next: Box<Generator>,
}

impl Generator {
    fn generate(&self, atom: &GroundAtom, result: &mut Vec<(usize, usize)>) {
        match self {
            Generator::Leaf(leaf) => {
                result.extend_from_slice(&leaf.matches);
            }
            Generator::Match(mg) => {
                result.extend_from_slice(&mg.matches);
                if mg.index < atom.args.len() {
                    if let Some(gener) = mg.match_generator.get(&atom.args[mg.index]) {
                        gener.generate(atom, result);
                    }
                }
                mg.next.generate(atom, result);
            }
        }
    }

    fn insert(self, args: &[(usize, String)], value: (usize, usize)) -> Generator {
        match self {
            Generator::Leaf(mut leaf) => {
                if args.is_empty() {
                    leaf.matches.push(value);
                    Generator::Leaf(leaf)
                } else {
                    let mut root = Generator::Leaf(LeafGenerator {
                        matches: vec![value],
                    });
                    for &(arg_index, ref arg) in args.iter().rev() {
                        let mut new_root = MatchGenerator {
                            index: arg_index,
                            matches: vec![],
                            match_generator: HashMap::new(),
                            next: Box::new(Generator::Leaf(LeafGenerator { matches: vec![] })),
                        };
                        new_root.match_generator.insert(arg.clone(), Box::new(root));
                        root = Generator::Match(new_root);
                    }
                    // Transfer existing matches
                    match &mut root {
                        Generator::Match(mg) => {
                            mg.matches = leaf.matches;
                        }
                        _ => unreachable!(),
                    }
                    root
                }
            }
            Generator::Match(mut mg) => {
                if args.is_empty() {
                    mg.matches.push(value);
                    Generator::Match(mg)
                } else {
                    let (arg_index, ref arg) = args[0];
                    if mg.index < arg_index {
                        let next = (*mg.next).insert(args, value);
                        mg.next = Box::new(next);
                        Generator::Match(mg)
                    } else if mg.index > arg_index {
                        let mut new_parent = MatchGenerator {
                            index: arg_index,
                            matches: vec![],
                            match_generator: HashMap::new(),
                            next: Box::new(Generator::Match(mg)),
                        };
                        let new_branch = Generator::Leaf(LeafGenerator { matches: vec![] })
                            .insert(&args[1..], value);
                        new_parent
                            .match_generator
                            .insert(arg.clone(), Box::new(new_branch));
                        Generator::Match(new_parent)
                    } else {
                        // mg.index == arg_index
                        let branch = mg.match_generator.remove(arg).unwrap_or_else(|| {
                            Box::new(Generator::Leaf(LeafGenerator { matches: vec![] }))
                        });
                        let new_branch = (*branch).insert(&args[1..], value);
                        mg.match_generator.insert(arg.clone(), Box::new(new_branch));
                        Generator::Match(mg)
                    }
                }
            }
        }
    }
}

struct Unifier {
    predicate_to_generator: HashMap<String, Generator>,
}

impl Unifier {
    fn new(rules: &[Box<dyn BuildRule>]) -> Self {
        let mut pred_to_gen: HashMap<String, Generator> = HashMap::new();

        for (rule_idx, rule) in rules.iter().enumerate() {
            for (cond_idx, cond) in rule.conditions().iter().enumerate() {
                let constant_args: Vec<(usize, String)> = cond
                    .args
                    .iter()
                    .enumerate()
                    .filter_map(|(i, arg)| match arg {
                        Arg::Str(s) if !s.starts_with('?') => Some((i, s.clone())),
                        _ => None,
                    })
                    .collect();

                let gener = pred_to_gen
                    .remove(&cond.predicate)
                    .unwrap_or(Generator::Leaf(LeafGenerator { matches: vec![] }));
                let new_gen = gener.insert(&constant_args, (rule_idx, cond_idx));
                pred_to_gen.insert(cond.predicate.clone(), new_gen);
            }
        }

        Unifier {
            predicate_to_generator: pred_to_gen,
        }
    }

    fn unify(&self, atom: &GroundAtom) -> Vec<(usize, usize)> {
        let mut result = vec![];
        if let Some(gener) = self.predicate_to_generator.get(&atom.predicate) {
            gener.generate(atom, &mut result);
        }
        result
    }
}

// ============== Queue ==============

struct Queue {
    queue: Vec<GroundAtom>,
    queue_pos: usize,
    enqueued: HashSet<Vec<String>>, // (pred, arg1, arg2, ...)
    num_pushes: usize,
}

impl Queue {
    fn new(atoms: Vec<GroundAtom>) -> Self {
        let enqueued: HashSet<Vec<String>> = atoms
            .iter()
            .map(|a| {
                let mut key = vec![a.predicate.clone()];
                key.extend(a.args.clone());
                key
            })
            .collect();
        let num_pushes = atoms.len();
        Queue {
            queue: atoms,
            queue_pos: 0,
            enqueued,
            num_pushes,
        }
    }

    fn is_empty(&self) -> bool {
        self.queue_pos >= self.queue.len()
    }

    fn push(&mut self, predicate: &str, args: &[Option<String>]) {
        self.num_pushes += 1;
        // Only enqueue if all args are bound
        let bound_args: Vec<String> = match args
            .iter()
            .map(|a| a.clone())
            .collect::<Option<Vec<String>>>()
        {
            Some(a) => a,
            None => return,
        };
        let mut key = vec![predicate.to_string()];
        key.extend(bound_args.clone());
        if self.enqueued.insert(key) {
            self.queue
                .push(GroundAtom::new(predicate.to_string(), bound_args));
        }
    }

    fn pop(&mut self) -> GroundAtom {
        let result = self.queue[self.queue_pos].clone();
        self.queue_pos += 1;
        result
    }
}

/// Convert program rules to typed BuildRule objects.
fn convert_rules(prog: &PrologProgram) -> Vec<Box<dyn BuildRule>> {
    let mut result: Vec<Box<dyn BuildRule>> = vec![];
    for rule in &prog.rules {
        let (new_effect, new_conditions) = variables_to_numbers(&rule.effect, &rule.conditions);
        let rule_type = rule.rule_type.as_ref().unwrap_or(&RuleType::Join);
        let build_rule: Box<dyn BuildRule> = match rule_type {
            RuleType::Join => Box::new(JoinRule::new(new_effect, new_conditions)),
            RuleType::Product => Box::new(ProductRule::new(new_effect, new_conditions)),
            RuleType::Project => Box::new(ProjectRule::new(new_effect, new_conditions)),
        };
        result.push(build_rule);
    }
    result
}

/// Python: compute_model(prog) -> list of Fact
/// Performs forward chaining to compute the model (set of reachable ground atoms).
pub fn compute_model(prog: &PrologProgram) -> Vec<Fact> {
    let mut rules = convert_rules(prog);
    let unifier = Unifier::new(&rules);

    // Convert program facts to GroundAtoms
    let mut fact_atoms: Vec<GroundAtom> = prog
        .facts
        .iter()
        .map(|f| GroundAtom::new(f.atom[0].clone(), f.atom[1..].to_vec()))
        .collect();
    fact_atoms.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));

    let mut queue = Queue::new(fact_atoms);

    println!("Generated {} rules.", rules.len());

    let mut relevant_atoms = 0;
    let mut auxiliary_atoms = 0;

    while !queue.is_empty() {
        let next_atom = queue.pop();
        if next_atom.predicate.contains('$') {
            auxiliary_atoms += 1;
        } else {
            relevant_atoms += 1;
        }

        let matches = unifier.unify(&next_atom);
        for (rule_idx, cond_idx) in matches {
            rules[rule_idx].update_index(&next_atom, cond_idx);
            rules[rule_idx].fire(&next_atom, cond_idx, &mut |pred, args| {
                queue.push(pred, args);
            });
        }
    }

    println!("{} relevant atoms", relevant_atoms);
    println!("{} auxiliary atoms", auxiliary_atoms);
    println!("{} final queue length", queue.queue.len());
    println!("{} total queue pushes", queue.num_pushes);

    queue.queue.into_iter().map(|a| a.to_fact()).collect()
}
