use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Arg {
    Var(usize),      // effect variable position index
    FreeVar(String), // projected variable like "?x"
    Const(String),   // object name
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Atom {
    pub predicate: String,
    pub args: Vec<Arg>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SymAtom {
    // symbolic: args are strings like "?x" or constants
    pub predicate: String,
    pub args: Vec<String>,
}

impl SymAtom {
    pub fn new<S: Into<String>>(pred: S, args: Vec<S>) -> Self {
        Self {
            predicate: pred.into(),
            args: args.into_iter().map(|s| s.into()).collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RuleSpec {
    pub rtype: String, // "join" | "product" | "project"
    pub effect: SymAtom,
    pub conditions: Vec<SymAtom>,
}

pub fn variables_to_numbers(effect: &SymAtom, conditions: &[SymAtom]) -> (Atom, Vec<Atom>) {
    let mut rename_map: HashMap<String, usize> = HashMap::new();
    let mut new_effect_args: Vec<Arg> = effect
        .args
        .iter()
        .enumerate()
        .map(|(i, a)| {
            if a.starts_with('?') {
                if let Some(&idx) = rename_map.get(a) {
                    Arg::Var(idx)
                } else {
                    rename_map.insert(a.clone(), i);
                    Arg::Var(i)
                }
            } else {
                Arg::Const(a.clone())
            }
        })
        .collect();
    // new_effect_args is constructed; in Python they overwrite, but we already set.
    let new_effect = Atom {
        predicate: effect.predicate.clone(),
        args: new_effect_args.drain(..).collect(),
    };
    let mut new_conditions: Vec<Atom> = Vec::new();
    for cond in conditions {
        let new_args: Vec<Arg> = cond
            .args
            .iter()
            .map(|a| {
                if let Some(&idx) = rename_map.get(a) {
                    Arg::Var(idx)
                } else if a.starts_with('?') {
                    Arg::FreeVar(a.clone())
                } else {
                    Arg::Const(a.clone())
                }
            })
            .collect();
        new_conditions.push(Atom {
            predicate: cond.predicate.clone(),
            args: new_args,
        });
    }
    (new_effect, new_conditions)
}

pub trait BuildRule {
    fn validate(&self);
    fn update_index(&mut self, new_atom: &Atom, cond_index: usize);
    fn fire(&self, new_atom: &Atom, cond_index: usize, enqueue: &mut dyn FnMut(&str, &Vec<String>));
    fn conditions(&self) -> &Vec<Atom>;
    fn effect(&self) -> &Atom;
}

#[derive(Clone)]
pub struct JoinRule {
    effect: Atom,
    conditions: Vec<Atom>,
    common_var_positions: [Vec<usize>; 2],
    atoms_by_key: [HashMap<Vec<String>, Vec<Atom>>; 2],
}

impl JoinRule {
    pub fn new(effect: Atom, conditions: Vec<Atom>) -> Self {
        assert_eq!(conditions.len(), 2);
        let left = &conditions[0];
        let right = &conditions[1];
        let left_vars: HashSet<usize> = left
            .args
            .iter()
            .filter_map(|a| match a {
                Arg::Var(i) => Some(*i),
                _ => None,
            })
            .collect();
        let right_vars: HashSet<usize> = right
            .args
            .iter()
            .filter_map(|a| match a {
                Arg::Var(i) => Some(*i),
                _ => None,
            })
            .collect();
        let mut common: Vec<usize> = left_vars.intersection(&right_vars).cloned().collect();
        common.sort();
        let positions_for = |cond: &Atom, commons: &Vec<usize>| -> Vec<usize> {
            let mut pos = Vec::new();
            for c in commons {
                let idx = cond
                    .args
                    .iter()
                    .position(|a| matches!(a, Arg::Var(v) if v==c))
                    .expect("var must appear in cond");
                pos.push(idx);
            }
            pos
        };
        let common_var_positions = [positions_for(left, &common), positions_for(right, &common)];
        let atoms_by_key = [HashMap::new(), HashMap::new()];
        Self {
            effect,
            conditions,
            common_var_positions,
            atoms_by_key,
        }
    }
    #[allow(dead_code)]
    fn prepare_effect(&self, new_atom: &Atom, cond_index: usize) -> Vec<String> {
        let cond = &self.conditions[cond_index];
        let mut bindings: HashMap<usize, String> = HashMap::new();
        for (arg_pos, arg) in cond.args.iter().enumerate() {
            if let Arg::Var(var_no) = arg {
                if let Arg::Const(ref obj) = new_atom.args[arg_pos] {
                    bindings.insert(*var_no, obj.clone());
                }
            }
        }
        self.effect
            .args
            .iter()
            .map(|a| match a {
                Arg::Var(var_no) => bindings.get(var_no).cloned().unwrap_or_default(),
                Arg::Const(c) => c.clone(),
                Arg::FreeVar(_) => "".to_string(),
            })
            .collect()
    }
}

impl BuildRule for JoinRule {
    fn validate(&self) {
        assert_eq!(self.conditions.len(), 2, "JoinRule must have 2 conditions");
        let left_vars: HashSet<_> = self.conditions[0]
            .args
            .iter()
            .filter_map(|a| match a {
                Arg::Var(i) => Some(*i),
                Arg::FreeVar(_s) => Some(usize::MAX),
                _ => None,
            })
            .collect();
        let right_vars: HashSet<_> = self.conditions[1]
            .args
            .iter()
            .filter_map(|a| match a {
                Arg::Var(i) => Some(*i),
                Arg::FreeVar(_s) => Some(usize::MAX),
                _ => None,
            })
            .collect();
        assert!(
            !left_vars.is_empty() && !right_vars.is_empty(),
            "JoinRule needs shared variables"
        );
    }
    fn update_index(&mut self, new_atom: &Atom, cond_index: usize) {
        if self.conditions[cond_index].args.len() != new_atom.args.len() {
            return;
        }
        let positions = &self.common_var_positions[cond_index];
        let mut key: Vec<String> = Vec::with_capacity(positions.len());
        for &p in positions {
            if let Arg::Const(ref v) = new_atom.args[p] {
                key.push(v.clone());
            }
        }
        self.atoms_by_key[cond_index]
            .entry(key)
            .or_default()
            .push(new_atom.clone());
    }
    fn fire(
        &self,
        new_atom: &Atom,
        cond_index: usize,
        enqueue: &mut dyn FnMut(&str, &Vec<String>),
    ) {
        if self.conditions[cond_index].args.len() != new_atom.args.len() {
            return;
        }
        let cond = &self.conditions[cond_index];
        let mut bindings: HashMap<usize, String> = HashMap::new();
        for (arg_pos, arg) in cond.args.iter().enumerate() {
            if let Arg::Var(var_no) = arg {
                if let Arg::Const(ref obj) = new_atom.args[arg_pos] {
                    bindings.insert(*var_no, obj.clone());
                }
            }
        }
        let positions = &self.common_var_positions[cond_index];
        let mut key: Vec<String> = Vec::with_capacity(positions.len());
        for &p in positions {
            if let Arg::Const(ref v) = new_atom.args[p] {
                key.push(v.clone());
            }
        }
        let other = 1 - cond_index;
        if let Some(list) = self.atoms_by_key[other].get(&key) {
            let other_cond = &self.conditions[other];
            for atom in list {
                let mut local_bindings = bindings.clone();
                for (i, a) in other_cond.args.iter().enumerate() {
                    if let Arg::Var(var_no) = a {
                        if let Arg::Const(ref obj) = atom.args[i] {
                            local_bindings.insert(*var_no, obj.clone());
                        }
                    }
                }
                let eff_args: Vec<String> = self
                    .effect
                    .args
                    .iter()
                    .map(|a| match a {
                        Arg::Var(var_no) => local_bindings.get(var_no).cloned().unwrap_or_default(),
                        Arg::Const(c) => c.clone(),
                        Arg::FreeVar(_) => "".to_string(),
                    })
                    .collect();
                if eff_args.iter().any(|a| a.is_empty()) {
                    continue;
                }
                enqueue(&self.effect.predicate, &eff_args);
            }
        }
    }
    fn conditions(&self) -> &Vec<Atom> {
        &self.conditions
    }
    fn effect(&self) -> &Atom {
        &self.effect
    }
}

#[derive(Clone)]
pub struct ProductRule {
    effect: Atom,
    conditions: Vec<Atom>,
    atoms_by_index: Vec<Vec<Atom>>, // per condition
    empty_atom_list_no: usize,
}

impl ProductRule {
    pub fn new(effect: Atom, conditions: Vec<Atom>) -> Self {
        let k = conditions.len();
        Self {
            effect,
            conditions,
            atoms_by_index: vec![Vec::new(); k],
            empty_atom_list_no: k,
        }
    }
    fn bindings_for(atom: &Atom, cond: &Atom) -> Vec<(usize, String)> {
        if cond.args.len() != atom.args.len() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for (i, a) in cond.args.iter().enumerate() {
            if let Arg::Var(var_no) = a {
                if let Arg::Const(ref obj) = atom.args[i] {
                    out.push((*var_no, obj.clone()));
                }
            }
        }
        out
    }
    fn prepare_effect(&self, new_atom: &Atom, cond_index: usize) -> Vec<String> {
        let cond = &self.conditions[cond_index];
        let mut bindings: HashMap<usize, String> = HashMap::new();
        for (i, a) in cond.args.iter().enumerate() {
            if let Arg::Var(var_no) = a {
                if let Arg::Const(ref obj) = new_atom.args[i] {
                    bindings.insert(*var_no, obj.clone());
                }
            }
        }
        self.effect
            .args
            .iter()
            .map(|a| match a {
                Arg::Var(var_no) => bindings.get(var_no).cloned().unwrap_or_default(),
                Arg::Const(c) => c.clone(),
                Arg::FreeVar(_) => "".to_string(),
            })
            .collect()
    }
}

impl BuildRule for ProductRule {
    fn validate(&self) {
        assert!(
            self.conditions.len() >= 2,
            "ProductRule needs >=2 conditions"
        );
        // Lightweight validation: ensure effect vars cover all condition Vars
        let eff_vars: HashSet<_> = self
            .effect
            .args
            .iter()
            .filter_map(|a| match a {
                Arg::Var(i) => Some(*i),
                _ => None,
            })
            .collect();
        let mut cond_vars: HashSet<usize> = HashSet::new();
        for c in &self.conditions {
            for a in &c.args {
                if let Arg::Var(i) = a {
                    cond_vars.insert(*i);
                }
            }
        }
        assert_eq!(
            eff_vars, cond_vars,
            "Effect vars must equal union of cond vars"
        );
    }
    fn update_index(&mut self, new_atom: &Atom, cond_index: usize) {
        if self.conditions[cond_index].args.len() != new_atom.args.len() {
            return;
        }
        let list = &mut self.atoms_by_index[cond_index];
        if list.is_empty() {
            self.empty_atom_list_no = self.empty_atom_list_no.saturating_sub(1);
        }
        list.push(new_atom.clone());
    }
    fn fire(
        &self,
        new_atom: &Atom,
        cond_index: usize,
        enqueue: &mut dyn FnMut(&str, &Vec<String>),
    ) {
        if self.conditions[cond_index].args.len() != new_atom.args.len() {
            return;
        }
        if self.empty_atom_list_no > 0 {
            return;
        }
        let base_cond = &self.conditions[cond_index];
        let mut base_bindings: HashMap<usize, String> = HashMap::new();
        for (i, a) in base_cond.args.iter().enumerate() {
            if let Arg::Var(var_no) = a {
                if let Arg::Const(ref obj) = new_atom.args[i] {
                    base_bindings.insert(*var_no, obj.clone());
                }
            }
        }
        // Build binding factors for all other conditions
        // factors: Vec of (one Vec of bindings per atom)
        // Each inner Vec represents the bindings from one atom matching the condition
        let mut factors: Vec<Vec<Vec<(usize, String)>>> = Vec::new();
        for (pos, cond) in self.conditions.iter().enumerate() {
            if pos == cond_index {
                continue;
            }
            let atoms = &self.atoms_by_index[pos];
            if atoms.is_empty() {
                return;
            }
            // Build a factor: one binding list per matching atom
            let factor: Vec<Vec<(usize, String)>> =
                atoms.iter().map(|a| Self::bindings_for(a, cond)).collect();
            factors.push(factor);
        }
        // Cartesian product: pick one atom from each condition, combine their bindings
        // factors[i] is a Vec of binding-lists, one per atom matching condition i
        // We want to iterate over all combinations, picking one binding-list from each factor
        fn product_apply(
            factors: &[Vec<Vec<(usize, String)>>],
            acc: &mut Vec<Vec<(usize, String)>>,
            emit: &mut dyn FnMut(&[Vec<(usize, String)>]),
        ) {
            if factors.is_empty() {
                emit(acc);
                return;
            }
            let (first, rest) = factors.split_first().unwrap();
            // first is a Vec<Vec<(usize, String)>> - one binding-list per atom
            for bindings in first {
                acc.push(bindings.clone());
                product_apply(rest, acc, emit);
                acc.pop();
            }
        }
        let mut tmp_acc: Vec<Vec<(usize, String)>> = Vec::new();
        product_apply(&factors, &mut tmp_acc, &mut |bindings_list| {
            let mut local_bindings = base_bindings.clone();
            for bindings in bindings_list {
                for (var_no, obj) in bindings {
                    local_bindings.insert(*var_no, obj.clone());
                }
            }
            let eff_args: Vec<String> = self
                .effect
                .args
                .iter()
                .map(|a| match a {
                    Arg::Var(var_no) => local_bindings.get(var_no).cloned().unwrap_or_default(),
                    Arg::Const(c) => c.clone(),
                    Arg::FreeVar(_) => "".to_string(),
                })
                .collect();
            if eff_args.iter().any(|a| a.is_empty()) {
                return;
            }
            enqueue(&self.effect.predicate, &eff_args);
        });
    }
    fn conditions(&self) -> &Vec<Atom> {
        &self.conditions
    }
    fn effect(&self) -> &Atom {
        &self.effect
    }
}

#[derive(Clone)]
pub struct ProjectRule {
    effect: Atom,
    conditions: Vec<Atom>,
}

impl ProjectRule {
    pub fn new(effect: Atom, conditions: Vec<Atom>) -> Self {
        Self { effect, conditions }
    }
    fn prepare_effect(&self, new_atom: &Atom, cond_index: usize) -> Vec<String> {
        let cond = &self.conditions[cond_index];
        let mut eff_args: Vec<String> = self
            .effect
            .args
            .iter()
            .map(|a| match a {
                Arg::Var(_) => "".to_string(),
                Arg::Const(c) => c.clone(),
                Arg::FreeVar(_) => "".to_string(),
            })
            .collect();
        for (i, a) in cond.args.iter().enumerate() {
            if let Arg::Var(var_no) = a {
                if let Arg::Const(ref obj) = new_atom.args[i] {
                    eff_args[*var_no] = obj.clone();
                }
            }
        }
        eff_args
    }
}

impl BuildRule for ProjectRule {
    fn validate(&self) {
        assert_eq!(self.conditions.len(), 1, "ProjectRule needs 1 condition");
    }
    fn update_index(&mut self, _new_atom: &Atom, _cond_index: usize) {}
    fn fire(
        &self,
        new_atom: &Atom,
        cond_index: usize,
        enqueue: &mut dyn FnMut(&str, &Vec<String>),
    ) {
        let cond = &self.conditions[cond_index];
        if cond.args.len() != new_atom.args.len() {
            return;
        }
        let eff_args = self.prepare_effect(new_atom, cond_index);
        enqueue(&self.effect.predicate, &eff_args);
    }
    fn conditions(&self) -> &Vec<Atom> {
        &self.conditions
    }
    fn effect(&self) -> &Atom {
        &self.effect
    }
}

pub enum RuleKind {
    Join(JoinRule),
    Product(ProductRule),
    Project(ProjectRule),
}

impl RuleKind {
    pub fn as_rule(&self) -> &dyn BuildRule {
        match self {
            RuleKind::Join(r) => r,
            RuleKind::Product(r) => r,
            RuleKind::Project(r) => r,
        }
    }
    pub fn as_rule_mut(&mut self) -> &mut dyn BuildRule {
        match self {
            RuleKind::Join(r) => r,
            RuleKind::Product(r) => r,
            RuleKind::Project(r) => r,
        }
    }
}

pub fn convert_rules(specs: &[RuleSpec]) -> Vec<RuleKind> {
    eprintln!("DEBUG convert_rules: Converting {} rule specs", specs.len());
    let mut rules = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        eprintln!(
            "  Rule {}: {}({}) :- [{}]",
            i,
            spec.effect.predicate,
            spec.effect.args.join(","),
            spec.conditions
                .iter()
                .map(|c| format!("{}({})", c.predicate, c.args.join(",")))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let (eff, conds) = variables_to_numbers(&spec.effect, &spec.conditions);

        // Debug: show rule 8 in detail
        if i == 8 {
            eprintln!("    DEBUG Rule 8 after variables_to_numbers:");
            eprintln!("      Effect: {}({:?})", eff.predicate, eff.args);
            for (ci, cond) in conds.iter().enumerate() {
                eprintln!(
                    "      Condition {}: {}({:?})",
                    ci, cond.predicate, cond.args
                );
            }
        }
        let rk = match spec.rtype.as_str() {
            "join" => RuleKind::Join(JoinRule::new(eff, conds)),
            "product" => RuleKind::Product(ProductRule::new(eff, conds)),
            "project" => RuleKind::Project(ProjectRule::new(eff, conds)),
            _ => panic!("unknown rule type {}", spec.rtype),
        };
        rk.as_rule().validate();
        rules.push(rk);
    }
    rules
}

// Unifier machinery
#[derive(Clone)]
enum GenNode {
    Leaf {
        matches: Vec<(usize, usize)>,
    },
    Match {
        index: usize,
        matches: Vec<(usize, usize)>,
        map: HashMap<String, Box<GenNode>>,
        next: Box<GenNode>,
    },
}

impl GenNode {
    // helper methods for generator tree
    fn generate(&self, atom: &Atom, result: &mut Vec<(usize, usize)>) {
        match self {
            GenNode::Leaf { matches } => {
                result.extend_from_slice(matches);
            }
            GenNode::Match {
                index,
                matches,
                map,
                next,
            } => {
                result.extend_from_slice(matches);
                if let Arg::Const(ref c) = atom.args[*index] {
                    if let Some(node) = map.get(c) {
                        node.generate(atom, result);
                    }
                }
                next.generate(atom, result);
            }
        }
    }
    fn insert(self, args: &[(usize, String)], value: (usize, usize)) -> GenNode {
        if args.is_empty() {
            return match self {
                GenNode::Leaf { mut matches } => {
                    matches.push(value);
                    GenNode::Leaf { matches }
                }
                GenNode::Match {
                    index,
                    mut matches,
                    map,
                    next,
                } => {
                    matches.push(value);
                    GenNode::Match {
                        index,
                        matches,
                        map,
                        next,
                    }
                }
            };
        }
        match self {
            GenNode::Leaf { matches } => {
                // Important: preserve matches from the old Leaf at the TOP level
                // These are rules with no constants that should match ANY atom
                let preserved_matches = matches.clone();

                // build a chain: for each (arg_index, arg) reversed
                let mut root = GenNode::Leaf {
                    matches: Vec::new(),
                }; // Empty leaf at bottom
                for (arg_index, arg) in args.iter().rev() {
                    let mut map = HashMap::new();
                    map.insert(arg.clone(), Box::new(root));
                    root = GenNode::Match {
                        index: *arg_index,
                        matches: Vec::new(),
                        map,
                        next: Box::new(GenNode::Leaf {
                            matches: Vec::new(),
                        }),
                    };
                }
                // insert value at top AND preserve old matches
                match root {
                    GenNode::Match {
                        index,
                        mut matches,
                        map,
                        next,
                    } => {
                        // Add preserved matches (rules with no constants)
                        matches.extend(preserved_matches);
                        // Add new value (rule with constants)
                        matches.push(value);
                        GenNode::Match {
                            index,
                            matches,
                            map,
                            next,
                        }
                    }
                    _ => unreachable!(),
                }
            }
            GenNode::Match {
                index,
                matches,
                mut map,
                mut next,
            } => {
                let (arg_index, arg) = args[0].clone();
                if index < arg_index {
                    *next = next.insert(args, value);
                    GenNode::Match {
                        index,
                        matches,
                        map,
                        next,
                    }
                } else if index > arg_index {
                    let new_branch = GenNode::Leaf {
                        matches: Vec::new(),
                    }
                    .insert(&args[1..], value);
                    map.insert(arg, Box::new(new_branch));
                    GenNode::Match {
                        index,
                        matches,
                        map,
                        next,
                    }
                } else {
                    let entry = map.remove(&arg);
                    let child = if let Some(node) = entry {
                        node.insert(&args[1..], value)
                    } else {
                        GenNode::Leaf {
                            matches: Vec::new(),
                        }
                        .insert(&args[1..], value)
                    };
                    map.insert(arg, Box::new(child));
                    GenNode::Match {
                        index,
                        matches,
                        map,
                        next,
                    }
                }
            }
        }
    }
}

pub struct Unifier {
    root_by_pred: HashMap<String, GenNode>,
}

impl Unifier {
    pub fn new(rules: &Vec<RuleKind>) -> Self {
        let mut root_by_pred: HashMap<String, GenNode> = HashMap::new();
        for (ri, rk) in rules.iter().enumerate() {
            let conds = rk.as_rule().conditions().clone();
            for (ci, cond) in conds.iter().enumerate() {
                let entry = root_by_pred
                    .remove(&cond.predicate)
                    .unwrap_or(GenNode::Leaf {
                        matches: Vec::new(),
                    });
                // constant arguments to index on
                let mut const_args: Vec<(usize, String)> = Vec::new();
                for (i, a) in cond.args.iter().enumerate() {
                    if let Arg::Const(ref s) = a {
                        const_args.push((i, s.clone()));
                    }
                }

                // Debug: print at conditions
                if cond.predicate == "at" {
                    eprintln!("    DEBUG Unifier: Adding rule {} cond {} predicate {} with {:?} const_args, arity {}",
                             ri, ci, cond.predicate, const_args, cond.args.len());
                }

                let newroot = entry.insert(&const_args, (ri, ci));
                root_by_pred.insert(cond.predicate.clone(), newroot);
            }
        }

        // Debug: show what's in the at predicate index
        if let Some(at_root) = root_by_pred.get("at") {
            eprintln!("DEBUG Unifier built for 'at' predicate:");
            let mut test_matches = Vec::new();
            at_root.generate(
                &Atom {
                    predicate: "at".to_string(),
                    args: vec![
                        Arg::Const("test1".to_string()),
                        Arg::Const("test2".to_string()),
                    ],
                },
                &mut test_matches,
            );
            eprintln!(
                "  Total rules that would match at(test1, test2): {}",
                test_matches.len()
            );
            if test_matches.len() < 20 {
                eprintln!("  Matches: {:?}", test_matches);
            }
        }

        Self { root_by_pred }
    }
    pub fn unify(&self, atom: &Atom) -> Vec<(usize, usize)> {
        let mut res = Vec::new();
        if let Some(root) = self.root_by_pred.get(&atom.predicate) {
            root.generate(atom, &mut res);
        }

        // Debug: if it's an at or item atom, show what we found
        if atom.predicate == "at" || atom.predicate == "item" {
            if res.len() < 3 {
                eprintln!(
                    "    DEBUG Unify {}({:?}): found {} matches: {:?}",
                    atom.predicate,
                    atom.args,
                    res.len(),
                    res
                );
            }
        }

        res
    }
}

pub struct Queue {
    pub queue: Vec<Atom>,
    pub queue_pos: usize,
    pub enqueued: HashSet<(String, Vec<String>)>,
    pub num_pushes: usize,
}

impl Queue {
    pub fn new(atoms: Vec<Atom>) -> Self {
        let mut enq: HashSet<(String, Vec<String>)> = HashSet::new();
        for a in &atoms {
            let args = a
                .args
                .iter()
                .map(|x| match x {
                    Arg::Const(s) => s.clone(),
                    _ => String::new(),
                })
                .collect();
            enq.insert((a.predicate.clone(), args));
        }
        let num_pushes = atoms.len();
        Self {
            queue: atoms,
            queue_pos: 0,
            enqueued: enq,
            num_pushes,
        }
    }
    pub fn has_next(&self) -> bool {
        self.queue_pos < self.queue.len()
    }
    pub fn push(&mut self, predicate: &str, args: &Vec<String>) {
        self.num_pushes += 1;
        let key = (predicate.to_string(), args.clone());
        if !self.enqueued.contains(&key) {
            self.enqueued.insert(key);
            let atom = Atom {
                predicate: predicate.to_string(),
                args: args.iter().map(|s| Arg::Const(s.clone())).collect(),
            };
            self.queue.push(atom);
        }
    }
    pub fn pop(&mut self) -> Atom {
        let a = self.queue[self.queue_pos].clone();
        self.queue_pos += 1;
        a
    }
}

pub fn compute_model(rules: &mut Vec<RuleKind>, facts: &[Atom]) -> Vec<Atom> {
    eprintln!(
        "DEBUG build_model: Starting with {} rules and {} facts",
        rules.len(),
        facts.len()
    );
    let mut queue = Queue::new(facts.to_vec());
    let unifier = Unifier::new(rules);

    let mut iterations = 0;
    while queue.has_next() {
        iterations += 1;
        let next = queue.pop();
        let matches = unifier.unify(&next);

        // Extra debug for item and at predicates
        if next.predicate == "item" || next.predicate == "at" {
            eprintln!(
                "  DEBUG: Processing {}({:?}), {} rule matches",
                next.predicate,
                next.args,
                matches.len()
            );
            for (ri, ci) in &matches {
                eprintln!("    Match: rule {} condition {}", ri, ci);
            }
        }

        if iterations <= 5 || iterations % 100 == 0 {
            eprintln!(
                "  Iteration {}: Processing {}({:?}), {} matches",
                iterations,
                next.predicate,
                next.args,
                matches.len()
            );
        }

        for (ri, ci) in matches {
            let rule = rules.get_mut(ri).unwrap();
            rule.as_rule_mut().update_index(&next, ci);
            let mut push = |pred: &str, args: &Vec<String>| {
                queue.push(pred, args);
            };
            rule.as_rule().fire(&next, ci, &mut push);
        }
    }

    eprintln!(
        "DEBUG build_model: Completed after {} iterations",
        iterations
    );
    eprintln!("  Final queue length: {}", queue.queue.len());
    eprintln!("  Total pushes: {}", queue.num_pushes);

    // Print all model atoms for debugging
    eprintln!("  All {} model atoms:", queue.queue.len());
    for (i, atom) in queue.queue.iter().enumerate() {
        eprintln!("    {}: {}/{:?}", i + 1, atom.predicate, atom.args);
    }

    queue.queue
}

#[cfg(test)]
mod tests {
    use super::*;

    fn const_atom(pred: &str, args: &[&str]) -> Atom {
        Atom {
            predicate: pred.to_string(),
            args: args.iter().map(|s| Arg::Const((*s).to_string())).collect(),
        }
    }

    #[test]
    fn join_rule_produces_effect() {
        // r(X) :- p(X), q(X) with facts p(a), q(a) -> r(a)
        let facts = vec![const_atom("p", &["a"]), const_atom("q", &["a"])];
        let spec = RuleSpec {
            rtype: "join".to_string(),
            effect: SymAtom::new("r", vec!["?x"]),
            conditions: vec![SymAtom::new("p", vec!["?x"]), SymAtom::new("q", vec!["?x"])],
        };
        let mut rules = convert_rules(&[spec]);
        let model = compute_model(&mut rules, &facts);
        assert!(model
            .iter()
            .any(|a| a.predicate == "r" && matches!(&a.args[0], Arg::Const(s) if s=="a")));
    }

    #[test]
    fn product_rule_crosses_bindings() {
        // r(X,Y) :- p(X), q(Y) with p(a), q(b) -> r(a,b)
        let facts = vec![const_atom("p", &["a"]), const_atom("q", &["b"])];
        let spec = RuleSpec {
            rtype: "product".to_string(),
            effect: SymAtom::new("r", vec!["?x", "?y"]),
            conditions: vec![SymAtom::new("p", vec!["?x"]), SymAtom::new("q", vec!["?y"])],
        };
        let mut rules = convert_rules(&[spec]);
        let model = compute_model(&mut rules, &facts);
        assert!(model.iter().any(|a| a.predicate=="r" && matches!((&a.args[0], &a.args[1]), (Arg::Const(x), Arg::Const(y)) if x=="a" && y=="b")));
    }

    #[test]
    fn project_rule_projects() {
        // r(X) :- p(X, ?z) with p(a,b) -> r(a)
        let facts = vec![const_atom("p", &["a", "b"])];
        let spec = RuleSpec {
            rtype: "project".to_string(),
            effect: SymAtom::new("r", vec!["?x"]),
            conditions: vec![SymAtom::new("p", vec!["?x", "?z"])],
        };
        let mut rules = convert_rules(&[spec]);
        let model = compute_model(&mut rules, &facts);
        assert!(model
            .iter()
            .any(|a| a.predicate == "r" && matches!(&a.args[0], Arg::Const(s) if s=="a")));
    }
}
// End of build_model.rs
