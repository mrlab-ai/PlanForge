//! Forward chaining over the grounding program.
//!
//! This fixpoint is the translator's hottest loop, so predicate and object
//! names are interned before it starts. Join keys, duplicate checks and rule
//! lookups then work on dense `u32`s instead of strings, an atom's arguments
//! are allocated once and shared by reference count, and the tables a derived
//! atom has to consult are indexed by its predicate rather than hashed.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use tracing::info;

use crate::pddl_to_prolog::{PrologProgram, RuleType};
use crate::symbols::{ObjectId, PredicateId, Symbols};
use crate::tools::cmp_quoted_slice;

/// A ground atom derived by the model builder.
pub struct GroundAtom {
    pub predicate: PredicateId,
    pub args: Rc<[ObjectId]>,
}

/// Every ground atom the program derives, plus the table that reads the ids
/// back as names.
pub struct GroundModel {
    pub symbols: Symbols,
    pub atoms: Vec<GroundAtom>,
}

/// `(rule, condition)`: one place a ground atom can be plugged into.
type Slot = (u32, u32);

/// One condition of a rule, compiled for matching.
///
/// Head variables are numbered by their position in the head, so a binding
/// names the head slot it fills directly.
struct Condition {
    predicate: PredicateId,
    /// `(argument position, head position)` for arguments holding a head
    /// variable. Condition variables absent from the head match anything and
    /// are not recorded.
    bindings: Box<[(usize, usize)]>,
    /// `(argument position, object)` for arguments fixed to a constant.
    constants: Box<[(usize, ObjectId)]>,
}

impl Condition {
    fn bind(&self, args: &[ObjectId], head: &mut [ObjectId]) {
        for &(position, slot) in &self.bindings {
            head[slot] = args[position];
        }
    }
}

/// How a rule combines the matches of its conditions.
enum Body {
    /// One condition: every match fires on its own.
    Project,
    /// Two conditions, joined on the head variables they share. Each side
    /// indexes its matches by the values at those shared positions.
    Join {
        key_positions: [Box<[usize]>; 2],
        matched: [HashMap<Box<[ObjectId]>, Vec<Rc<[ObjectId]>>>; 2],
    },
    /// Any number of conditions whose matches are enumerated as a product.
    Product {
        matched: Vec<Vec<Rc<[ObjectId]>>>,
        /// Conditions that have not matched anything yet: while this is
        /// positive the product is empty.
        unmatched: usize,
    },
}

struct Rule {
    head_predicate: PredicateId,
    /// Head arguments with the constants filled in. Variable slots hold
    /// [`ObjectId::UNBOUND`] and are overwritten by every firing.
    head: Box<[ObjectId]>,
    conditions: Box<[Condition]>,
    body: Body,
}

/// Buffers reused across firings so that a rule that fires millions of times
/// allocates only for the atoms it actually derives.
#[derive(Default)]
struct Scratch {
    head: Vec<ObjectId>,
    key: Vec<ObjectId>,
    cursor: Vec<usize>,
}

impl Rule {
    fn compile(rule: &crate::pddl_to_prolog::Rule, symbols: &mut Symbols) -> Rule {
        let (head_name, head_args) = rule
            .effect
            .split_first()
            .expect("rule head has a predicate");
        let variables: HashMap<&str, usize> = head_args
            .iter()
            .enumerate()
            .filter(|(_, arg)| arg.starts_with('?'))
            .map(|(position, arg)| (arg.as_str(), position))
            .collect();
        let head: Box<[ObjectId]> = head_args
            .iter()
            .map(|arg| {
                if arg.starts_with('?') {
                    ObjectId::UNBOUND
                } else {
                    symbols.object(arg)
                }
            })
            .collect();

        let mut bound = vec![false; head.len()];
        let conditions: Box<[Condition]> = rule
            .conditions
            .iter()
            .map(|condition| {
                let (name, args) = condition.split_first().expect("condition has a predicate");
                let mut bindings = Vec::new();
                let mut constants = Vec::new();
                for (position, arg) in args.iter().enumerate() {
                    match variables.get(arg.as_str()) {
                        Some(&slot) => {
                            bound[slot] = true;
                            bindings.push((position, slot));
                        }
                        // A variable the head does not mention matches anything.
                        None if arg.starts_with('?') => {}
                        None => constants.push((position, symbols.object(arg))),
                    }
                }
                Condition {
                    predicate: symbols.predicate(name),
                    bindings: bindings.into(),
                    constants: constants.into(),
                }
            })
            .collect();

        // Every rule kind consults all of its conditions when it fires, so a
        // head variable bound by any of them is bound by the time the derived
        // atom is enqueued. `PrologProgram::remove_free_effect_variables`
        // establishes this; without it `UNBOUND` would escape into the model.
        for (slot, bound) in bound.iter().enumerate() {
            assert!(
                *bound || head[slot] != ObjectId::UNBOUND,
                "head variable {} of rule {:?} is bound by no condition",
                head_args[slot],
                rule.effect
            );
        }

        let body = match rule.rule_type.as_ref().unwrap_or(&RuleType::Join) {
            RuleType::Project => {
                assert_eq!(conditions.len(), 1, "project rule needs one condition");
                Body::Project
            }
            RuleType::Join => {
                assert_eq!(conditions.len(), 2, "join rule needs two conditions");
                Body::Join {
                    key_positions: join_key_positions(&conditions[0], &conditions[1]),
                    matched: Default::default(),
                }
            }
            RuleType::Product => Body::Product {
                matched: vec![Vec::new(); conditions.len()],
                unmatched: conditions.len(),
            },
        };

        Rule {
            head_predicate: symbols.predicate(head_name),
            head,
            conditions,
            body,
        }
    }

    /// Records a match so that later firings of the *other* conditions can
    /// combine with it.
    fn record(&mut self, index: usize, args: &Rc<[ObjectId]>, key: &mut Vec<ObjectId>) {
        match &mut self.body {
            Body::Project => {}
            Body::Join {
                key_positions,
                matched,
            } => {
                fill_key(key, &key_positions[index], args);
                matched[index]
                    .entry(key.as_slice().into())
                    .or_default()
                    .push(Rc::clone(args));
            }
            Body::Product { matched, unmatched } => {
                if matched[index].is_empty() {
                    *unmatched -= 1;
                }
                matched[index].push(Rc::clone(args));
            }
        }
    }

    /// Enqueues the head for every way the rest of the body can be satisfied
    /// alongside `args` matching condition `index`.
    fn fire(
        &self,
        index: usize,
        args: &[ObjectId],
        scratch: &mut Scratch,
        enqueue: &mut dyn FnMut(PredicateId, &[ObjectId]),
    ) {
        let Scratch { head, key, cursor } = scratch;
        head.clear();
        head.extend_from_slice(&self.head);
        self.conditions[index].bind(args, head);

        match &self.body {
            Body::Project => enqueue(self.head_predicate, head),
            Body::Join {
                key_positions,
                matched,
            } => {
                fill_key(key, &key_positions[index], args);
                let other = 1 - index;
                let Some(partners) = matched[other].get(key.as_slice()) else {
                    return;
                };
                // The two conditions agree on exactly the head slots the key
                // covers, so binding the partner cannot invalidate the slots
                // already written and no reset is needed between partners.
                for partner in partners {
                    self.conditions[other].bind(partner, head);
                    enqueue(self.head_predicate, head);
                }
            }
            Body::Product { matched, unmatched } => {
                if *unmatched > 0 {
                    return;
                }
                cursor.clear();
                cursor.resize(self.conditions.len(), 0);
                loop {
                    head.clear();
                    head.extend_from_slice(&self.head);
                    self.conditions[index].bind(args, head);
                    for (other, condition) in self.conditions.iter().enumerate() {
                        if other != index {
                            condition.bind(&matched[other][cursor[other]], head);
                        }
                    }
                    enqueue(self.head_predicate, head);
                    if !advance(cursor, index, matched) {
                        return;
                    }
                }
            }
        }
    }
}

fn fill_key(key: &mut Vec<ObjectId>, positions: &[usize], args: &[ObjectId]) {
    key.clear();
    key.extend(positions.iter().map(|&position| args[position]));
}

/// Positions, in each condition, of the head variables both conditions bind.
/// Both sides list them in the same order, so equal keys mean equal bindings.
fn join_key_positions(left: &Condition, right: &Condition) -> [Box<[usize]>; 2] {
    let right_slots: HashMap<usize, usize> = right
        .bindings
        .iter()
        .map(|&(position, slot)| (slot, position))
        .collect();
    let mut shared: Vec<(usize, usize, usize)> = left
        .bindings
        .iter()
        .filter_map(|&(position, slot)| Some((slot, position, *right_slots.get(&slot)?)))
        .collect();
    shared.sort_unstable();
    [
        shared.iter().map(|&(_, left, _)| left).collect(),
        shared.iter().map(|&(_, _, right)| right).collect(),
    ]
}

/// Advances the product odometer, last condition fastest. Returns false once
/// every combination has been visited.
fn advance(cursor: &mut [usize], skip: usize, matched: &[Vec<Rc<[ObjectId]>>]) -> bool {
    for position in (0..cursor.len()).rev() {
        if position == skip {
            continue;
        }
        cursor[position] += 1;
        if cursor[position] < matched[position].len() {
            return true;
        }
        cursor[position] = 0;
    }
    false
}

/// Trie over the constant arguments of the rule conditions sharing a
/// predicate: given a ground atom it yields every slot the atom can fill.
enum Generator {
    Leaf(Box<[Slot]>),
    Branch(Branch),
}

struct Branch {
    /// Argument position this node discriminates on.
    position: usize,
    /// Slots whose conditions have no constant left to check.
    matches: Box<[Slot]>,
    by_object: HashMap<ObjectId, Generator>,
    /// Slots whose next constant sits at a later position.
    rest: Box<Generator>,
}

impl Generator {
    fn build(entries: Vec<(&[(usize, ObjectId)], Slot)>) -> Generator {
        let next_position = entries
            .iter()
            .filter_map(|(constants, _)| constants.first())
            .map(|&(position, _)| position)
            .min();
        let Some(position) = next_position else {
            return Generator::Leaf(entries.into_iter().map(|(_, slot)| slot).collect());
        };

        let mut matches = Vec::new();
        let mut by_object: HashMap<ObjectId, Vec<(&[(usize, ObjectId)], Slot)>> = HashMap::new();
        let mut rest = Vec::new();
        for (constants, slot) in entries {
            match constants.first() {
                None => matches.push(slot),
                Some(&(at, object)) if at == position => by_object
                    .entry(object)
                    .or_default()
                    .push((&constants[1..], slot)),
                Some(_) => rest.push((constants, slot)),
            }
        }
        Generator::Branch(Branch {
            position,
            matches: matches.into(),
            by_object: by_object
                .into_iter()
                .map(|(object, entries)| (object, Generator::build(entries)))
                .collect(),
            rest: Box::new(Generator::build(rest)),
        })
    }

    fn collect_slots(&self, args: &[ObjectId], out: &mut Vec<Slot>) {
        match self {
            Generator::Leaf(matches) => out.extend_from_slice(matches),
            Generator::Branch(branch) => {
                out.extend_from_slice(&branch.matches);
                if let Some(next) = args
                    .get(branch.position)
                    .and_then(|object| branch.by_object.get(object))
                {
                    next.collect_slots(args, out);
                }
                branch.rest.collect_slots(args, out);
            }
        }
    }
}

/// Maps a derived atom to the rule conditions it matches, by predicate.
struct Unifier {
    by_predicate: Vec<Generator>,
}

impl Unifier {
    fn new(rules: &[Rule], predicate_count: usize) -> Self {
        let mut entries: Vec<Vec<(&[(usize, ObjectId)], Slot)>> = vec![Vec::new(); predicate_count];
        for (rule_index, rule) in rules.iter().enumerate() {
            for (condition_index, condition) in rule.conditions.iter().enumerate() {
                entries[condition.predicate.index()].push((
                    &condition.constants[..],
                    (rule_index as u32, condition_index as u32),
                ));
            }
        }
        Unifier {
            by_predicate: entries.into_iter().map(Generator::build).collect(),
        }
    }

    fn slots(&self, predicate: PredicateId, args: &[ObjectId], out: &mut Vec<Slot>) {
        out.clear();
        self.by_predicate[predicate.index()].collect_slots(args, out);
    }
}

/// The derived atoms, in derivation order, with duplicates suppressed.
struct Queue {
    atoms: Vec<GroundAtom>,
    position: usize,
    /// Argument tuples already derived, per predicate. Sharing the `Rc` with
    /// the queue keeps the deduplication index free of extra allocations.
    enqueued: Vec<HashSet<Rc<[ObjectId]>>>,
    pushes: usize,
}

impl Queue {
    fn new(predicate_count: usize) -> Self {
        Queue {
            atoms: Vec::new(),
            position: 0,
            enqueued: vec![HashSet::new(); predicate_count],
            pushes: 0,
        }
    }

    fn push(&mut self, predicate: PredicateId, args: &[ObjectId]) {
        self.pushes += 1;
        let derived = &mut self.enqueued[predicate.index()];
        if derived.contains(args) {
            return;
        }
        let args: Rc<[ObjectId]> = Rc::from(args);
        derived.insert(Rc::clone(&args));
        self.atoms.push(GroundAtom { predicate, args });
    }
}

/// Computes the least model of `prog` by forward chaining.
pub fn compute_model(prog: &PrologProgram) -> GroundModel {
    let mut symbols = Symbols::default();

    // The seed order decides the order of everything the model feeds, down to
    // the SAS variable order, so it is fixed before any interning happens.
    let mut facts: Vec<&[String]> = prog.facts.iter().map(Vec::as_slice).collect();
    facts.sort_unstable_by(|left, right| cmp_quoted_slice(left, right));

    let mut rules: Vec<Rule> = prog
        .rules
        .iter()
        .map(|rule| Rule::compile(rule, &mut symbols))
        .collect();
    let interned_facts: Vec<(PredicateId, Vec<ObjectId>)> = facts
        .iter()
        .map(|fact| {
            let (predicate, args) = fact.split_first().expect("fact has a predicate");
            let predicate = symbols.predicate(predicate);
            (predicate, args.iter().map(|a| symbols.object(a)).collect())
        })
        .collect();

    // Interning is complete, so the per-predicate tables can be sized once.
    let predicate_count = symbols.predicate_count();
    let auxiliary: Vec<bool> = symbols
        .predicates()
        .map(|(_, name)| name.contains('$'))
        .collect();
    let unifier = Unifier::new(&rules, predicate_count);

    let mut queue = Queue::new(predicate_count);
    for (predicate, args) in &interned_facts {
        queue.push(*predicate, args);
    }

    info!("Generated {} rules.", rules.len());

    let mut scratch = Scratch::default();
    let mut slots: Vec<Slot> = Vec::new();
    let mut relevant_atoms = 0;
    let mut auxiliary_atoms = 0;
    while queue.position < queue.atoms.len() {
        let predicate = queue.atoms[queue.position].predicate;
        // The queue grows while the atom fires, so its arguments are held by
        // reference count rather than borrowed out of it.
        let args = Rc::clone(&queue.atoms[queue.position].args);
        queue.position += 1;
        if auxiliary[predicate.index()] {
            auxiliary_atoms += 1;
        } else {
            relevant_atoms += 1;
        }

        unifier.slots(predicate, &args, &mut slots);
        for &(rule_index, condition_index) in &slots {
            let rule = &mut rules[rule_index as usize];
            let condition_index = condition_index as usize;
            rule.record(condition_index, &args, &mut scratch.key);
            rule.fire(
                condition_index,
                &args,
                &mut scratch,
                &mut |predicate, args| queue.push(predicate, args),
            );
        }
    }

    info!("{relevant_atoms} relevant atoms");
    info!("{auxiliary_atoms} auxiliary atoms");
    info!("{} final queue length", queue.atoms.len());
    info!("{} total queue pushes", queue.pushes);

    GroundModel {
        symbols,
        atoms: queue.atoms,
    }
}
