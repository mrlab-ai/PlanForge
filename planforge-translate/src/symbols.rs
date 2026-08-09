//! Interning for the names that grounding manipulates.
//!
//! The model builder compares, hashes and copies atom arguments in its
//! innermost loop. Interning every name once turns all of that into work on
//! dense `u32`s and lets consumers index tables by symbol instead of hashing
//! strings.

use std::collections::HashMap;
use std::rc::Rc;

/// Interned name of a predicate, i.e. argument 0 of a Prolog atom.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct PredicateId(u32);

/// Interned name of a constant, i.e. any non-variable argument of a Prolog atom.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ObjectId(u32);

impl PredicateId {
    /// Index of this predicate in a table sized by [`Symbols::predicate_count`].
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl ObjectId {
    /// Filler for head arguments that a rule body is guaranteed to overwrite.
    ///
    /// [`crate::build_model`] checks when a rule is compiled that every head
    /// variable is bound by some condition, so this value never reaches a
    /// derived atom.
    pub const UNBOUND: ObjectId = ObjectId(u32::MAX);
}

/// A string table handing out dense ids.
#[derive(Default)]
struct Interner {
    names: Vec<Rc<str>>,
    ids: HashMap<Rc<str>, u32>,
}

impl Interner {
    fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = u32::try_from(self.names.len()).expect("symbol table exceeds u32::MAX entries");
        let name: Rc<str> = Rc::from(name);
        self.names.push(Rc::clone(&name));
        self.ids.insert(name, id);
        id
    }
}

/// The symbol table shared by the grounder and everything reading its model.
#[derive(Default)]
pub struct Symbols {
    predicates: Interner,
    objects: Interner,
}

impl Symbols {
    pub fn predicate(&mut self, name: &str) -> PredicateId {
        PredicateId(self.predicates.intern(name))
    }

    pub fn object(&mut self, name: &str) -> ObjectId {
        ObjectId(self.objects.intern(name))
    }

    pub fn predicate_name(&self, id: PredicateId) -> &str {
        &self.predicates.names[id.index()]
    }

    pub fn object_name(&self, id: ObjectId) -> &str {
        &self.objects.names[id.0 as usize]
    }

    /// Number of distinct predicates, i.e. the size of a table indexed by
    /// [`PredicateId::index`].
    pub fn predicate_count(&self) -> usize {
        self.predicates.names.len()
    }

    /// Every predicate, in [`PredicateId::index`] order.
    pub fn predicates(&self) -> impl Iterator<Item = (PredicateId, &str)> {
        self.predicates
            .names
            .iter()
            .enumerate()
            .map(|(index, name)| (PredicateId(index as u32), name.as_ref()))
    }
}
