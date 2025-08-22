use std::collections::HashMap;

use crate::search::numeric::state_registry::{self, ConcreteState};

use crate::search::numeric::state_registry::StateID;
use crate::search::numeric::state_registry::StateRegistry;

type RegistryId = usize; //TODO: Consider using smaller numbers
pub struct StateInfo<'a, Entry: Default + Clone> {
    default_value: Entry,
    entries_by_registry: HashMap<RegistryId, Vec<Entry>>,
    cached_registry: Option<RegistryId>,
    cached_entries: Option<&'a Vec<Entry>>,
    subscribed_registries: Vec<RegistryId>,
}

impl<'a, Entry: Default + Clone> StateInfo<'a, Entry> {
    pub fn new() -> StateInfo<'a, Entry> {
        Self {
            default_value: Entry::default(),
            entries_by_registry: HashMap::new(),
            cached_registry: None,
            cached_entries: None,
            subscribed_registries: Vec::new(),
        }
    }

    pub fn with_default(default_value: Entry) -> Self {
        Self {
            default_value,
            entries_by_registry: HashMap::new(),
            cached_registry: None,
            cached_entries: None,
            subscribed_registries: Vec::new(),
        }
    }

    fn get_entries(&mut self, registry: &&StateRegistry) -> Option<&Vec<Entry>> {
        if self.cached_registry != Some(registry.id()) {
            self.cached_registry = Some(registry.id());
            let entries = self.entries_by_registry.get(&registry.id());
            return entries;
        } else {
            return self.cached_entries;
        }
    }

    pub fn get_mut(&mut self, state: &ConcreteState) -> &mut Entry {
        todo!()
    }

    pub fn get(&self, state: &ConcreteState) -> &Entry {
        todo!()
    }

    pub fn remove_state_registry(&mut self, registry: &StateRegistry) {
        todo!()
    }

    pub fn iter<'b>(&'b self, registry: &'b StateRegistry) -> impl Iterator<Item = StateID> + 'b {
        // Assuming StateRegistry has a method `states()` returning an iterator over StateID
        std::iter::empty()
    }
}
