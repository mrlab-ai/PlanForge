use std::collections::HashSet;
use std::hash::{ BuildHasherDefault, Hasher, Hash };
use std::fmt;

struct State {}

struct PackedStateBin;

struct StateIDSemanticHash<'a> {
    state_data_pool: &'a Vec<PackedStateBin>,
}
struct StatePacker { num_bins: usize }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateID {
    value: usize,
}

impl StateID {
    pub const NO_STATE: StateID = StateID { value: usize::MAX };

    pub(crate) fn new(value: usize) -> Self {
        StateID { value }
    }
}

impl Hash for StateID {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

// The `Display` trait allows `StateID` to be formatted for printing.
// This is the equivalent of the C++ `operator<<`.
impl fmt::Display for StateID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

struct StateRegistry {
    state_data_pool: Vec<State>,
    numeric_constants: Vec<f64>,
    numeric_indices: Vec<usize>,
    registered_states: HashSet<usize>,
}

impl StateRegistry {
    pub fn new() -> Self {
        StateRegistry {
            state_data_pool: Vec::new(),
            numeric_constants: Vec::new(),
            numeric_indices: Vec::new(),
            registered_states: HashSet::new(),
        }
    }

    pub fn insert_state(&mut self, state: State) -> &State {
        let is_new_entry = self.registered_states.insert(self.state_data_pool.len());
        if !is_new_entry {
            self.state_data_pool.pop();
        }
        self.state_data_pool.push(state);
        self.state_data_pool.last().unwrap()
    }

    pub fn lookup_state(&self, index: usize) -> Option<&State> {
        self.state_data_pool.get(index)
    }
}
