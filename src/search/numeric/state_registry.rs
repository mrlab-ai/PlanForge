use std::hash::{ Hash, Hasher };
use std::collections::HashSet;

//TODO: Replace the Vector with the segmented array vector eventually. Since it is unsafe, we should stick to regular vectors for now.
// Placeholder for `PackedStateBin`
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct PackedStateBin(u32);

#[derive(Clone, Copy)]
struct StateID<'a> {
    value: usize,
    state_data_pool: &'a Vec<PackedStateBin>,
}

struct StatePacker {
    num_bins: usize,
}

impl StatePacker {
    fn get_num_bins(&self) -> usize {
        self.num_bins
    }
}

static NUM_BINS: usize = 10; //TODO: Replace with actual number of bins from IntDoublePacker

impl<'a> PartialEq for StateID<'a> {
    fn eq(&self, other: &Self) -> bool {
        let size = NUM_BINS;
        let lhs_data = &self.state_data_pool[self.value..self.value + size];
        let rhs_data = &other.state_data_pool[other.value..other.value + size];
        lhs_data == rhs_data
    }
}

impl<'a> Eq for StateID<'a> {}

impl<'a> Hash for StateID<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let size = NUM_BINS;
        let data = &self.state_data_pool[self.value..self.value + size];
        for bin in data {
            bin.0.hash(state);
        }
    }
}

struct State {}

type StateIDSet<'a> = HashSet<StateID<'a>>;

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
