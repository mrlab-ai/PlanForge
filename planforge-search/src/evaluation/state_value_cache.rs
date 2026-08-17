use planforge_sas::state_registry::StateID;

#[derive(Default)]
pub(crate) struct StateValueCache {
    values: Vec<f64>,
}

impl StateValueCache {
    pub(crate) fn get(&self, state_id: StateID) -> Option<f64> {
        self.values
            .get(state_id)
            .copied()
            .filter(|value| !value.is_nan())
    }

    pub(crate) fn insert(&mut self, state_id: StateID, value: f64) {
        if self.values.len() <= state_id {
            self.values.resize(state_id + 1, f64::NAN);
        }
        self.values[state_id] = value;
    }
}
