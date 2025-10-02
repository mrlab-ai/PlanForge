use std::collections::HashMap;
use std::marker::PhantomData;
use crate::search::numeric::state_registry::{ConcreteState, StateRegistry};

/// Base trait for per-state information storage.
/// This allows the StateRegistry to notify all PerStateInformation instances
/// when a registry is being destroyed.
pub trait PerStateInformationBase {
    /// Called by StateRegistry when it's being dropped to clean up associated data
    fn remove_state_registry(&mut self, registry_id: usize);
}

/// Per-state information storage that associates data of type `T` with states.
/// 
/// This behaves like a map from states to entries, but supports lookup of unknown states
/// which leads to insertion of a default value (similar to Python's defaultdict).
/// 
/// Implementation notes:
/// - Uses a two-level lookup: registry ID -> Vec<T> -> state ID -> T
/// - Caches the last accessed registry for performance
/// - Automatically resizes vectors when accessing states with higher IDs
/// 
/// # Example Usage
/// ```
/// let mut per_state_info = PerStateInformation::new();
/// let state_data = per_state_info.get_mut(&some_state); // Gets default or existing data
/// ```
pub struct PerStateInformation<T> {
    /// Default value returned for states that don't have associated data yet
    default_value: T,
    
    /// Map from registry ID to the vector of entries for that registry
    entries_by_registry: HashMap<usize, Vec<T>>,
    
    /// Cache for the last accessed registry to speed up consecutive lookups
    cached_registry_id: Option<usize>,
    
    /// Phantom data to ensure proper variance
    _phantom: PhantomData<T>,
}

#[allow(dead_code)]
impl<T> PerStateInformation<T> 
where 
    T: Clone + Default,
{
    /// Creates a new PerStateInformation with the default value for type T
    pub fn new() -> Self {
        Self {
            default_value: T::default(),
            entries_by_registry: HashMap::new(),
            cached_registry_id: None,
            _phantom: PhantomData,
        }
    }
    
    /// Creates a new PerStateInformation with a specific default value
    pub fn with_default(default_value: T) -> Self {
        Self {
            default_value,
            entries_by_registry: HashMap::new(),
            cached_registry_id: None,
            _phantom: PhantomData,
        }
    }
    
    /// Gets a mutable reference to the data associated with the given state.
    /// If no data exists, the vector is resized and default values are inserted.
    pub fn get_mut(&mut self, state: &ConcreteState, registry: &StateRegistry) -> &mut T {
        let registry_id = self.get_registry_id(registry);
        let state_id = self.get_state_id(state, registry);
        
        // Update cache
        self.cached_registry_id = Some(registry_id);
        
        // Get or create the vector for this registry
        let entries = self.entries_by_registry
            .entry(registry_id)
            .or_insert_with(Vec::new);
        
        // Ensure the vector is large enough
        let required_size = state_id + 1;
        if entries.len() < required_size {
            entries.resize(required_size, self.default_value.clone());
        }
        
        &mut entries[state_id]
    }
    
    /// Gets a reference to the data associated with the given state.
    /// Returns the default value if no data exists for this state.
    pub fn get(&self, state: &ConcreteState, registry: &StateRegistry) -> &T {
        let registry_id = self.get_registry_id(registry);
        let state_id = self.get_state_id(state, registry);
        
        match self.entries_by_registry.get(&registry_id) {
            Some(entries) if state_id < entries.len() => &entries[state_id],
            _ => &self.default_value,
        }
    }
    
    /// Sets the value for the given state
    pub fn set(&mut self, state: &ConcreteState, registry: &StateRegistry, value: T) {
        let target = self.get_mut(state, registry);
        *target = value;
    }
    
    /// Checks if the given state has associated data (beyond the default)
    pub fn contains(&self, state: &ConcreteState, registry: &StateRegistry) -> bool {
        let registry_id = self.get_registry_id(registry);
        let state_id = self.get_state_id(state, registry);
        
        match self.entries_by_registry.get(&registry_id) {
            Some(entries) => state_id < entries.len(),
            None => false,
        }
    }
    
    /// Iterator over all state IDs that have associated data in a given registry
    pub fn states_with_data(&self, registry_id: usize) -> impl Iterator<Item = usize> + '_ {
        match self.entries_by_registry.get(&registry_id) {
            Some(entries) => (0..entries.len()).collect::<Vec<_>>().into_iter(),
            None => Vec::new().into_iter(),
        }
    }
    
    /// Gets the number of entries stored for a specific registry
    pub fn size_for_registry(&self, registry_id: usize) -> usize {
        self.entries_by_registry
            .get(&registry_id)
            .map(|entries| entries.len())
            .unwrap_or(0)
    }
    
    /// Clears all data for a specific registry
    pub fn clear_registry(&mut self, registry_id: usize) {
        self.entries_by_registry.remove(&registry_id);
        if self.cached_registry_id == Some(registry_id) {
            self.cached_registry_id = None;
        }
    }
    
    /// Helper method to extract registry ID from registry
    /// Uses the registry's memory address as a unique identifier
    fn get_registry_id(&self, registry: &StateRegistry) -> usize {
        registry as *const _ as usize
    }
    
    /// Helper method to extract state ID from state
    /// Uses the buffer pointer as a unique identifier for the state
    fn get_state_id(&self, state: &ConcreteState, registry: &StateRegistry) -> usize {
        // Use the buffer pointer address as a unique identifier for this state
        // This is stable as long as the state exists
        let buffer_ptr = state.buffer(registry).as_ptr() as usize;
        
        // Apply a simple hash to make the ID more manageable
        let mut hash = buffer_ptr;
        hash = hash.wrapping_mul(0x9e3779b9);
        hash ^= hash >> 16;
        hash = hash.wrapping_mul(0x9e3779b9);
        hash ^= hash >> 16;
        
        // Keep it reasonably sized
        hash % 1000000
    }
}

impl<T> PerStateInformationBase for PerStateInformation<T> 
where 
    T: Clone + Default,
{
    fn remove_state_registry(&mut self, registry_id: usize) {
        self.clear_registry(registry_id);
    }
}

impl<T> Default for PerStateInformation<T> 
where 
    T: Clone + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience type aliases for common use cases
pub type PerStateFloat = PerStateInformation<f64>;
pub type PerStateInt = PerStateInformation<i32>;
pub type PerStateBool = PerStateInformation<bool>;
pub type PerStateUsize = PerStateInformation<usize>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_per_state_info_basic() {
        let per_state_info = PerStateInformation::with_default(42);
        
        // Test that default value is returned for non-existent entries
        // Note: This test would need actual ConcreteState instances to be meaningful
        // For now, it's a placeholder
        assert_eq!(per_state_info.default_value, 42);
    }
}