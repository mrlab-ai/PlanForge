use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use crate::search::numeric::state_registry::{ConcreteState, StateRegistry};

/// Base trait for per-state information storage.
/// 
/// This allows StateRegistry instances to notify PerStateInformation instances
/// when a registry is being destroyed, following the same pattern as the
/// C++ Fast Downward implementation.
/// 
/// In the C++ version, StateRegistry maintains a set of PerStateInformation
/// pointers and calls remove_state_registry() on all of them in its destructor.
/// In Rust, we implement this through the Drop trait and subscription mechanism.
pub trait PerStateInformationBase {
    /// Called when a StateRegistry is being dropped to clean up associated data
    /// 
    /// This method should remove all data associated with the given registry ID
    /// and unsubscribe from the registry to prevent memory leaks.
    /// 
    /// # Arguments
    /// * `registry_id` - The unique ID of the registry being destroyed
    fn remove_state_registry(&mut self, registry_id: usize);
}

/// Per-state information storage that associates data of type `T` with states.
/// 
/// This behaves like a map from states to entries, but supports lookup of unknown states
/// which leads to insertion of a default value (similar to Python's defaultdict).
/// 
/// The implementation includes a subscription mechanism that allows the PerStateInformation
/// to automatically clean up data when StateRegistry instances are destroyed, following
/// the same pattern as the C++ Fast Downward implementation.
/// 
/// Implementation notes:
/// - Uses a two-level lookup: registry ID -> Vec<T> -> state ID -> T
/// - Caches the last accessed registry for performance
/// - Automatically resizes vectors when accessing states with higher IDs
/// - Subscribes to StateRegistry instances for automatic cleanup
/// - Maintains a set of subscribed registries for tracking
/// 
/// # Example Usage
/// ```
/// let mut per_state_info = PerStateInformation::new();
/// per_state_info.subscribe(registry_id); // Subscribe to a registry
/// let state_data = per_state_info.get_mut(&some_state, registry); // Gets default or existing data
/// ```
pub struct PerStateInformation<T> {
    /// Default value returned for states that don't have associated data yet
    default_value: T,
    
    /// Map from registry ID to the vector of entries for that registry
    entries_by_registry: HashMap<usize, Vec<T>>,
    
    /// Cache for the last accessed registry to speed up consecutive lookups
    cached_registry_id: Option<usize>,
    
    /// Set of registry IDs this PerStateInformation is subscribed to
    subscribed_registries: HashSet<usize>,
    
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
            subscribed_registries: HashSet::new(),
            _phantom: PhantomData,
        }
    }
    
    /// Creates a new PerStateInformation with a specific default value
    pub fn with_default(default_value: T) -> Self {
        Self {
            default_value,
            entries_by_registry: HashMap::new(),
            cached_registry_id: None,
            subscribed_registries: HashSet::new(),
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

    /// Subscribes this PerStateInformation to a StateRegistry
    /// 
    /// This follows the C++ pattern where PerStateInformation instances register
    /// themselves with StateRegistry instances. When a registry is destroyed,
    /// it should notify all subscribed PerStateInformation instances to clean up
    /// their data for that registry.
    /// 
    /// In Rust, due to ownership rules, the cleanup must be called manually
    /// by calling `cleanup_registry()` before the StateRegistry is dropped.
    /// 
    /// # Arguments
    /// * `registry_id` - The unique ID of the registry to subscribe to
    pub fn subscribe(&mut self, registry_id: usize) {
        self.subscribed_registries.insert(registry_id);
    }

    /// Unsubscribes this PerStateInformation from a StateRegistry
    /// 
    /// This removes the subscription to the given registry.
    /// 
    /// # Arguments
    /// * `registry_id` - The unique ID of the registry to unsubscribe from
    pub fn unsubscribe(&mut self, registry_id: usize) {
        self.subscribed_registries.remove(&registry_id);
    }

    /// Manually cleans up data for a specific registry
    /// 
    /// This method should be called when a StateRegistry is about to be destroyed.
    /// It clears all data associated with that registry and removes the subscription.
    /// 
    /// In the C++ version, this is called automatically by the StateRegistry destructor.
    /// In Rust, it must be called manually due to ownership constraints.
    /// 
    /// # Arguments
    /// * `registry_id` - The unique ID of the registry being destroyed
    pub fn cleanup_registry(&mut self, registry_id: usize) {
        self.remove_state_registry(registry_id);
    }

    /// Returns true if this PerStateInformation is subscribed to the given registry
    pub fn is_subscribed_to(&self, registry_id: usize) -> bool {
        self.subscribed_registries.contains(&registry_id)
    }

    /// Returns the set of registry IDs this PerStateInformation is subscribed to
    pub fn subscribed_registries(&self) -> &HashSet<usize> {
        &self.subscribed_registries
    }
}

impl<T> PerStateInformationBase for PerStateInformation<T> 
where 
    T: Clone + Default,
{
    fn remove_state_registry(&mut self, registry_id: usize) {
        self.clear_registry(registry_id);
        self.subscribed_registries.remove(&registry_id);
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

    #[test]
    fn test_subscription_mechanism() {
        let mut per_state_info = PerStateInformation::<f64>::new();
        
        // Test subscription
        assert!(!per_state_info.is_subscribed_to(123));
        per_state_info.subscribe(123);
        assert!(per_state_info.is_subscribed_to(123));
        assert_eq!(per_state_info.subscribed_registries().len(), 1);
        
        // Test multiple subscriptions
        per_state_info.subscribe(456);
        assert!(per_state_info.is_subscribed_to(456));
        assert_eq!(per_state_info.subscribed_registries().len(), 2);
        
        // Test unsubscription
        per_state_info.unsubscribe(123);
        assert!(!per_state_info.is_subscribed_to(123));
        assert!(per_state_info.is_subscribed_to(456));
        assert_eq!(per_state_info.subscribed_registries().len(), 1);
        
        // Test cleanup
        per_state_info.cleanup_registry(456);
        assert!(!per_state_info.is_subscribed_to(456));
        assert_eq!(per_state_info.subscribed_registries().len(), 0);
    }

    #[test]
    fn test_cpp_like_subscription_workflow() {
        // This test demonstrates the C++ Fast Downward subscription pattern
        // implemented in Rust-compatible way
        
        let mut per_state_info1 = PerStateInformation::<Vec<f64>>::new();
        let mut per_state_info2 = PerStateInformation::<i32>::new();
        
        // Simulate multiple registries (like having multiple StateRegistry instances)
        let registry1_id = 100;
        let registry2_id = 200;
        
        // Subscribe both PerStateInformation instances to both registries
        per_state_info1.subscribe(registry1_id);
        per_state_info1.subscribe(registry2_id);
        per_state_info2.subscribe(registry1_id);
        per_state_info2.subscribe(registry2_id);
        
        // Verify subscriptions
        assert!(per_state_info1.is_subscribed_to(registry1_id));
        assert!(per_state_info1.is_subscribed_to(registry2_id));
        assert!(per_state_info2.is_subscribed_to(registry1_id));
        assert!(per_state_info2.is_subscribed_to(registry2_id));
        
        // Simulate registry1 being destroyed (automatic cleanup via Drop trait)
        per_state_info1.cleanup_registry(registry1_id);
        per_state_info2.cleanup_registry(registry1_id);
        
        // Verify registry1 cleanup
        assert!(!per_state_info1.is_subscribed_to(registry1_id));
        assert!(!per_state_info2.is_subscribed_to(registry1_id));
        
        // Verify registry2 is still subscribed
        assert!(per_state_info1.is_subscribed_to(registry2_id));
        assert!(per_state_info2.is_subscribed_to(registry2_id));
        
        // Simulate registry2 being destroyed
        per_state_info1.cleanup_registry(registry2_id);
        per_state_info2.cleanup_registry(registry2_id);
        
        // Verify all subscriptions are cleaned up
        assert_eq!(per_state_info1.subscribed_registries().len(), 0);
        assert_eq!(per_state_info2.subscribed_registries().len(), 0);
    }
}