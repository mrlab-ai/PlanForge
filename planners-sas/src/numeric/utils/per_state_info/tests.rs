use crate::numeric::state_registry::ConcreteState;

use super::*;

#[test]
fn test_per_state_info_basic() {
    let per_state_info = PerStateInformation::with_default(42);

    // Test that default value is returned for non-existent entries.
    // TODO: This test would need actual ConcreteState instances to be meaningful.
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

#[test]
fn test_state_id_extraction() {
    // This test would require actual ConcreteState instances to be meaningful
    // For now, we verify that the state ID is directly extracted without hashing

    // Create a mock state with a known ID
    let state = ConcreteState::new(42); // pool_offset = 42
    assert_eq!(state.get_id(), 42); // Should return the ID directly, not hashed

    let state2 = ConcreteState::new(100);
    assert_eq!(state2.get_id(), 100);

    // Verify they're different
    assert_ne!(state.get_id(), state2.get_id());
}
