use super::*;

#[test]
fn collection_normalizes_and_deduplicates_patterns() {
    let collection = PatternCollection::new(vec![
        Pattern {
            regular: vec![3, 1, 3],
            numeric: vec![5, 4, 5],
        },
        Pattern {
            regular: vec![1, 3],
            numeric: vec![4, 5],
        },
    ]);

    assert_eq!(collection.len(), 1);
    assert_eq!(
        collection.as_slice(),
        &[Pattern {
            regular: vec![1, 3],
            numeric: vec![4, 5],
        }]
    );
}
