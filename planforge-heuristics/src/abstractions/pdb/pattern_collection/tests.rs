use planforge_sas::numeric_task::VariableIndex;

use super::*;

#[test]
fn collection_normalizes_and_deduplicates_patterns() {
    let collection = PatternCollection::new(vec![
        Pattern {
            regular: vec![
                VariableIndex::from_usize(3),
                VariableIndex::from_usize(1),
                VariableIndex::from_usize(3),
            ],
            numeric: vec![
                VariableIndex::from_usize(5),
                VariableIndex::from_usize(4),
                VariableIndex::from_usize(5),
            ],
        },
        Pattern {
            regular: vec![VariableIndex::from_usize(1), VariableIndex::from_usize(3)],
            numeric: vec![VariableIndex::from_usize(4), VariableIndex::from_usize(5)],
        },
    ]);

    assert_eq!(collection.len(), 1);
    assert_eq!(
        collection.as_slice(),
        &[Pattern {
            regular: vec![VariableIndex::from_usize(1), VariableIndex::from_usize(3)],
            numeric: vec![VariableIndex::from_usize(4), VariableIndex::from_usize(5)],
        }]
    );
}
