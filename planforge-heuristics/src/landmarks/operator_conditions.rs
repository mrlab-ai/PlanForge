//! The effects of each operator on each numeric condition, for numeric LM-cut.
//!
//! An entry is the constant effect the operator has on the condition, and the
//! second-order simple effect (SOSE) it supports there.
//!
//! Most pairs carry nothing -- an operator changes few of the variables a
//! condition reads -- so the table offers two layouts: dense, one entry per
//! pair with constant-time access, and sparse, the entries that were written,
//! found by binary search. Dense is the default; sparse trades a logarithmic
//! lookup for memory on tasks with many operators and conditions, where the
//! dense table grows with their product (3.5 GB on a 15x15 minecraft task).

use planforge_sas::numeric_task::OperatorIndex;
use planforge_sas::utils::linear_effects::LinearExpression;

use planforge_search::config::option_enum;

/// How LM-cut stores its table of operator conditions (`condition_table=`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ConditionTableLayout {
    /// One entry per operator and condition.
    #[default]
    Dense,
    /// Only the entries that were written, per operator sorted by condition.
    Sparse,
}

option_enum!("condition_table", ConditionTableLayout {
    Dense => "dense",
    Sparse => "sparse",
});

/// What one operator does to one condition.
#[derive(Debug, Clone, Default)]
pub(super) struct OperatorConditionEval {
    pub(super) simple_effect: Option<f64>,
    pub(super) has_sose: bool,
    pub(super) composite_expression: Option<LinearExpression>,
    pub(super) has_upper_bound: bool,
    pub(super) upper_bound: f64,
}

/// The entry of every pair without one of its own.
static NO_EFFECT: OperatorConditionEval = OperatorConditionEval {
    simple_effect: None,
    has_sose: false,
    composite_expression: None,
    has_upper_bound: false,
    upper_bound: 0.0,
};

/// The [`OperatorConditionEval`] of every operator and condition. Entries have
/// a global index, so data kept per entry (such as the composite-expression
/// memo) can live in vectors aligned with [`Self::len`].
#[derive(Debug, Default)]
pub(super) struct OperatorConditionTable {
    entries: Vec<OperatorConditionEval>,
    layout: Layout,
}

#[derive(Debug)]
enum Layout {
    Dense {
        num_conditions: usize,
    },
    /// Per operator, `(condition, entry index)` sorted by condition.
    Sparse {
        rows: Vec<Vec<(u32, u32)>>,
    },
}

impl Default for Layout {
    fn default() -> Self {
        Self::Dense { num_conditions: 0 }
    }
}

impl OperatorConditionTable {
    pub(super) fn new(
        layout: ConditionTableLayout,
        num_operators: usize,
        num_conditions: usize,
    ) -> Self {
        match layout {
            ConditionTableLayout::Dense => Self {
                entries: vec![
                    OperatorConditionEval::default();
                    num_operators
                        .checked_mul(num_conditions)
                        .expect("the dense LM-cut condition table size overflows")
                ],
                layout: Layout::Dense { num_conditions },
            },
            ConditionTableLayout::Sparse => Self {
                entries: Vec::new(),
                layout: Layout::Sparse {
                    rows: vec![Vec::new(); num_operators],
                },
            },
        }
    }

    /// The number of entries, which entry indices range below.
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    /// The index of the entry of `operator` and `condition`; `None` for a
    /// sparse table without one, whose entry is the default.
    #[inline]
    pub(super) fn index(&self, operator: OperatorIndex, condition: usize) -> Option<usize> {
        match &self.layout {
            Layout::Dense { num_conditions } => {
                debug_assert!(condition < *num_conditions);
                Some(operator.index() * num_conditions + condition)
            }
            Layout::Sparse { rows } => {
                let row = &rows[operator.index()];
                let condition = condition_id(condition);
                row.binary_search_by_key(&condition, |&(id, _)| id)
                    .ok()
                    .map(|position| row[position].1 as usize)
            }
        }
    }

    #[inline]
    pub(super) fn get(&self, operator: OperatorIndex, condition: usize) -> &OperatorConditionEval {
        self.index(operator, condition)
            .map_or(&NO_EFFECT, |index| &self.entries[index])
    }

    /// The entry of `operator` and `condition`, created in a sparse table
    /// that has none yet.
    pub(super) fn get_mut(
        &mut self,
        operator: OperatorIndex,
        condition: usize,
    ) -> &mut OperatorConditionEval {
        let index = match &mut self.layout {
            Layout::Dense { num_conditions } => operator.index() * *num_conditions + condition,
            Layout::Sparse { rows } => {
                let row = &mut rows[operator.index()];
                let condition = condition_id(condition);
                match row.binary_search_by_key(&condition, |&(id, _)| id) {
                    Ok(position) => row[position].1 as usize,
                    Err(position) => {
                        let index = self.entries.len();
                        row.insert(
                            position,
                            (
                                condition,
                                u32::try_from(index).expect(
                                    "the sparse LM-cut condition table exceeds u32 entries",
                                ),
                            ),
                        );
                        self.entries.push(OperatorConditionEval::default());
                        index
                    }
                }
            }
        };
        &mut self.entries[index]
    }
}

fn condition_id(condition: usize) -> u32 {
    u32::try_from(condition).expect("LM-cut condition ids fit in u32")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_layouts_answer_alike() {
        for layout in [ConditionTableLayout::Dense, ConditionTableLayout::Sparse] {
            let mut table = OperatorConditionTable::new(layout, 3, 4);
            table.get_mut(OperatorIndex::new(2), 1).simple_effect = Some(5.0);
            table.get_mut(OperatorIndex::new(0), 3).has_sose = true;
            assert_eq!(table.get(OperatorIndex::new(2), 1).simple_effect, Some(5.0));
            assert!(table.get(OperatorIndex::new(0), 3).has_sose);
            assert_eq!(table.get(OperatorIndex::new(1), 1).simple_effect, None);
            let written = table.index(OperatorIndex::new(2), 1).unwrap();
            assert!(written < table.len());
            assert_eq!(
                table.index(OperatorIndex::new(1), 2).is_some(),
                layout == ConditionTableLayout::Dense
            );
        }
        let sparse = {
            let mut table = OperatorConditionTable::new(ConditionTableLayout::Sparse, 1000, 1000);
            table.get_mut(OperatorIndex::new(7), 9);
            table
        };
        assert_eq!(sparse.len(), 1);
    }
}
