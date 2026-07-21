//! Split-selection policies from Schindler, Speck, and Helmert (ICAPS 2026).
//!
//! The artifact constructs one Cartesian abstraction for the original task,
//! replays the first flawed abstract plan, and selects a split randomly or by
//! the number of values excluded from the desired child. The generator owns
//! replay and refinement; this module keeps the artifact-specific policy
//! explicit so native collection generation does not silently inherit it.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Icaps26SplitSelection {
    Random,
    MinUnwanted,
    MaxUnwanted,
}
