//! The initial splits of a Cartesian abstraction: every split of
//! [`initial_splits`] divides each abstract state it divides strictly, as
//! long as the abstraction stays within `max_states`.

use anyhow::Result;

use super::semantics::CartesianSemantics;
use super::working::WorkingAbstraction;
use crate::evaluation::abstractions::common::cegar::initial_splits::{
    RefinementScope, initial_splits,
};
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;

/// Applies the initial splits `scope` asks for to `working`.
pub(super) fn apply_initial_splits(
    working: &mut WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    options: &CegarOptions,
    scope: &RefinementScope,
) -> Result<()> {
    for initial_split in initial_splits(semantics.cegar(), scope, options.random_seed) {
        for state_id in 0..working.states.len() {
            if working.states.len() >= options.max_states {
                return Ok(());
            }
            if let Some(split) = initial_split.in_state(state_id, &working.states[state_id]) {
                working.apply_split(semantics, split)?;
            }
        }
    }
    Ok(())
}
