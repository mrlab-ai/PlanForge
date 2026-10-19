//! Propositional values of packed concrete states.

use super::*;

impl CartesianSemantics<'_> {
    pub(in super::super) fn concrete_prop_values(
        &self,
        packer: &StatePacker,
        packed: &[u64],
        out: &mut Vec<ExplicitValueIndex>,
    ) {
        out.clear();
        out.extend(
            (0..self.task.get_num_variables())
                .map(|var_id| ExplicitValueIndex::from_usize(packer.get(packed, var_id) as usize)),
        );
    }
}
