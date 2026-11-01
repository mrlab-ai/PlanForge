//! Seeded choices among split candidates.

use super::*;

impl CartesianSemantics<'_> {
    pub(in super::super) fn choose_icaps_random_index(&self, candidate_count: usize) -> usize {
        debug_assert!(candidate_count > 0, "cannot choose from an empty split set");
        self.icaps_random
            .as_ref()
            .expect("ICAPS random selector must initialize its RNG")
            .borrow_mut()
            .uniform_index(candidate_count)
    }
}
