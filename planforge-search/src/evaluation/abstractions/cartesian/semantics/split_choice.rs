//! Seeded choices among split candidates.

use super::*;

impl CartesianSemantics<'_> {
    pub(in super::super) fn choose_keyed_index(&self, keys: &[u64], tag: u64) -> usize {
        debug_assert!(
            !keys.is_empty(),
            "cannot choose from an empty Cartesian candidate set"
        );
        let Some(seed) = self.random_seed else {
            return 0;
        };
        keys.iter()
            .enumerate()
            .min_by_key(|(_, key)| mix_seed(seed ^ tag ^ **key))
            .map(|(index, _)| index)
            .expect("nonempty Cartesian key set has no minimum")
    }

    pub(in super::super) fn choose_split_index(&self, candidates: &[Split], tag: u64) -> usize {
        debug_assert!(
            !candidates.is_empty(),
            "cannot choose from an empty split set"
        );
        if let Some(rank) = self.split_selection_rank {
            let mut indices = (0..candidates.len()).collect::<Vec<_>>();
            indices.sort_by_key(|&index| {
                let dimension = match candidates[index].dimension() {
                    SplitDimension::Propositional(var_id) => (0usize, var_id),
                    SplitDimension::Numeric(var_id) => (1usize, var_id),
                };
                (dimension, split_choice_key(self, &candidates[index]))
            });
            return indices[rank % indices.len()];
        }
        let keys = candidates
            .iter()
            .map(|split| split_choice_key(self, split))
            .collect::<Vec<_>>();
        self.choose_keyed_index(&keys, tag)
    }

    pub(in super::super) fn choose_icaps_random_index(&self, candidate_count: usize) -> usize {
        debug_assert!(candidate_count > 0, "cannot choose from an empty split set");
        self.icaps_random
            .as_ref()
            .expect("ICAPS random selector must initialize its RNG")
            .borrow_mut()
            .uniform_index(candidate_count)
    }

    pub(in super::super) fn choose_random_split_index(&self, candidate_count: usize) -> usize {
        debug_assert!(candidate_count > 0, "cannot choose from an empty split set");
        self.random_split_rng
            .as_ref()
            .expect("native random selector must initialize its RNG")
            .borrow_mut()
            .gen_range(0..candidate_count)
    }
}
