#[cfg(test)]
mod tests;

use std::cell::OnceCell;

use planforge_sas::numeric::numeric_task::AbstractNumericTask;

use super::max_additive_subsets::{compute_additive_vars, compute_max_additive_subsets};
use super::pattern_collection::PatternCollection;
use super::pdb_collection::PdbCollection;

pub struct PatternCollectionInformation<'task> {
    task: &'task dyn AbstractNumericTask,
    patterns: PatternCollection,
    max_pdb_states: usize,
    pdb_collection: OnceCell<PdbCollection<'task>>,
    max_additive_subsets: OnceCell<Vec<Vec<usize>>>,
}

impl<'task> PatternCollectionInformation<'task> {
    pub fn new(
        task: &'task dyn AbstractNumericTask,
        patterns: PatternCollection,
        max_pdb_states: usize,
    ) -> Self {
        Self {
            task,
            patterns: PatternCollection::new(patterns.into_vec()),
            max_pdb_states,
            pdb_collection: OnceCell::new(),
            max_additive_subsets: OnceCell::new(),
        }
    }

    pub fn patterns(&self) -> &PatternCollection {
        &self.patterns
    }

    pub fn get_pdbs(&self) -> Result<&PdbCollection<'task>, String> {
        if self.pdb_collection.get().is_none() {
            let pdb_collection =
                PdbCollection::new(self.task, self.patterns.clone(), self.max_pdb_states)?;
            let _ = self.pdb_collection.set(pdb_collection);
        }
        self.pdb_collection
            .get()
            .ok_or_else(|| "failed to initialize PDB collection".to_string())
    }

    pub fn get_max_additive_subsets(&self) -> Result<&[Vec<usize>], String> {
        let pdbs = self.get_pdbs()?;
        let max_additive_subsets = self.max_additive_subsets.get_or_init(|| {
            let are_additive = compute_additive_vars(self.task);
            compute_max_additive_subsets(pdbs.patterns(), &are_additive)
        });
        Ok(max_additive_subsets.as_slice())
    }
}
