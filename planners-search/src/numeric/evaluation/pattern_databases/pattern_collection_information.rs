use std::cell::OnceCell;

use planners_sas::numeric::numeric_task::AbstractNumericTask;

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

#[cfg(test)]
mod tests {
    use planners_sas::numeric::numeric_task::{
        Effect, ExplicitVariable, Fact, Metric, NumericRootTask, NumericType, NumericVariable,
        Operator,
    };

    use super::*;
    use crate::numeric::evaluation::pattern_databases::projected_task::Pattern;

    fn simple_var(name: &str) -> ExplicitVariable {
        ExplicitVariable::new(
            2,
            name.to_string(),
            vec![format!("{name}=0"), format!("{name}=1")],
            -1,
            1,
        )
    }

    fn sample_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![simple_var("p"), simple_var("q")],
            vec![NumericVariable::new(
                "x".to_string(),
                NumericType::Regular,
                -1,
            )],
            vec![Fact::new(0, 1), Fact::new(1, 1)],
            vec![],
            vec![0, 0],
            vec![0.0],
            vec![
                Operator::new(
                    "set-p".to_string(),
                    vec![],
                    vec![Effect::new(vec![], 0, 0, 1)],
                    vec![],
                    2,
                ),
                Operator::new(
                    "set-q".to_string(),
                    vec![],
                    vec![Effect::new(vec![], 1, 0, 1)],
                    vec![],
                    3,
                ),
            ],
            vec![],
            vec![],
            vec![],
            (0, 0),
        )
    }

    #[test]
    fn collection_information_builds_pdbs_and_additive_subsets() {
        let task = sample_task();
        let info = PatternCollectionInformation::new(
            &task,
            PatternCollection::new(vec![
                Pattern::new(vec![0], vec![]),
                Pattern::new(vec![1], vec![]),
            ]),
            32,
        );

        assert_eq!(info.get_pdbs().unwrap().len(), 2);
        assert_eq!(info.get_max_additive_subsets().unwrap(), &[vec![0, 1]]);
    }
}
