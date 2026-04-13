use super::pattern_collection::PatternCollection;
use super::pattern_database::{PatternDatabase, PdbHeuristicConfig};
use super::projected_task::{Pattern, ProjectedTask};
use super::utils;
use planners_sas::numeric::numeric_task::AbstractNumericTask;

pub struct PdbCollection<'task> {
    patterns: PatternCollection,
    pdbs: Vec<PatternDatabase<'task>>,
}

impl<'task> PdbCollection<'task> {
    pub fn new(
        task: &'task dyn AbstractNumericTask,
        patterns: PatternCollection,
        max_pdb_states: usize,
    ) -> Result<Self, String> {
        Self::with_heuristic_config(task, patterns, max_pdb_states, PdbHeuristicConfig::default())
    }

    pub fn with_heuristic_config(
        task: &'task dyn AbstractNumericTask,
        patterns: PatternCollection,
        max_pdb_states: usize,
        heuristic_config: PdbHeuristicConfig,
    ) -> Result<Self, String> {
        let patterns = PatternCollection::new(patterns.into_vec());
        let mut pdbs = Vec::with_capacity(patterns.len());

        for pattern in patterns.iter() {
            let projected_task =
                ProjectedTask::new(task, pattern).map_err(|err| err.to_string())?;
            utils::print_projection_summary(task, pattern, &projected_task);
            pdbs.push(PatternDatabase::with_heuristic_config(
                projected_task,
                max_pdb_states,
                heuristic_config,
            )?);
        }

        Ok(Self { patterns, pdbs })
    }

    pub fn len(&self) -> usize {
        self.pdbs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pdbs.is_empty()
    }

    pub fn patterns(&self) -> &PatternCollection {
        &self.patterns
    }

    pub fn pattern(&self, index: usize) -> Option<&Pattern> {
        self.patterns.as_slice().get(index)
    }

    pub fn pdbs(&self) -> &[PatternDatabase<'task>] {
        &self.pdbs
    }

    pub fn pdb(&self, index: usize) -> Option<&PatternDatabase<'task>> {
        self.pdbs.get(index)
    }

    pub fn expand_numeric_state_values_into(
        &self,
        numeric_values: &[f64],
        expanded_numeric_values: &mut Vec<f64>,
    ) -> Result<(), String> {
        match self.pdbs.first() {
            Some(pdb) => {
                pdb.expand_numeric_state_values_into(numeric_values, expanded_numeric_values)
            }
            None => {
                expanded_numeric_values.clear();
                expanded_numeric_values.extend_from_slice(numeric_values);
                Ok(())
            }
        }
    }

    pub fn requires_derived_numeric_values(&self) -> bool {
        self.pdbs
            .iter()
            .any(|pdb| pdb.requires_derived_numeric_values())
    }

    pub fn singleton_additive_subsets(&self) -> Vec<Vec<usize>> {
        (0..self.pdbs.len()).map(|index| vec![index]).collect()
    }
}

#[cfg(test)]
mod tests {
    use planners_sas::numeric::numeric_task::{
        Effect, ExplicitVariable, Fact, Metric, NumericRootTask, NumericType, NumericVariable,
        Operator,
    };

    use super::*;

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
            vec![Fact::new(1, 1)],
            vec![],
            vec![0, 0],
            vec![0.0],
            vec![Operator::new(
                "advance".to_string(),
                vec![Fact::new(0, 1)],
                vec![Effect::new(vec![], 1, 0, 1)],
                vec![],
                1,
            )],
            vec![],
            vec![],
            vec![],
            (0, 0),
        )
    }

    #[test]
    fn pdb_collection_builds_all_patterns() {
        let task = sample_task();
        let patterns = PatternCollection::new(vec![
            Pattern::new(vec![1], vec![]),
            Pattern::new(vec![0, 1], vec![]),
        ]);

        let collection = PdbCollection::new(&task, patterns, 32).unwrap();

        assert_eq!(collection.len(), 2);
        assert_eq!(
            collection.singleton_additive_subsets(),
            vec![vec![0], vec![1]]
        );
    }
}
