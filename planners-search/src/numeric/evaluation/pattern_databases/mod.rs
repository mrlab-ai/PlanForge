pub mod causal_graph;
pub mod numeric_size_estimator;
pub(crate) mod numeric_support;
pub mod pattern_database;
pub mod pattern_generator_greedy;
pub mod pdb_heuristic;
pub mod projected_task;
pub(crate) mod utils;
pub mod variable_order_finder;

use planners_sas::numeric::numeric_task::AbstractNumericTask;

pub trait NumericAbstractTask: AbstractNumericTask {
	fn abstract_state_values(
		&self,
		propositional_values: &[i32],
		numeric_values: &[f64],
	) -> Result<(Vec<i32>, Vec<f64>), String>;

	fn evaluated_initial_abstract_state_values(&self) -> Result<(Vec<i32>, Vec<f64>), String>;

	fn abstract_operator_costs(&self) -> &[f64];

	fn abstract_propositional_var_ids(&self) -> &[usize];

	fn abstract_numeric_var_ids(&self) -> &[usize];
}
