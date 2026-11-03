//! Numeric pattern databases: projections of a task to a pattern, explored up
//! to a size limit, and the generators of their patterns.

pub mod causal_graph;
pub mod numeric_size_estimator;
pub mod pattern_collection;
pub mod pattern_database;
pub mod pattern_generator_greedy;
pub mod pattern_generator_hillclimbing;
pub mod pattern_generator_systematic;
pub mod pdb_collection;
pub mod variable_order_finder;
