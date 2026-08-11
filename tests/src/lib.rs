//! Integration tests for the planner, parameterised over the fixture corpora in
//! `tests/assets`.
//!
//! The crate has no library surface of its own; everything lives behind
//! `cfg(test)`. See [`corpus`] for the shared harness and for why every table in
//! here is compared set-wise against what is on disk.

#[cfg(test)]
mod corpus;
#[cfg(test)]
mod derived_predicate_tests;
#[cfg(test)]
mod determinism_tests;
#[cfg(test)]
mod goal_census;
#[cfg(test)]
mod numeric_condition_tests;
#[cfg(test)]
mod numeric_corpus_tests;
#[cfg(test)]
mod sailing_simple_tests;
#[cfg(test)]
mod sgd_engine_tests;
#[cfg(test)]
mod sgd_transcription_tests;
#[cfg(test)]
mod strips_corpus_tests;
#[cfg(test)]
mod task_equivalence_tests;
