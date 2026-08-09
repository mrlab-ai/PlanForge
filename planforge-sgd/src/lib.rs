//! Search-free plan synthesis by gradient descent.
//!
//! This crate synthesizes a plan by optimizing a horizon-wide continuous
//! *transcription* of bounded planning: every timestep holds a distribution over
//! actions and, per finite-domain variable, a distribution over that variable's
//! values. Constraint residuals encode preconditions, transitions and goals so
//! that a zero-residual integral assignment is exactly a valid plan; gradient
//! descent then looks for one.
//!
//! **There is no search here, by construction.** This crate depends only on
//! `planforge-sas`, so it cannot reach an open list, a successor generator, or a
//! heuristic even by accident. The only place a plan is applied step by step is
//! `planforge_sas::plan_verification`, which replays one fixed sequence to check
//! it — that is verification, not search, and it is what makes every returned
//! plan sound.
//!
//! Layers, in dependency order:
//!
//! * [`classical`] — the classical (propositional, non-numeric) task fragment
//!   this engine accepts, and why each restriction is needed.
//! * [`transcription`] — the static incidence structure, built once per task.
//! * [`residuals`] — constraint residuals over plain `f64` buffers. This is the
//!   reference semantics and the test oracle for the tensor backend.
//! * [`config`] — the engine's knobs, validated up front.
//!
//! Behind the `candle` feature:
//!
//! * [`tensor`] — the same residuals as candle tensors, plus exact segmented
//!   sum/product operators for sparse incidence arithmetic and causal support.
//! * [`adam`] — Adam with per-slice moment reset, which `candle_nn` cannot do.
//! * [`engine`] — the optimizer loop and the exact-verifier feedback.

pub mod classical;
pub mod config;
pub mod controller;
pub mod residuals;
pub mod transcription;

#[cfg(feature = "candle")]
pub mod adam;
#[cfg(feature = "candle")]
pub mod engine;
#[cfg(feature = "candle")]
pub mod tensor;

#[cfg(test)]
mod exactness;
#[cfg(all(test, feature = "candle"))]
mod tensor_tests;
#[cfg(test)]
mod testing;
