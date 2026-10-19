//! Regression flaws on the Cartesian abstract plan; progression flaws are
//! [shared](crate::evaluation::abstractions::common::cegar::progression).

mod regression;

pub(super) use regression::*;
