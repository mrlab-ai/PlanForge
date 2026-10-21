//! Collections of Cartesian abstractions: the generic collection generator
//! with Cartesian members.

use super::*;
use crate::evaluation::abstractions::collections::generator::CollectionGenerator;

pub type CartesianAbstractionCollectionGenerator =
    CollectionGenerator<CartesianAbstractionGenerator>;
