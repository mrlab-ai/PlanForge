//! Collections of Cartesian abstractions: the generic collection generator
//! with Cartesian members.

use super::generator::CartesianAbstractionGenerator;
use crate::abstractions::collections::generator::CollectionGenerator;

pub type CartesianAbstractionCollectionGenerator =
    CollectionGenerator<CartesianAbstractionGenerator>;
