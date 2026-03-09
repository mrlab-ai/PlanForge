/// Port of numeric_axiom_rules.py
/// Handles numeric axiom layers, constant identification, and equivalence.

use std::collections::{HashMap, HashSet, BTreeMap};
use ordered_float::OrderedFloat;

use super::pddl::axioms::InstantiatedNumericAxiom;
use super::pddl::f_expression::*;

/// Python: def handle_axioms(axioms)
/// Returns (axioms_by_layer, max_layer, axiom_map, constant_axioms)
pub fn handle_axioms(
    axioms: &[InstantiatedNumericAxiom],
) -> (
    BTreeMap<i32, Vec<InstantiatedNumericAxiom>>,  // axioms_by_layer
    i32,                                            // max_layer
    HashMap<PrimitiveNumericExpression, PrimitiveNumericExpression>, // axiom_map (equivalences)
    HashSet<InstantiatedNumericAxiom>,              // constant_axioms
) {
    if axioms.is_empty() {
        return (BTreeMap::new(), -1, HashMap::new(), HashSet::new());
    }

    // Step 1: Identify constant axioms
    let constant_axioms = identify_constants(axioms);

    // Step 2: Compute axiom layers
    let (axioms_by_layer, max_layer) = compute_axiom_layers(axioms, &constant_axioms);

    // Step 3: Identify equivalent axioms
    let axiom_map = identify_equivalent_axioms(axioms);

    (axioms_by_layer, max_layer, axiom_map, constant_axioms)
}

/// Python: def axiom_by_PNE(axioms)
fn axiom_by_pne(axioms: &[InstantiatedNumericAxiom]) -> HashMap<PrimitiveNumericExpression, Vec<usize>> {
    let mut result: HashMap<PrimitiveNumericExpression, Vec<usize>> = HashMap::new();
    for (i, axiom) in axioms.iter().enumerate() {
        result.entry(axiom.effect.clone())
            .or_insert_with(Vec::new)
            .push(i);
    }
    result
}

/// Python: def identify_constants(axioms)
fn identify_constants(axioms: &[InstantiatedNumericAxiom]) -> HashSet<InstantiatedNumericAxiom> {
    let mut constants: HashSet<InstantiatedNumericAxiom> = HashSet::new();

    // An axiom is constant if all its parts are numeric constants
    for axiom in axioms {
        let all_constant = axiom.parts.iter().all(|p| {
            matches!(p, FunctionalExpression::NumericConstant(_))
        });
        if all_constant {
            constants.insert(axiom.clone());
        }
    }

    // Transitively: if an axiom depends only on constants and other constant axioms
    let mut changed = true;
    while changed {
        changed = false;
        for axiom in axioms {
            if constants.contains(axiom) {
                continue;
            }
            let all_parts_constant = axiom.parts.iter().all(|p| {
                match p {
                    FunctionalExpression::NumericConstant(_) => true,
                    FunctionalExpression::PrimitiveNumericExpression(pne) => {
                        // Check if this PNE is the effect of a constant axiom
                        constants.iter().any(|ca| ca.effect == *pne)
                    }
                    _ => false,
                }
            });
            if all_parts_constant {
                constants.insert(axiom.clone());
                changed = true;
            }
        }
    }

    constants
}

/// Python: def compute_axiom_layers(axioms, constant_axioms)
fn compute_axiom_layers(
    axioms: &[InstantiatedNumericAxiom],
    constant_axioms: &HashSet<InstantiatedNumericAxiom>,
) -> (BTreeMap<i32, Vec<InstantiatedNumericAxiom>>, i32) {
    let mut layers: HashMap<PrimitiveNumericExpression, i32> = HashMap::new();

    // Initialize constant axioms to layer -1
    for axiom in constant_axioms {
        layers.insert(axiom.effect.clone(), -1);
    }

    // Initialize non-constant axioms to layer 0
    for axiom in axioms {
        if !constant_axioms.contains(axiom) {
            layers.entry(axiom.effect.clone()).or_insert(0);
        }
    }

    // Fixed-point computation
    let mut changed = true;
    while changed {
        changed = false;
        for axiom in axioms {
            if constant_axioms.contains(axiom) {
                continue;
            }
            let current_layer = *layers.get(&axiom.effect).unwrap_or(&0);
            let mut max_dep_layer = 0i32;

            for part in &axiom.parts {
                if let FunctionalExpression::PrimitiveNumericExpression(pne) = part {
                    if let Some(&dep_layer) = layers.get(pne) {
                        if dep_layer >= 0 {
                            max_dep_layer = max_dep_layer.max(dep_layer + 1);
                        }
                    }
                }
            }

            if max_dep_layer > current_layer {
                layers.insert(axiom.effect.clone(), max_dep_layer);
                changed = true;
            }
        }
    }

    // Group by layer
    let mut axioms_by_layer: BTreeMap<i32, Vec<InstantiatedNumericAxiom>> = BTreeMap::new();
    let mut max_layer = -1i32;

    for axiom in axioms {
        let layer = *layers.get(&axiom.effect).unwrap_or(&-1);
        axioms_by_layer.entry(layer)
            .or_insert_with(Vec::new)
            .push(axiom.clone());
        max_layer = max_layer.max(layer);
    }

    (axioms_by_layer, max_layer)
}

/// Python: def identify_equivalent_axioms(axioms)
fn identify_equivalent_axioms(
    axioms: &[InstantiatedNumericAxiom],
) -> HashMap<PrimitiveNumericExpression, PrimitiveNumericExpression> {
    let mut equivalences: HashMap<PrimitiveNumericExpression, PrimitiveNumericExpression> = HashMap::new();

    // Two axioms are equivalent if they have the same op and parts
    let mut axiom_by_definition: HashMap<(String, Vec<FunctionalExpression>), PrimitiveNumericExpression> = HashMap::new();

    for axiom in axioms {
        let key = (axiom.op.clone(), axiom.parts.clone());
        if let Some(existing) = axiom_by_definition.get(&key) {
            equivalences.insert(axiom.effect.clone(), existing.clone());
        } else {
            axiom_by_definition.insert(key, axiom.effect.clone());
        }
    }

    equivalences
}
