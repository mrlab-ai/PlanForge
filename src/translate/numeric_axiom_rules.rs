/// Port of numeric_axiom_rules.py
/// Handles numeric axiom constant folding, layer computation, and equivalence detection.
use std::collections::{BTreeMap, HashMap, HashSet};

use ordered_float::OrderedFloat;

use super::pddl::axioms::InstantiatedNumericAxiom;
use super::pddl::f_expression::{
    FunctionalExpression, NumericConstant, PrimitiveNumericExpression,
};

/// Python: def handle_axioms(axioms)
/// Returns (processed_axioms, axioms_by_layer, max_layer, axiom_map, constant_axioms)
pub fn handle_axioms(
    axioms: &[InstantiatedNumericAxiom],
) -> (
    Vec<InstantiatedNumericAxiom>,
    BTreeMap<i32, Vec<InstantiatedNumericAxiom>>,
    i32,
    HashMap<PrimitiveNumericExpression, PrimitiveNumericExpression>,
    HashSet<InstantiatedNumericAxiom>,
) {
    if axioms.is_empty() {
        return (
            Vec::new(),
            BTreeMap::new(),
            -1,
            HashMap::new(),
            HashSet::new(),
        );
    }

    let mut processed_axioms = axioms.to_vec();
    identify_constants(&mut processed_axioms);

    let constant_axioms: HashSet<InstantiatedNumericAxiom> = processed_axioms
        .iter()
        .filter(|axiom| is_folded_constant_axiom(axiom))
        .cloned()
        .collect();

    let (axioms_by_layer, max_layer) = compute_axiom_layers(&processed_axioms, &constant_axioms);
    let axiom_map = identify_equivalent_axioms(&axioms_by_layer);

    (
        processed_axioms,
        axioms_by_layer,
        max_layer,
        axiom_map,
        constant_axioms,
    )
}

fn is_folded_constant_axiom(axiom: &InstantiatedNumericAxiom) -> bool {
    axiom.op.is_empty()
        && matches!(
            axiom.parts.first(),
            Some(FunctionalExpression::NumericConstant(_))
        )
}

fn axiom_by_pne(axioms: &[InstantiatedNumericAxiom]) -> HashMap<PrimitiveNumericExpression, usize> {
    axioms
        .iter()
        .enumerate()
        .map(|(idx, axiom)| (axiom.effect.clone(), idx))
        .collect()
}

fn identify_constants(axioms: &mut [InstantiatedNumericAxiom]) {
    let axiom_index = axiom_by_pne(axioms);
    let mut memo: HashMap<PrimitiveNumericExpression, Option<OrderedFloat<f64>>> = HashMap::new();
    let mut visiting: HashSet<PrimitiveNumericExpression> = HashSet::new();

    for idx in 0..axioms.len() {
        let _ = fold_axiom_if_constant(idx, axioms, &axiom_index, &mut memo, &mut visiting);
    }
}

fn fold_axiom_if_constant(
    idx: usize,
    axioms: &mut [InstantiatedNumericAxiom],
    axiom_index: &HashMap<PrimitiveNumericExpression, usize>,
    memo: &mut HashMap<PrimitiveNumericExpression, Option<OrderedFloat<f64>>>,
    visiting: &mut HashSet<PrimitiveNumericExpression>,
) -> Option<OrderedFloat<f64>> {
    let effect = axioms[idx].effect.clone();
    if let Some(cached) = memo.get(&effect) {
        return *cached;
    }
    if !visiting.insert(effect.clone()) {
        return None;
    }

    let op = axioms[idx].op.clone();
    let parts = axioms[idx].parts.clone();

    let result = if op.is_empty() {
        match parts.first() {
            Some(FunctionalExpression::NumericConstant(nc)) => Some(nc.value),
            Some(part) if parts.len() == 1 => {
                resolve_constant_part(part, axioms, axiom_index, memo, visiting)
            }
            _ => None,
        }
    } else {
        let mut values = Vec::with_capacity(parts.len());
        for part in &parts {
            if let Some(value) = resolve_constant_part(part, axioms, axiom_index, memo, visiting) {
                values.push(value);
            } else {
                visiting.remove(&effect);
                memo.insert(effect, None);
                return None;
            }
        }
        evaluate_constant_expression(&op, &values)
    };

    if let Some(value) = result {
        axioms[idx].op.clear();
        axioms[idx].parts = vec![FunctionalExpression::NumericConstant(NumericConstant {
            value,
        })];
        axioms[idx].effect.ntype = 'C';
    }

    visiting.remove(&effect);
    memo.insert(effect, result);
    result
}

fn resolve_constant_part(
    part: &FunctionalExpression,
    axioms: &mut [InstantiatedNumericAxiom],
    axiom_index: &HashMap<PrimitiveNumericExpression, usize>,
    memo: &mut HashMap<PrimitiveNumericExpression, Option<OrderedFloat<f64>>>,
    visiting: &mut HashSet<PrimitiveNumericExpression>,
) -> Option<OrderedFloat<f64>> {
    match part {
        FunctionalExpression::NumericConstant(nc) => Some(nc.value),
        FunctionalExpression::PrimitiveNumericExpression(pne) => {
            axiom_index.get(pne).and_then(|&dep_idx| {
                fold_axiom_if_constant(dep_idx, axioms, axiom_index, memo, visiting)
            })
        }
        _ => None,
    }
}

fn evaluate_constant_expression(
    op: &str,
    values: &[OrderedFloat<f64>],
) -> Option<OrderedFloat<f64>> {
    if values.is_empty() {
        return None;
    }

    let numeric_values: Vec<f64> = values.iter().map(|value| value.into_inner()).collect();
    let result = match op {
        "+" => numeric_values.into_iter().sum(),
        "*" => numeric_values.into_iter().product(),
        "-" => {
            if numeric_values.len() == 1 {
                -numeric_values[0]
            } else {
                let mut iter = numeric_values.into_iter();
                let first = iter.next()?;
                iter.fold(first, |acc, value| acc - value)
            }
        }
        "/" => {
            let mut iter = numeric_values.into_iter();
            let first = iter.next()?;
            iter.fold(first, |acc, value| acc / value)
        }
        _ => return None,
    };

    Some(OrderedFloat(result))
}

fn compute_axiom_layers(
    axioms: &[InstantiatedNumericAxiom],
    constant_axioms: &HashSet<InstantiatedNumericAxiom>,
) -> (BTreeMap<i32, Vec<InstantiatedNumericAxiom>>, i32) {
    let axiom_index = axiom_by_pne(axioms);
    let constant_effects: HashSet<PrimitiveNumericExpression> = constant_axioms
        .iter()
        .map(|axiom| axiom.effect.clone())
        .collect();
    let mut layer_cache: HashMap<PrimitiveNumericExpression, i32> = HashMap::new();

    fn compute_layer_for_expr(
        expr: &FunctionalExpression,
        axioms: &[InstantiatedNumericAxiom],
        axiom_index: &HashMap<PrimitiveNumericExpression, usize>,
        constant_effects: &HashSet<PrimitiveNumericExpression>,
        layer_cache: &mut HashMap<PrimitiveNumericExpression, i32>,
    ) -> i32 {
        match expr {
            FunctionalExpression::PrimitiveNumericExpression(pne) => {
                if let Some(&idx) = axiom_index.get(pne) {
                    compute_layer_for_axiom(idx, axioms, axiom_index, constant_effects, layer_cache)
                } else {
                    -1
                }
            }
            _ => -1,
        }
    }

    fn compute_layer_for_axiom(
        idx: usize,
        axioms: &[InstantiatedNumericAxiom],
        axiom_index: &HashMap<PrimitiveNumericExpression, usize>,
        constant_effects: &HashSet<PrimitiveNumericExpression>,
        layer_cache: &mut HashMap<PrimitiveNumericExpression, i32>,
    ) -> i32 {
        let effect = axioms[idx].effect.clone();
        if let Some(&layer) = layer_cache.get(&effect) {
            return layer;
        }

        let layer = if constant_effects.contains(&effect) {
            -1
        } else {
            let mut current = 0;
            for part in &axioms[idx].parts {
                current = current.max(
                    compute_layer_for_expr(
                        part,
                        axioms,
                        axiom_index,
                        constant_effects,
                        layer_cache,
                    ) + 1,
                );
            }
            current
        };

        layer_cache.insert(effect, layer);
        layer
    }

    let mut max_layer = -2;
    for idx in 0..axioms.len() {
        max_layer = max_layer.max(compute_layer_for_axiom(
            idx,
            axioms,
            &axiom_index,
            &constant_effects,
            &mut layer_cache,
        ));
    }

    let mut axioms_by_layer: BTreeMap<i32, Vec<InstantiatedNumericAxiom>> = BTreeMap::new();
    for axiom in axioms {
        let layer = *layer_cache.get(&axiom.effect).unwrap_or(&-1);
        axioms_by_layer
            .entry(layer)
            .or_default()
            .push(axiom.clone());
    }

    (axioms_by_layer, max_layer)
}

fn identify_equivalent_axioms(
    axioms_by_layer: &BTreeMap<i32, Vec<InstantiatedNumericAxiom>>,
) -> HashMap<PrimitiveNumericExpression, PrimitiveNumericExpression> {
    let mut axiom_map: HashMap<PrimitiveNumericExpression, PrimitiveNumericExpression> =
        HashMap::new();

    for axioms in axioms_by_layer.values() {
        let mut key_to_unique: HashMap<
            (String, Vec<FunctionalExpression>),
            PrimitiveNumericExpression,
        > = HashMap::new();

        for axiom in axioms {
            let mapped_args: Vec<FunctionalExpression> = axiom
                .parts
                .iter()
                .map(|part| match part {
                    FunctionalExpression::PrimitiveNumericExpression(pne) => {
                        if let Some(mapped_effect) = axiom_map.get(pne) {
                            FunctionalExpression::PrimitiveNumericExpression(mapped_effect.clone())
                        } else {
                            FunctionalExpression::PrimitiveNumericExpression(pne.clone())
                        }
                    }
                    _ => part.clone(),
                })
                .collect();

            let key = (axiom.op.clone(), mapped_args);
            if let Some(existing_effect) = key_to_unique.get(&key) {
                axiom_map.insert(axiom.effect.clone(), existing_effect.clone());
            } else {
                key_to_unique.insert(key, axiom.effect.clone());
            }
        }
    }

    axiom_map
}
