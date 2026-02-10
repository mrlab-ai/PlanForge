// Port of python/translate/numeric_axiom_rules.py into Rust.
// The Python module operates on NumericAxiom / InstantiatedNumericAxiom structures
// which contain fields: op (operator), parts (vector of other numeric expressions
// or constants), and effect (a primitive numeric expression). We mirror a
// minimal subset of these types here so the logic can be reused from Rust.

use ordered_float::OrderedFloat;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Eq)]
pub struct PrimitiveNumericExpression {
    pub name: String,
    pub args: Vec<String>,
}

impl PartialEq for PrimitiveNumericExpression {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.args == other.args
    }
}

impl Hash for PrimitiveNumericExpression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        for a in &self.args {
            a.hash(state);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NumericConstant(pub OrderedFloat<f64>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NumericPart {
    Primitive(PrimitiveNumericExpression),
    Axiom(Box<InstantiatedNumericAxiom>),
    Constant(NumericConstant),
}

#[derive(Debug)]
pub enum NumericAxiomError {
    DivideByZero { axiom: String },
    UnknownOperator { axiom: String, op: String },
    NonConstantPart { axiom: String },
}

impl std::fmt::Display for NumericAxiomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumericAxiomError::DivideByZero { axiom } => {
                write!(f, "division by zero while evaluating {}", axiom)
            }
            NumericAxiomError::UnknownOperator { axiom, op } => {
                write!(f, "unknown operator '{}' in {}", op, axiom)
            }
            NumericAxiomError::NonConstantPart { axiom } => {
                write!(f, "non-constant part in {}", axiom)
            }
        }
    }
}

impl std::error::Error for NumericAxiomError {}

#[derive(Debug, Clone)]
pub struct InstantiatedNumericAxiom {
    pub name: String,
    pub op: Option<String>,
    pub parts: Vec<NumericPart>,
    pub effect: PrimitiveNumericExpression,
    // ntype not modeled here; Python uses it to mark constants vs derived
}

impl PartialEq for InstantiatedNumericAxiom {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl Eq for InstantiatedNumericAxiom {}

impl Hash for InstantiatedNumericAxiom {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

/// Build a map from effect (PrimitiveNumericExpression) to the axiom that
/// produces it, similar to python's axiom_by_PNE
pub fn axiom_by_pne(
    axioms: &[InstantiatedNumericAxiom],
) -> HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom> {
    let mut m = HashMap::new();
    for ax in axioms {
        m.insert(ax.effect.clone(), ax.clone());
    }
    m
}

/// Identify constant axioms: axioms whose computation reduces to a numeric constant.
/// Mirrors identify_constants in Python file.
pub fn identify_constants_checked(
    axioms: &[InstantiatedNumericAxiom],
    axiom_by_pne: &HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom>,
) -> Result<Vec<InstantiatedNumericAxiom>, NumericAxiomError> {
    fn is_constant(
        ax: &InstantiatedNumericAxiom,
        ax_by_pne: &HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom>,
    ) -> Result<Option<f64>, NumericAxiomError> {
        // Recursive helper. If the axiom can be reduced to a single numeric constant,
        // return Some(value), otherwise None.
        if ax.op.is_none() && ax.parts.len() == 1 {
            if let NumericPart::Constant(NumericConstant(v)) = &ax.parts[0] {
                return Ok(Some(v.into_inner()));
            }
        }
        // Otherwise, check if all parts are constants (recursively).
        let mut values: Vec<f64> = Vec::new();
        for part in &ax.parts {
            match part {
                NumericPart::Constant(NumericConstant(v)) => values.push(v.into_inner()),
                NumericPart::Primitive(pne) => {
                    if let Some(dep_ax) = ax_by_pne.get(pne) {
                        if let Some(v) = is_constant(dep_ax, ax_by_pne)? {
                            values.push(v);
                        } else {
                            return Ok(None);
                        }
                    } else {
                        return Ok(None);
                    }
                }
                NumericPart::Axiom(boxed) => {
                    if let Some(v) = is_constant(boxed, ax_by_pne)? {
                        values.push(v);
                    } else {
                        return Ok(None);
                    }
                }
            }
        }
        if !values.is_empty() {
            // reduce using the operator
            if let Some(op) = &ax.op {
                // build string like "v1 op v2 op v3" and eval
                // to avoid an eval, implement simple ops: + - * /
                let mut iter = values.into_iter();
                if let Some(mut acc) = iter.next() {
                    if op == "-" && ax.parts.len() == 1 {
                        return Ok(Some(-acc));
                    }
                    for v in iter {
                        match op.as_str() {
                            "+" => acc = acc + v,
                            "-" => acc = acc - v,
                            "*" => acc = acc * v,
                            "/" => {
                                if v != 0.0 {
                                    acc = acc / v
                                } else {
                                    return Err(NumericAxiomError::DivideByZero {
                                        axiom: ax.name.clone(),
                                    });
                                }
                            }
                            _ => {
                                return Err(NumericAxiomError::UnknownOperator {
                                    axiom: ax.name.clone(),
                                    op: op.clone(),
                                })
                            }
                        }
                    }
                    return Ok(Some(acc));
                }
            } else {
                // no op and single value was handled earlier; but if multiple values and no op, can't combine
                if values.len() == 1 {
                    return Ok(Some(values[0]));
                }
                return Err(NumericAxiomError::NonConstantPart {
                    axiom: ax.name.clone(),
                });
            }
        }
        Ok(None)
    }

    let mut constants = Vec::new();
    for ax in axioms {
        if let Some(_v) = is_constant(ax, axiom_by_pne)? {
            constants.push(ax.clone());
        }
    }
    Ok(constants)
}

pub fn identify_constants(
    axioms: &[InstantiatedNumericAxiom],
    axiom_by_pne: &HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom>,
) -> Vec<InstantiatedNumericAxiom> {
    identify_constants_checked(axioms, axiom_by_pne).unwrap_or_else(|_| Vec::new())
}

/// Identify constant axioms and simplify them in-place (matching Python behavior).
pub fn identify_constants_inplace(
    axioms: &mut Vec<InstantiatedNumericAxiom>,
) -> Vec<InstantiatedNumericAxiom> {
    let mut index_by_effect: HashMap<PrimitiveNumericExpression, usize> = HashMap::new();
    for (idx, ax) in axioms.iter().enumerate() {
        index_by_effect.insert(ax.effect.clone(), idx);
    }

    let mut cache: HashMap<usize, Option<f64>> = HashMap::new();

    fn eval_part(
        part: &NumericPart,
        axioms: &mut Vec<InstantiatedNumericAxiom>,
        index_by_effect: &HashMap<PrimitiveNumericExpression, usize>,
        cache: &mut HashMap<usize, Option<f64>>,
        visiting: &mut HashSet<usize>,
    ) -> Option<f64> {
        match part {
            NumericPart::Constant(NumericConstant(v)) => Some(v.into_inner()),
            NumericPart::Primitive(pne) => {
                if let Some(idx) = index_by_effect.get(pne) {
                    eval_axiom(*idx, axioms, index_by_effect, cache, visiting)
                } else {
                    None
                }
            }
            NumericPart::Axiom(boxed) => {
                if let Some(idx) = index_by_effect.get(&boxed.effect) {
                    eval_axiom(*idx, axioms, index_by_effect, cache, visiting)
                } else {
                    // Fallback: try to evaluate boxed axiom directly
                    eval_parts(&boxed.op, &boxed.parts, axioms, index_by_effect, cache, visiting)
                }
            }
        }
    }

    fn eval_parts(
        op: &Option<String>,
        parts: &[NumericPart],
        axioms: &mut Vec<InstantiatedNumericAxiom>,
        index_by_effect: &HashMap<PrimitiveNumericExpression, usize>,
        cache: &mut HashMap<usize, Option<f64>>,
        visiting: &mut HashSet<usize>,
    ) -> Option<f64> {
        let mut values: Vec<f64> = Vec::new();
        for part in parts {
            if let Some(v) = eval_part(part, axioms, index_by_effect, cache, visiting) {
                values.push(v);
            } else {
                return None;
            }
        }
        if values.is_empty() {
            return None;
        }

        if let Some(op) = op {
            if op == "-" && values.len() == 1 {
                return Some(-values[0]);
            }
            let mut iter = values.into_iter();
            let mut acc = iter.next()?;
            for v in iter {
                match op.as_str() {
                    "+" => acc += v,
                    "-" => acc -= v,
                    "*" => acc *= v,
                    "/" => {
                        if v == 0.0 {
                            return None;
                        }
                        acc /= v;
                    }
                    _ => return None,
                }
            }
            Some(acc)
        } else if values.len() == 1 {
            Some(values[0])
        } else {
            None
        }
    }

    fn eval_axiom(
        idx: usize,
        axioms: &mut Vec<InstantiatedNumericAxiom>,
        index_by_effect: &HashMap<PrimitiveNumericExpression, usize>,
        cache: &mut HashMap<usize, Option<f64>>,
        visiting: &mut HashSet<usize>,
    ) -> Option<f64> {
        if let Some(v) = cache.get(&idx) {
            return *v;
        }
        if !visiting.insert(idx) {
            return None;
        }

        let (op, parts) = {
            let ax = &axioms[idx];
            (ax.op.clone(), ax.parts.clone())
        };

        let result = eval_parts(&op, &parts, axioms, index_by_effect, cache, visiting);
        if let Some(value) = result {
            let ax = &mut axioms[idx];
            ax.op = None;
            ax.parts = vec![NumericPart::Constant(NumericConstant(OrderedFloat(value)))];
            cache.insert(idx, Some(value));
        } else {
            cache.insert(idx, None);
        }

        visiting.remove(&idx);
        result
    }

    let mut visiting: HashSet<usize> = HashSet::new();
    for idx in 0..axioms.len() {
        let _ = eval_axiom(idx, axioms, &index_by_effect, &mut cache, &mut visiting);
    }

    let mut constants = Vec::new();
    for idx in 0..axioms.len() {
        if let Some(Some(_)) = cache.get(&idx) {
            constants.push(axioms[idx].clone());
        }
    }
    constants
}

/// Compute axiom layers (dependency depth), similar to compute_axiom_layers in Python.
pub fn compute_axiom_layers(
    axioms: &[InstantiatedNumericAxiom],
    constant_axioms: &[InstantiatedNumericAxiom],
    axiom_by_pne: &HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom>,
) -> (HashMap<i32, Vec<InstantiatedNumericAxiom>>, i32) {
    const CONSTANT_OR_NO_AXIOM: i32 = -1;
    const UNKNOWN_LAYER: i32 = -2;

    let mut layers: HashMap<PrimitiveNumericExpression, i32> = HashMap::new();
    for ax in axioms {
        layers.insert(ax.effect.clone(), UNKNOWN_LAYER);
    }

    let const_set: HashSet<PrimitiveNumericExpression> =
        constant_axioms.iter().map(|a| a.effect.clone()).collect();

    fn compute_layer_rec(
        pne: &PrimitiveNumericExpression,
        axiom_by_pne: &HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom>,
        layers: &mut HashMap<PrimitiveNumericExpression, i32>,
        const_set: &HashSet<PrimitiveNumericExpression>,
    ) -> i32 {
        const CONSTANT_OR_NO_AXIOM: i32 = -1;
        const UNKNOWN_LAYER: i32 = -2;
        if let Some(layer) = layers.get(pne) {
            if *layer != UNKNOWN_LAYER {
                return *layer;
            }
        } else {
            return CONSTANT_OR_NO_AXIOM;
        }

        if const_set.contains(pne) {
            layers.insert(pne.clone(), CONSTANT_OR_NO_AXIOM);
            return CONSTANT_OR_NO_AXIOM;
        }

        let ax = match axiom_by_pne.get(pne) {
            Some(ax) => ax,
            None => {
                layers.insert(pne.clone(), CONSTANT_OR_NO_AXIOM);
                return CONSTANT_OR_NO_AXIOM;
            }
        };

        let mut layer = 0;
        for part in &ax.parts {
            let part_layer = match part {
                NumericPart::Primitive(p) => {
                    if axiom_by_pne.contains_key(p) {
                        compute_layer_rec(p, axiom_by_pne, layers, const_set) + 1
                    } else {
                        0
                    }
                }
                NumericPart::Axiom(boxed) => {
                    if axiom_by_pne.contains_key(&boxed.effect) {
                        compute_layer_rec(&boxed.effect, axiom_by_pne, layers, const_set) + 1
                    } else {
                        0
                    }
                }
                NumericPart::Constant(_) => 0,
            };
            layer = layer.max(part_layer);
        }

        layers.insert(pne.clone(), layer);
        layer
    }

    for ax in axioms {
        let _ = compute_layer_rec(&ax.effect, axiom_by_pne, &mut layers, &const_set);
    }

    let mut max_layer = -2i32;
    for v in layers.values() {
        max_layer = max_layer.max(*v);
    }

    let mut layer_to_axioms: HashMap<i32, Vec<InstantiatedNumericAxiom>> = HashMap::new();
    for ax in axioms {
        let layer = *layers.get(&ax.effect).unwrap_or(&CONSTANT_OR_NO_AXIOM);
        layer_to_axioms.entry(layer).or_default().push(ax.clone());
    }

    (layer_to_axioms, max_layer)
}

/// Identify equivalent axioms within each layer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AxiomKeyPart {
    PNE(PrimitiveNumericExpression),
    Constant(OrderedFloat<f64>),
}

pub fn identify_equivalent_axioms(
    axioms_by_layer: &HashMap<i32, Vec<InstantiatedNumericAxiom>>,
    _axiom_by_pne: &HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom>,
) -> HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom> {
    let mut axiom_map: HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom> = HashMap::new();
    for (_layer, axioms) in axioms_by_layer {
        let mut key_to_unique: HashMap<(Option<String>, Vec<AxiomKeyPart>), InstantiatedNumericAxiom> =
            HashMap::new();
        for ax in axioms {
            let mut mapped_args: Vec<AxiomKeyPart> = Vec::new();
            for part in &ax.parts {
                match part {
                    NumericPart::Primitive(p) => {
                        if let Some(mapped) = axiom_map.get(p) {
                            mapped_args.push(AxiomKeyPart::PNE(mapped.effect.clone()));
                        } else {
                            mapped_args.push(AxiomKeyPart::PNE(p.clone()));
                        }
                    }
                    NumericPart::Axiom(boxed) => {
                        if let Some(mapped) = axiom_map.get(&boxed.effect) {
                            mapped_args.push(AxiomKeyPart::PNE(mapped.effect.clone()));
                        } else {
                            mapped_args.push(AxiomKeyPart::PNE(boxed.effect.clone()));
                        }
                    }
                    NumericPart::Constant(c) => mapped_args.push(AxiomKeyPart::Constant(c.0)),
                }
            }
            let key = (ax.op.clone(), mapped_args.clone());
            if let Some(existing) = key_to_unique.get(&key) {
                axiom_map.insert(ax.effect.clone(), existing.clone());
            } else {
                key_to_unique.insert(key, ax.clone());
            }
        }
    }
    axiom_map
}

/// Top-level handler matching the Python API: returns (axioms_by_layer, max_layer, axiom_map, constant_axioms)
pub fn handle_axioms(
    axioms: &[InstantiatedNumericAxiom],
) -> (
    HashMap<i32, Vec<InstantiatedNumericAxiom>>,
    i32,
    HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom>,
    Vec<InstantiatedNumericAxiom>,
) {
    let axiom_by_pne = axiom_by_pne(axioms);
    let constant_axioms = identify_constants(axioms, &axiom_by_pne);
    let (axioms_by_layer, max_layer) =
        compute_axiom_layers(axioms, &constant_axioms, &axiom_by_pne);
    let axiom_map = identify_equivalent_axioms(&axioms_by_layer, &axiom_by_pne);
    (axioms_by_layer, max_layer, axiom_map, constant_axioms)
}

pub fn handle_axioms_checked(
    axioms: &[InstantiatedNumericAxiom],
) -> Result<
    (
        HashMap<i32, Vec<InstantiatedNumericAxiom>>,
        i32,
        HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom>,
        Vec<InstantiatedNumericAxiom>,
    ),
    NumericAxiomError,
> {
    let axiom_by_pne = axiom_by_pne(axioms);
    let constant_axioms = identify_constants_checked(axioms, &axiom_by_pne)?;
    let (axioms_by_layer, max_layer) =
        compute_axiom_layers(axioms, &constant_axioms, &axiom_by_pne);
    let axiom_map = identify_equivalent_axioms(&axioms_by_layer, &axiom_by_pne);
    Ok((axioms_by_layer, max_layer, axiom_map, constant_axioms))
}
