// Port of python/translate/numeric_axiom_rules.py into Rust.
// The Python module operates on NumericAxiom / InstantiatedNumericAxiom structures
// which contain fields: op (operator), parts (vector of other numeric expressions
// or constants), and effect (a primitive numeric expression). We mirror a
// minimal subset of these types here so the logic can be reused from Rust.

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
pub struct NumericConstant(pub i64);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NumericPart {
    Primitive(PrimitiveNumericExpression),
    Axiom(Box<InstantiatedNumericAxiom>),
    Constant(NumericConstant),
}

#[derive(Debug)]
pub enum NumericAxiomError {
    DivideByZero {
        axiom: String,
    },
    UnknownOperator {
        axiom: String,
        op: String,
    },
    NonConstantPart {
        axiom: String,
    },
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
            && self.parts == other.parts
            && self.op == other.op
            && self.effect == other.effect
    }
}
impl Eq for InstantiatedNumericAxiom {}

impl Hash for InstantiatedNumericAxiom {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.op.hash(state);
        for p in &self.parts {
            p.hash(state);
        }
        self.effect.hash(state);
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
    ) -> Result<Option<i64>, NumericAxiomError> {
        // Recursive helper. If the axiom can be reduced to a single numeric constant,
        // return Some(value), otherwise None.
        if ax.op.is_none() && ax.parts.len() == 1 {
            if let NumericPart::Constant(NumericConstant(v)) = &ax.parts[0] {
                return Ok(Some(*v));
            }
        }
        // Otherwise, check if all parts are constants (recursively).
        let mut values: Vec<i64> = Vec::new();
        for part in &ax.parts {
            match part {
                NumericPart::Constant(NumericConstant(v)) => values.push(*v),
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
                                if v != 0 {
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

/// Compute axiom layers (dependency depth), similar to compute_axiom_layers in Python.
pub fn compute_axiom_layers(
    axioms: &[InstantiatedNumericAxiom],
    constant_axioms: &[InstantiatedNumericAxiom],
    _axiom_by_pne: &HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom>,
) -> (HashMap<i32, Vec<InstantiatedNumericAxiom>>, i32) {
    const CONSTANT_OR_NO_AXIOM: i32 = -1;
    const UNKNOWN_LAYER: i32 = -2;

    // Build dependency map: axiom -> set of parents (parts)
    let mut depends_on: HashMap<String, Vec<String>> = HashMap::new();
    for ax in axioms {
        let key = ax.name.clone();
        let mut deps = Vec::new();
        for part in &ax.parts {
            match part {
                NumericPart::Primitive(p) => deps.push(format!("PNE:{}", p.name)),
                NumericPart::Axiom(boxed) => deps.push(format!("AX:{}", boxed.name)),
                NumericPart::Constant(_) => {}
            }
        }
        depends_on.insert(key, deps);
    }

    let mut layers: HashMap<String, i32> = HashMap::new();
    for ax in axioms {
        layers.insert(ax.name.clone(), UNKNOWN_LAYER);
    }

    let const_set: HashSet<String> = constant_axioms.iter().map(|a| a.name.clone()).collect();

    fn compute_layer_rec(
        name: &str,
        depends_on: &HashMap<String, Vec<String>>,
        layers: &mut HashMap<String, i32>,
        const_set: &HashSet<String>,
    ) -> i32 {
        const CONSTANT_OR_NO_AXIOM: i32 = -1;
        const UNKNOWN_LAYER: i32 = -2;
        let layer = *layers.get(name).unwrap_or(&CONSTANT_OR_NO_AXIOM);
        if layer != UNKNOWN_LAYER {
            return layer;
        }
        if const_set.contains(name) {
            layers.insert(name.to_string(), CONSTANT_OR_NO_AXIOM);
            return CONSTANT_OR_NO_AXIOM;
        }
        // compute max of parents
        let mut max_layer = 0;
        if let Some(parents) = depends_on.get(name) {
            for p in parents {
                // parent keys may be encoded; ignore constants
                if p.starts_with("AX:") {
                    let pname = &p[3..];
                    let child_layer = compute_layer_rec(pname, depends_on, layers, const_set);
                    max_layer = max_layer.max(child_layer + 1);
                } else if p.starts_with("PNE:") {
                    // If a PNE refers to an axiom, the axiom key would be in depends_on; otherwise treat as CONSTANT_OR_NO_AXIOM (-1)
                    // Here we conservatively treat PNE as CONSTANT_OR_NO_AXIOM unless there is a matching axiom name.
                    // No-op.
                }
            }
        }
        layers.insert(name.to_string(), max_layer);
        max_layer
    }

    for ax in axioms {
        let _ = compute_layer_rec(&ax.name, &depends_on, &mut layers, &const_set);
    }

    // find max layer
    let mut max_layer = -2i32;
    for (_k, v) in &layers {
        max_layer = max_layer.max(*v);
    }

    // invert map: layer -> axioms
    let mut layer_to_axioms: HashMap<i32, Vec<InstantiatedNumericAxiom>> = HashMap::new();
    for ax in axioms {
        let layer = *layers.get(&ax.name).unwrap_or(&CONSTANT_OR_NO_AXIOM);
        layer_to_axioms.entry(layer).or_default().push(ax.clone());
    }

    (layer_to_axioms, max_layer)
}

/// Identify equivalent axioms within each layer.
pub fn identify_equivalent_axioms(
    axioms_by_layer: &HashMap<i32, Vec<InstantiatedNumericAxiom>>,
    _axiom_by_pne: &HashMap<PrimitiveNumericExpression, InstantiatedNumericAxiom>,
) -> HashMap<InstantiatedNumericAxiom, InstantiatedNumericAxiom> {
    let mut axiom_map: HashMap<InstantiatedNumericAxiom, InstantiatedNumericAxiom> = HashMap::new();
    for (_layer, axioms) in axioms_by_layer {
        let mut key_to_unique: HashMap<
            (Option<String>, Vec<PrimitiveNumericExpression>),
            InstantiatedNumericAxiom,
        > = HashMap::new();
        for ax in axioms {
            // mapped_args: translate parts to either the mapped effect or the primitive expression
            let mut mapped_args: Vec<PrimitiveNumericExpression> = Vec::new();
            for part in &ax.parts {
                match part {
                    NumericPart::Primitive(p) => mapped_args.push(p.clone()),
                    NumericPart::Axiom(boxed) => {
                        if let Some(mapped) = axiom_map.get(boxed.as_ref()) {
                            mapped_args.push(mapped.effect.clone());
                        } else {
                            mapped_args.push(boxed.effect.clone());
                        }
                    }
                    NumericPart::Constant(c) => {
                        // represent constant as a PrimitiveNumericExpression with special name
                        mapped_args.push(PrimitiveNumericExpression {
                            name: format!("const:{}", c.0),
                            args: vec![],
                        });
                    }
                }
            }
            let key = (ax.op.clone(), mapped_args.clone());
            if let Some(existing) = key_to_unique.get(&key) {
                axiom_map.insert(ax.clone(), existing.clone());
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
    HashMap<InstantiatedNumericAxiom, InstantiatedNumericAxiom>,
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
        HashMap<InstantiatedNumericAxiom, InstantiatedNumericAxiom>,
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
