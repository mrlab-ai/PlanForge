//! Function expression types for numeric PDDL
//!
//! This module defines types for representing numeric expressions in PDDL,
//! mirroring the Python f_expression module structure.

#[cfg(test)]
mod tests;

use crate::translate::pddl_parser::SExpr;
use ordered_float::OrderedFloat;

/// A functional expression in PDDL (base trait-like concept)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FunctionalExpression {
    /// Primitive numeric expression: a function symbol with arguments
    Primitive(PrimitiveNumericExpression),
    /// Numeric constant
    Constant(NumericConstant),
    /// Arithmetic expression: binary operation on two expressions
    Arithmetic(ArithmeticExpression),
    /// Additive inverse (unary minus)
    AdditiveInverse(AdditiveInverse),
}

/// A primitive numeric expression: function symbol with arguments
/// Example: (fuel ?v), (distance ?from ?to)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrimitiveNumericExpression {
    pub symbol: String,
    pub args: Vec<String>,
    /// 'D' for derived, 'S' for state, 'C' for constant
    pub ntype: char,
}

impl PrimitiveNumericExpression {
    pub fn new(symbol: String, args: Vec<String>, ntype: char) -> Self {
        PrimitiveNumericExpression {
            symbol,
            args,
            ntype,
        }
    }
}

impl std::fmt::Display for PrimitiveNumericExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.args.is_empty() {
            write!(f, "{}", self.symbol)
        } else {
            write!(f, "{}({})", self.symbol, self.args.join(", "))
        }
    }
}

/// A numeric constant value
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NumericConstant {
    pub value: OrderedFloat<f64>,
}

impl NumericConstant {
    pub fn new(value: f64) -> Self {
        NumericConstant {
            value: OrderedFloat(value),
        }
    }
}

impl std::fmt::Display for NumericConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", format_float(self.value.into_inner()))
    }
}

pub fn format_float(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{:.1}", value)
    } else {
        value.to_string()
    }
}

/// An arithmetic expression: op(parts...)
/// Examples: (+ (fuel) 10), (* (distance ?a ?b) 2)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArithmeticExpression {
    pub op: String, // "+", "-", "*", "/"
    pub parts: Vec<FunctionalExpression>,
}

impl ArithmeticExpression {
    pub fn new(op: String, parts: Vec<FunctionalExpression>) -> Self {
        ArithmeticExpression { op, parts }
    }
}

impl std::fmt::Display for ArithmeticExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}", self.op)?;
        for part in &self.parts {
            write!(f, " {}", part)?;
        }
        write!(f, ")")
    }
}

/// Additive inverse (unary minus): -(exp)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdditiveInverse {
    pub op: String, // "-"
    pub part: Box<FunctionalExpression>,
}

impl AdditiveInverse {
    pub fn new(part: FunctionalExpression) -> Self {
        AdditiveInverse {
            op: "-".to_string(),
            part: Box::new(part),
        }
    }
}

impl std::fmt::Display for AdditiveInverse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(- {})", self.part)
    }
}

impl std::fmt::Display for FunctionalExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FunctionalExpression::Primitive(p) => write!(f, "{}", p),
            FunctionalExpression::Constant(c) => write!(f, "{}", c),
            FunctionalExpression::Arithmetic(a) => write!(f, "{}", a),
            FunctionalExpression::AdditiveInverse(a) => write!(f, "{}", a),
        }
    }
}

/// Parse a SExpr into a FunctionalExpression
pub fn parse_functional_expression(sexpr: &SExpr) -> Option<FunctionalExpression> {
    match sexpr {
        SExpr::Atom(a) => {
            // Try to parse as number (try float first to handle both "3" and "3.0")
            if let Ok(value) = a.parse::<f64>() {
                Some(FunctionalExpression::Constant(NumericConstant::new(value)))
            } else {
                // Treat as primitive numeric expression (0-arity function)
                Some(FunctionalExpression::Primitive(
                    PrimitiveNumericExpression::new(a.clone(), vec![], 'S'),
                ))
            }
        }
        SExpr::List(list) if !list.is_empty() => {
            if let SExpr::Atom(op) = &list[0] {
                match op.as_str() {
                    "+" | "*" | "/" => {
                        // Binary arithmetic operation
                        let mut parts = Vec::new();
                        for part_sexpr in &list[1..] {
                            if let Some(part) = parse_functional_expression(part_sexpr) {
                                parts.push(part);
                            } else {
                                return None;
                            }
                        }
                        if parts.len() >= 2 {
                            Some(FunctionalExpression::Arithmetic(ArithmeticExpression::new(
                                op.clone(),
                                parts,
                            )))
                        } else {
                            None
                        }
                    }
                    "-" => {
                        // Could be unary minus or binary subtraction
                        if list.len() == 2 {
                            // Unary minus
                            if let Some(part) = parse_functional_expression(&list[1]) {
                                Some(FunctionalExpression::AdditiveInverse(AdditiveInverse::new(
                                    part,
                                )))
                            } else {
                                None
                            }
                        } else if list.len() >= 3 {
                            // Binary subtraction
                            let mut parts = Vec::new();
                            for part_sexpr in &list[1..] {
                                if let Some(part) = parse_functional_expression(part_sexpr) {
                                    parts.push(part);
                                } else {
                                    return None;
                                }
                            }
                            Some(FunctionalExpression::Arithmetic(ArithmeticExpression::new(
                                op.clone(),
                                parts,
                            )))
                        } else {
                            None
                        }
                    }
                    _ => {
                        // Primitive function call
                        let args = list[1..]
                            .iter()
                            .filter_map(|x| match x {
                                SExpr::Atom(a) => Some(a.clone()),
                                _ => None,
                            })
                            .collect();
                        Some(FunctionalExpression::Primitive(
                            PrimitiveNumericExpression::new(op.clone(), args, 'S'),
                        ))
                    }
                }
            } else {
                None
            }
        }
        _ => None,
    }
}
