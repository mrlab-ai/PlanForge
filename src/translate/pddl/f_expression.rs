//! PDDL function expressions
//! Port of python/translate/pddl/f_expression.py

use std::collections::HashMap;

// Alias for compatibility
pub type FunctionExpression = NumericExpression;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumericExpression {
    Constant(i64),
    PrimitiveNumeric(PrimitiveNumericExpression),
    Arithmetic(ArithmeticExpression),
    AdditiveInverse(Box<NumericExpression>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimitiveNumericExpression {
    pub symbol: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArithmeticExpression {
    pub operator: ArithmeticOperator,
    pub operands: Vec<NumericExpression>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArithmeticOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl NumericExpression {
    pub fn simplified(self) -> Self {
        // TODO: Implement expression simplification
        self
    }

    pub fn substitute(&self, _substitution: &HashMap<String, String>) -> Self {
        // TODO: Implement variable substitution
        self.clone()
    }
}
