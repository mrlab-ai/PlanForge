use ordered_float::OrderedFloat;
/// Functional expression hierarchy for numeric PDDL.
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};

/// Root enum for functional expressions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FunctionalExpression {
    NumericConstant(NumericConstant),
    PrimitiveNumericExpression(PrimitiveNumericExpression),
    ArithmeticExpression(ArithmeticExpression),
    AdditiveInverse(AdditiveInverse),
}

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

impl fmt::Display for NumericConstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = self.value.into_inner();
        if value.fract() == 0.0 {
            write!(f, "{:.1}", value)
        } else {
            write!(f, "{}", value)
        }
    }
}

/// ntype is one of 'C' (constant), 'D' (derived), 'I' (instrumental/total-cost), 'R' (regular)
#[derive(Debug, Clone)]
pub struct PrimitiveNumericExpression {
    pub symbol: String,
    pub args: Vec<String>,
    pub ntype: char,
}

impl PrimitiveNumericExpression {
    pub fn new(symbol: String, args: Vec<String>) -> Self {
        PrimitiveNumericExpression {
            symbol,
            args,
            ntype: 'R',
        }
    }

    pub fn with_type(symbol: String, args: Vec<String>, ntype: char) -> Self {
        PrimitiveNumericExpression {
            symbol,
            args,
            ntype,
        }
    }
}

impl fmt::Display for PrimitiveNumericExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.args.is_empty() {
            write!(f, "PNE {}()", self.symbol)
        } else {
            write!(f, "PNE {}({})", self.symbol, self.args.join(", "))
        }
    }
}

impl PartialEq for PrimitiveNumericExpression {
    fn eq(&self, other: &Self) -> bool {
        self.symbol == other.symbol && self.args == other.args
    }
}

impl Eq for PrimitiveNumericExpression {}

impl Hash for PrimitiveNumericExpression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.symbol.hash(state);
        self.args.hash(state);
    }
}

/// op is one of "+", "-", "*", "/"
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArithmeticExpression {
    pub op: String,
    pub parts: Vec<FunctionalExpression>,
}

impl ArithmeticExpression {
    pub fn new(op: String, parts: Vec<FunctionalExpression>) -> Self {
        ArithmeticExpression { op, parts }
    }
}

impl fmt::Display for ArithmeticExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ArithExpr({}, {:?})", self.op, self.parts)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdditiveInverse {
    pub parts: Vec<FunctionalExpression>,
}

impl AdditiveInverse {
    pub fn new(parts: Vec<FunctionalExpression>) -> Self {
        AdditiveInverse { parts }
    }
}

impl fmt::Display for AdditiveInverse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AdditiveInverse({:?})", self.parts)
    }
}

// ============== FunctionAssignment and subclasses ==============

/// Represents assign/increase/decrease/scale-up/scale-down operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionAssignment {
    pub symbol: String, // "=", "+", "-", "*", "/"
    pub fluent: PrimitiveNumericExpression,
    pub expression: FunctionalExpression,
}

impl FunctionAssignment {
    pub fn new(
        symbol: String,
        fluent: PrimitiveNumericExpression,
        expression: FunctionalExpression,
    ) -> Self {
        FunctionAssignment {
            symbol,
            fluent,
            expression,
        }
    }

    pub fn instantiate(
        &self,
        var_mapping: &super::tasks::VarMapping,
        fluent_functions: &HashSet<PrimitiveNumericExpression>,
        init_function_vals: &HashMap<PrimitiveNumericExpression, f64>,
        task_function_admin: &mut super::tasks::DerivedFunctionAdministrator,
        new_constant_axioms: &mut Vec<super::axioms::InstantiatedNumericAxiom>,
    ) -> FunctionAssignment {
        let new_fluent_args: Vec<String> = self
            .fluent
            .args
            .iter()
            .map(|arg| var_mapping.resolve(arg).to_owned())
            .collect();
        let new_fluent = PrimitiveNumericExpression::with_type(
            self.fluent.symbol.clone(),
            new_fluent_args,
            self.fluent.ntype,
        );
        let new_expr = instantiate_expression(
            &self.expression,
            var_mapping,
            fluent_functions,
            init_function_vals,
            task_function_admin,
            new_constant_axioms,
        );
        FunctionAssignment::new(self.symbol.clone(), new_fluent, new_expr)
    }

    pub fn rename_variables(&self, renamings: &HashMap<String, String>) -> FunctionAssignment {
        FunctionAssignment::new(
            self.symbol.clone(),
            PrimitiveNumericExpression::with_type(
                self.fluent.symbol.clone(),
                self.fluent
                    .args
                    .iter()
                    .map(|arg| renamings.get(arg).cloned().unwrap_or_else(|| arg.clone()))
                    .collect(),
                self.fluent.ntype,
            ),
            self.expression.rename_variables(renamings),
        )
    }

    pub fn is_cost_assignment(&self) -> bool {
        self.fluent.symbol == "total-cost"
    }
}

impl fmt::Display for FunctionAssignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FunctionAssignment({}, {}, {})",
            self.symbol, self.fluent, self.expression
        )
    }
}

// ============== Helper methods on FunctionalExpression ==============

impl FunctionalExpression {
    pub fn primitive_numeric_expressions(&self) -> Vec<PrimitiveNumericExpression> {
        match self {
            FunctionalExpression::NumericConstant(_) => vec![],
            FunctionalExpression::PrimitiveNumericExpression(pne) => vec![pne.clone()],
            FunctionalExpression::ArithmeticExpression(ae) => {
                let mut result = vec![];
                for p in &ae.parts {
                    result.extend(p.primitive_numeric_expressions());
                }
                result
            }
            FunctionalExpression::AdditiveInverse(ai) => {
                let mut result = vec![];
                for p in &ai.parts {
                    result.extend(p.primitive_numeric_expressions());
                }
                result
            }
        }
    }

    pub fn rename_variables(&self, renamings: &HashMap<String, String>) -> FunctionalExpression {
        match self {
            FunctionalExpression::NumericConstant(_) => self.clone(),
            FunctionalExpression::PrimitiveNumericExpression(pne) => {
                let new_args = pne
                    .args
                    .iter()
                    .map(|a| renamings.get(a).cloned().unwrap_or_else(|| a.clone()))
                    .collect();
                FunctionalExpression::PrimitiveNumericExpression(
                    PrimitiveNumericExpression::with_type(pne.symbol.clone(), new_args, pne.ntype),
                )
            }
            FunctionalExpression::ArithmeticExpression(ae) => {
                let new_parts = ae
                    .parts
                    .iter()
                    .map(|p| p.rename_variables(renamings))
                    .collect();
                FunctionalExpression::ArithmeticExpression(ArithmeticExpression::new(
                    ae.op.clone(),
                    new_parts,
                ))
            }
            FunctionalExpression::AdditiveInverse(ai) => {
                let new_parts = ai
                    .parts
                    .iter()
                    .map(|p| p.rename_variables(renamings))
                    .collect();
                FunctionalExpression::AdditiveInverse(AdditiveInverse::new(new_parts))
            }
        }
    }

    /// Replaces a nested arithmetic expression by the derived function that
    /// stands for it, so that what is left is flat: a term, or an operator
    /// applied to terms.
    pub fn flattened(
        &self,
        task_function_admin: &mut super::tasks::DerivedFunctionAdministrator,
    ) -> FunctionalExpression {
        let is_term = |part: &FunctionalExpression| {
            matches!(
                part,
                FunctionalExpression::NumericConstant(_)
                    | FunctionalExpression::PrimitiveNumericExpression(_)
            )
        };
        match self {
            FunctionalExpression::NumericConstant(_)
            | FunctionalExpression::PrimitiveNumericExpression(_) => self.clone(),
            FunctionalExpression::ArithmeticExpression(ae) if ae.parts.iter().all(is_term) => {
                self.clone()
            }
            FunctionalExpression::ArithmeticExpression(_)
            | FunctionalExpression::AdditiveInverse(_) => {
                FunctionalExpression::PrimitiveNumericExpression(
                    task_function_admin.get_derived_function(self),
                )
            }
        }
    }
}

impl fmt::Display for FunctionalExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FunctionalExpression::NumericConstant(nc) => write!(f, "{}", nc),
            FunctionalExpression::PrimitiveNumericExpression(pne) => write!(f, "{}", pne),
            FunctionalExpression::ArithmeticExpression(ae) => write!(f, "{}", ae),
            FunctionalExpression::AdditiveInverse(ai) => write!(f, "{}", ai),
        }
    }
}

/// Helper: Instantiate a functional expression
pub fn instantiate_expression(
    expr: &FunctionalExpression,
    var_mapping: &super::tasks::VarMapping,
    fluent_functions: &HashSet<PrimitiveNumericExpression>,
    init_function_vals: &HashMap<PrimitiveNumericExpression, f64>,
    task_function_admin: &mut super::tasks::DerivedFunctionAdministrator,
    new_constant_axioms: &mut Vec<super::axioms::InstantiatedNumericAxiom>,
) -> FunctionalExpression {
    match expr {
        FunctionalExpression::NumericConstant(_) => expr.clone(),
        FunctionalExpression::PrimitiveNumericExpression(pne) => {
            let new_args: Vec<String> = pne
                .args
                .iter()
                .map(|arg| var_mapping.resolve(arg).to_owned())
                .collect();
            let instantiated =
                PrimitiveNumericExpression::with_type(pne.symbol.clone(), new_args, pne.ntype);
            let is_fluent = fluent_functions.contains(&instantiated);
            if !is_fluent && !instantiated.symbol.starts_with("derived!") {
                if let Some(value) = init_function_vals.get(&instantiated) {
                    let constant_expr =
                        FunctionalExpression::NumericConstant(NumericConstant::new(*value));
                    let derived = task_function_admin.get_derived_function(&constant_expr);
                    if let Some(axiom) = task_function_admin.axiom_named(&derived.symbol).cloned() {
                        let instantiated_axiom = axiom.instantiate(
                            &super::tasks::VarMapping::default(),
                            fluent_functions,
                            init_function_vals,
                            task_function_admin,
                            new_constant_axioms,
                        );
                        if !new_constant_axioms.contains(&instantiated_axiom) {
                            new_constant_axioms.push(instantiated_axiom);
                        }
                    }
                    FunctionalExpression::PrimitiveNumericExpression(derived)
                } else {
                    FunctionalExpression::PrimitiveNumericExpression(instantiated)
                }
            } else {
                FunctionalExpression::PrimitiveNumericExpression(instantiated)
            }
        }
        FunctionalExpression::ArithmeticExpression(ae) => {
            let new_parts: Vec<FunctionalExpression> = ae
                .parts
                .iter()
                .map(|p| {
                    instantiate_expression(
                        p,
                        var_mapping,
                        fluent_functions,
                        init_function_vals,
                        task_function_admin,
                        new_constant_axioms,
                    )
                })
                .collect();
            // Check if we need to create a derived function
            let new_expr = FunctionalExpression::ArithmeticExpression(ArithmeticExpression::new(
                ae.op.clone(),
                new_parts,
            ));
            new_expr.flattened(task_function_admin)
        }
        FunctionalExpression::AdditiveInverse(ai) => {
            let new_parts: Vec<FunctionalExpression> = ai
                .parts
                .iter()
                .map(|p| {
                    instantiate_expression(
                        p,
                        var_mapping,
                        fluent_functions,
                        init_function_vals,
                        task_function_admin,
                        new_constant_axioms,
                    )
                })
                .collect();
            let new_expr = FunctionalExpression::AdditiveInverse(AdditiveInverse::new(new_parts));
            new_expr.flattened(task_function_admin)
        }
    }
}
