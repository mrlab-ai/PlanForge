use ordered_float::OrderedFloat;
/// Functional expression hierarchy for numeric PDDL.
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::tools::OrderedSet;

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

    /// The same fluent with its arguments put through `substitution`.
    pub fn substituted(&self, substitution: &impl super::Substitution) -> Self {
        PrimitiveNumericExpression::with_type(
            self.symbol.clone(),
            super::substitute(&self.args, substitution),
            self.ntype,
        )
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
        new_constant_axioms: &mut OrderedSet<super::axioms::InstantiatedNumericAxiom>,
    ) -> FunctionAssignment {
        let new_fluent = self.fluent.substituted(var_mapping);
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
            self.fluent.substituted(renamings),
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
    /// Adds every variable this expression mentions to `out`.
    ///
    /// A numeric comparison is a leaf of the condition tree but a root of two
    /// expression trees, so a condition cannot find its own free variables
    /// without descending into them: `(> (fuel ?truck) 0)` mentions `?truck`
    /// nowhere else.
    pub fn collect_variables(&self, out: &mut std::collections::BTreeSet<String>) {
        if let FunctionalExpression::PrimitiveNumericExpression(pne) = self {
            out.extend(pne.args.iter().filter(|arg| arg.starts_with('?')).cloned());
        }
        for part in self.parts() {
            part.collect_variables(out);
        }
    }

    /// The operands a compound expression combines, in order; empty for a term,
    /// which is a constant or a primitive numeric expression.
    pub fn parts(&self) -> &[FunctionalExpression] {
        match self {
            FunctionalExpression::ArithmeticExpression(arithmetic) => &arithmetic.parts,
            FunctionalExpression::AdditiveInverse(inverse) => &inverse.parts,
            FunctionalExpression::NumericConstant(_)
            | FunctionalExpression::PrimitiveNumericExpression(_) => &[],
        }
    }

    /// The same kind of expression over `parts`, keeping the arithmetic operator
    /// where there is one. A term has no operands to replace, so handing this
    /// one any is a caller bug.
    pub fn with_parts(&self, parts: Vec<FunctionalExpression>) -> FunctionalExpression {
        match self {
            FunctionalExpression::ArithmeticExpression(arithmetic) => {
                FunctionalExpression::ArithmeticExpression(ArithmeticExpression::new(
                    arithmetic.op.clone(),
                    parts,
                ))
            }
            FunctionalExpression::AdditiveInverse(_) => {
                FunctionalExpression::AdditiveInverse(AdditiveInverse::new(parts))
            }
            term => {
                assert!(parts.is_empty(), "{term} has no operands to replace");
                term.clone()
            }
        }
    }

    /// The same expression with `map` applied to each of its operands. A term is
    /// its own image.
    pub fn map_parts(
        &self,
        map: impl FnMut(&FunctionalExpression) -> FunctionalExpression,
    ) -> FunctionalExpression {
        self.with_parts(self.parts().iter().map(map).collect())
    }

    pub fn primitive_numeric_expressions(&self) -> Vec<PrimitiveNumericExpression> {
        match self {
            FunctionalExpression::PrimitiveNumericExpression(pne) => vec![pne.clone()],
            other => other
                .parts()
                .iter()
                .flat_map(FunctionalExpression::primitive_numeric_expressions)
                .collect(),
        }
    }

    pub fn rename_variables(&self, renamings: &HashMap<String, String>) -> FunctionalExpression {
        match self {
            FunctionalExpression::PrimitiveNumericExpression(pne) => {
                FunctionalExpression::PrimitiveNumericExpression(
                    PrimitiveNumericExpression::with_type(
                        pne.symbol.clone(),
                        crate::pddl::substitute(&pne.args, renamings),
                        pne.ntype,
                    ),
                )
            }
            other => other.map_parts(|part| part.rename_variables(renamings)),
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
    new_constant_axioms: &mut OrderedSet<super::axioms::InstantiatedNumericAxiom>,
) -> FunctionalExpression {
    match expr {
        FunctionalExpression::NumericConstant(_) => expr.clone(),
        FunctionalExpression::PrimitiveNumericExpression(pne) => {
            let instantiated = pne.substituted(var_mapping);
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
                        new_constant_axioms.insert(instantiated_axiom);
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
