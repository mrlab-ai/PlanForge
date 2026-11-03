//! Numeric PDDL expressions: constants, fluents, arithmetic and the
//! assignments of numeric effects.

use planforge_sas::numeric_task::NumericValue;
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
    pub value: NumericValue,
}

impl NumericConstant {
    pub fn new(value: f64) -> Self {
        NumericConstant {
            value: NumericValue::new(value),
        }
    }
}

impl fmt::Display for NumericConstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = self.value.value();
        if value.fract() == 0.0 {
            write!(f, "{:.1}", value)
        } else {
            write!(f, "{}", value)
        }
    }
}

/// What a numeric fluent is. The variants are in the order of the SAS
/// letters, which is the order fluents of equal name sort in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FluentKind {
    /// `C`: a value no operator changes.
    Constant,
    /// `D`: a value numeric axioms derive from others.
    Derived,
    /// `I`: the instrumentation fluent `total-cost`, which only accumulates
    /// action costs.
    Instrumentation,
    /// `R`: a fluent operators change.
    Regular,
}

impl FluentKind {
    /// The letter the SAS format writes for the kind.
    pub fn sas_letter(self) -> char {
        match self {
            Self::Constant => 'C',
            Self::Derived => 'D',
            Self::Instrumentation => 'I',
            Self::Regular => 'R',
        }
    }

    /// Which of two kinds the same fluent was seen with it is: one that
    /// operators change outranks a derived one, which outranks a constant,
    /// and the instrumentation fluent outranks them all.
    pub fn merge_rank(self) -> u8 {
        match self {
            Self::Constant => 1,
            Self::Derived => 2,
            Self::Regular => 3,
            Self::Instrumentation => 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrimitiveNumericExpression {
    pub symbol: String,
    pub args: Vec<String>,
    pub kind: FluentKind,
}

/// The name of the fluent PDDL accumulates action costs in.
pub const TOTAL_COST: &str = "total-cost";

impl PrimitiveNumericExpression {
    /// The action-cost fluent `(total-cost)`.
    pub fn total_cost() -> Self {
        Self::with_kind(TOTAL_COST.to_string(), vec![], FluentKind::Instrumentation)
    }

    pub fn new(symbol: String, args: Vec<String>) -> Self {
        PrimitiveNumericExpression {
            symbol,
            args,
            kind: FluentKind::Regular,
        }
    }

    pub fn with_kind(symbol: String, args: Vec<String>, kind: FluentKind) -> Self {
        PrimitiveNumericExpression { symbol, args, kind }
    }

    /// The same fluent with its arguments put through `substitution`.
    pub fn substituted(&self, substitution: &impl super::Substitution) -> Self {
        PrimitiveNumericExpression::with_kind(
            self.symbol.clone(),
            super::substitute(&self.args, substitution),
            self.kind,
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
        init_function_vals: &HashMap<PrimitiveNumericExpression, NumericValue>,
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

    /// Whether this is an action cost: an increase of `total-cost`. Any other
    /// effect on `total-cost` is an ordinary numeric effect.
    pub fn is_cost_assignment(&self) -> bool {
        self.symbol == "+" && self.fluent.symbol == TOTAL_COST
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
                    PrimitiveNumericExpression::with_kind(
                        pne.symbol.clone(),
                        crate::pddl::substitute(&pne.args, renamings),
                        pne.kind,
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

/// The derived function that stands for the constant `value`, with its
/// defining axiom grounded into `new_constant_axioms`.
pub fn constant_function(
    value: NumericValue,
    fluent_functions: &HashSet<PrimitiveNumericExpression>,
    init_function_vals: &HashMap<PrimitiveNumericExpression, NumericValue>,
    task_function_admin: &mut super::tasks::DerivedFunctionAdministrator,
    new_constant_axioms: &mut OrderedSet<super::axioms::InstantiatedNumericAxiom>,
) -> PrimitiveNumericExpression {
    let constant_expr = FunctionalExpression::NumericConstant(NumericConstant { value });
    let derived = task_function_admin.get_derived_function(&constant_expr);
    let axiom = task_function_admin
        .axiom_named(&derived.symbol)
        .cloned()
        .expect("a constant's derived function is defined by an axiom");
    let instantiated_axiom = axiom.instantiate(
        &super::tasks::VarMapping::default(),
        fluent_functions,
        init_function_vals,
        task_function_admin,
        new_constant_axioms,
    );
    new_constant_axioms.insert(instantiated_axiom);
    derived
}

/// Helper: Instantiate a functional expression
pub fn instantiate_expression(
    expr: &FunctionalExpression,
    var_mapping: &super::tasks::VarMapping,
    fluent_functions: &HashSet<PrimitiveNumericExpression>,
    init_function_vals: &HashMap<PrimitiveNumericExpression, NumericValue>,
    task_function_admin: &mut super::tasks::DerivedFunctionAdministrator,
    new_constant_axioms: &mut OrderedSet<super::axioms::InstantiatedNumericAxiom>,
) -> FunctionalExpression {
    match expr {
        FunctionalExpression::NumericConstant(_) => expr.clone(),
        FunctionalExpression::PrimitiveNumericExpression(pne) => {
            let instantiated = pne.substituted(var_mapping);
            let is_fluent = fluent_functions.contains(&instantiated);
            if !is_fluent && !instantiated.symbol.starts_with("derived!") {
                if let Some(&value) = init_function_vals.get(&instantiated) {
                    FunctionalExpression::PrimitiveNumericExpression(constant_function(
                        value,
                        fluent_functions,
                        init_function_vals,
                        task_function_admin,
                        new_constant_axioms,
                    ))
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
