//! Derived Function Administrator for Normalization
//!
//! This module implements the full-featured DerivedFunctionAdministrator
//! that matches Python's behavior for the normalization pipeline.
//! This is separate from the lightweight stub used in instantiate.rs.

use crate::translate::function_expression::*;
use crate::translate::pddl_parser::SExpr;
use std::collections::HashMap;

/// Convert a PrimitiveNumericExpression to SExpr
pub fn pne_to_sexpr(pne: &PrimitiveNumericExpression) -> SExpr {
    if pne.args.is_empty() {
        // 0-arity function: just the symbol
        SExpr::Atom(pne.symbol.clone())
    } else {
        // n-arity function: (symbol arg1 arg2 ...)
        let mut items = vec![SExpr::Atom(pne.symbol.clone())];
        for arg in &pne.args {
            items.push(SExpr::Atom(arg.clone()));
        }
        SExpr::List(items)
    }
}

/// A numeric axiom representing a derived function
/// Matches Python's axioms.NumericAxiom
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NumericAxiom {
    pub name: String,
    pub parameters: Vec<String>,
    pub op: Option<String>, // None for constants, Some("+"|"-"|"*"|"/") for operations
    pub parts: Vec<FunctionalExpression>,
}

impl NumericAxiom {
    pub fn new(
        name: String,
        parameters: Vec<String>,
        op: Option<String>,
        parts: Vec<FunctionalExpression>,
    ) -> Self {
        assert!(
            !parts.is_empty(),
            "NumericAxiom must have at least one part"
        );
        NumericAxiom {
            name,
            parameters,
            op,
            parts,
        }
    }

    /// Get the head of this axiom as a PrimitiveNumericExpression
    pub fn get_head(&self) -> PrimitiveNumericExpression {
        let ntype = if self.op.is_some() { 'D' } else { 'C' };
        PrimitiveNumericExpression::new(self.name.clone(), self.parameters.clone(), ntype)
    }

    pub fn dump(&self, indent: &str) {
        let head = format!("({} {})", self.name, self.parameters.join(", "));
        let op = self
            .op
            .as_ref()
            .map(|s| format!("{} ", s))
            .unwrap_or_default();
        let body = self
            .parts
            .iter()
            .map(|p| format!("{}", p))
            .collect::<Vec<_>>()
            .join(" ");
        println!("{}{} -: {}{}", indent, head, op, body);
    }
}

/// Key type for caching derived functions
/// Matches Python's tuple-based keys
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DerivedFunctionKey {
    /// Constant: (value,)
    Constant(i64),
    /// Additive inverse: (op, part_string)
    AdditiveInverse(String, String),
    /// Arithmetic: (op, [part_strings...])
    Arithmetic(String, Vec<String>),
}

/// Full-featured Derived Function Administrator for normalization
/// Matches Python's DerivedFunctionAdministrator behavior exactly
#[derive(Debug, Clone)]
pub struct NormalizationFunctionAdministrator {
    /// Maps expression keys to NumericAxiom objects
    functions: HashMap<DerivedFunctionKey, NumericAxiom>,
    /// Counter for generating unique names if needed
    #[allow(dead_code)]
    counter: usize,
}

impl NormalizationFunctionAdministrator {
    pub fn new() -> Self {
        NormalizationFunctionAdministrator {
            functions: HashMap::new(),
            counter: 0,
        }
    }

    fn pne_to_string(pne: &PrimitiveNumericExpression) -> String {
        if pne.args.is_empty() {
            pne.symbol.clone()
        } else {
            format!("{}({})", pne.symbol, pne.args.join(", "))
        }
    }

    fn make_unique_symbol(&self, base: &str) -> String {
        let used_names: std::collections::HashSet<String> =
            self.functions.values().map(|ax| ax.name.clone()).collect();
        if !used_names.contains(base) {
            return base.to_string();
        }
        let mut counter = 2;
        loop {
            let candidate = format!("{}_{}", base, counter);
            if !used_names.contains(&candidate) {
                return candidate;
            }
            counter += 1;
        }
    }

    /// Get all axioms created by this administrator
    pub fn get_all_axioms(&self) -> Vec<NumericAxiom> {
        self.functions.values().cloned().collect()
    }

    /// Dump all axioms (for debugging)
    pub fn dump(&self, indent: &str) {
        for axiom in self.functions.values() {
            axiom.dump(indent);
        }
    }

    /// Get or create a derived function for the given expression
    /// Returns a PrimitiveNumericExpression representing the derived function
    pub fn get_derived_function(
        &mut self,
        exp: &FunctionalExpression,
    ) -> PrimitiveNumericExpression {
        match exp {
            // Case 1: Already a PrimitiveNumericExpression - return as-is
            FunctionalExpression::Primitive(pne) => pne.clone(),

            // Case 2: NumericConstant - create axiom if needed
            FunctionalExpression::Constant(nc) => {
                let key = DerivedFunctionKey::Constant(nc.value);

                if !self.functions.contains_key(&key) {
                    let symbol = format!("derived!{}.0()", nc.value);
                    let axiom = NumericAxiom::new(
                        symbol,
                        vec![],
                        None,
                        vec![FunctionalExpression::Constant(nc.clone())],
                    );
                    self.functions.insert(key.clone(), axiom);
                }

                self.functions.get(&key).unwrap().get_head()
            }

            // Case 3: AdditiveInverse (unary minus)
            FunctionalExpression::AdditiveInverse(ai) => {
                // Recursively process the sub-expression
                let subexp = self.get_derived_function(&ai.part);
                let subexp_str = Self::pne_to_string(&subexp);
                let key = DerivedFunctionKey::AdditiveInverse(ai.op.clone(), subexp_str.clone());

                if !self.functions.contains_key(&key) {
                    let base = format!("derived!difference_PNE {}", subexp_str);
                    let symbol = self.make_unique_symbol(&base);

                    // Generate default variables for parameters
                    let default_args = self.get_default_variables(subexp.args.len());

                    // Create PNE with default args
                    let pne = PrimitiveNumericExpression::new(
                        subexp.symbol.clone(),
                        default_args.clone(),
                        'S',
                    );

                    let axiom = NumericAxiom::new(
                        symbol,
                        default_args,
                        Some(ai.op.clone()),
                        vec![FunctionalExpression::Primitive(pne)],
                    );
                    self.functions.insert(key.clone(), axiom);
                }

                let axiom = self.functions.get(&key).unwrap();
                PrimitiveNumericExpression::new(axiom.get_head().symbol, subexp.args, 'D')
            }

            // Case 4: ArithmeticExpression (binary ops)
            FunctionalExpression::Arithmetic(ae) => {
                // Recursively process all parts
                let mut df_parts: Vec<PrimitiveNumericExpression> = Vec::new();
                for part in &ae.parts {
                    df_parts.push(self.get_derived_function(part));
                }

                // For commutative operations, sort the parts for canonicalization
                let mut part_pairs: Vec<(String, PrimitiveNumericExpression)> = df_parts
                    .into_iter()
                    .map(|p| (Self::pne_to_string(&p), p))
                    .collect();
                if ae.op == "+" || ae.op == "*" {
                    part_pairs.sort_by(|a, b| a.0.cmp(&b.0));
                }
                let part_strings: Vec<String> = part_pairs.iter().map(|p| p.0.clone()).collect();
                let df_parts: Vec<PrimitiveNumericExpression> =
                    part_pairs.into_iter().map(|p| p.1).collect();

                // Build key: (op, [part_strings...])
                let key = DerivedFunctionKey::Arithmetic(ae.op.clone(), part_strings.clone());

                // Collect all args
                let mut all_args = Vec::new();
                for df in &df_parts {
                    all_args.extend(df.args.clone());
                }

                if !self.functions.contains_key(&key) {
                    // Generate new symbol using PNE-style naming
                    let op_name = match ae.op.as_str() {
                        "+" => "sum_PNE",
                        "-" => "difference_PNE",
                        "*" => "product_PNE",
                        "/" => "quotient_PNE",
                        _ => "op_PNE",
                    };
                    let base = if part_strings.is_empty() {
                        format!("derived!{}", op_name)
                    } else {
                        format!("derived!{} {}", op_name, part_strings.join("_PNE "))
                    };
                    let symbol = self.make_unique_symbol(&base);

                    // Generate default variables
                    let default_args = self.get_default_variables(all_args.len());

                    // Build PNE list with sliced arguments
                    let mut argindex = 0;
                    let mut pnelist = Vec::new();
                    for df in &df_parts {
                        let arg_slice: Vec<String> =
                            default_args[argindex..argindex + df.args.len()].to_vec();
                        pnelist.push(FunctionalExpression::Primitive(
                            PrimitiveNumericExpression::new(df.symbol.clone(), arg_slice, 'D'),
                        ));
                        argindex += df.args.len();
                    }

                    let axiom =
                        NumericAxiom::new(symbol, default_args, Some(ae.op.clone()), pnelist);
                    self.functions.insert(key.clone(), axiom);
                }

                let axiom = self.functions.get(&key).unwrap();
                PrimitiveNumericExpression::new(axiom.get_head().symbol, all_args, 'D')
            }
        }
    }

    /// Generate default variable names: ?v0, ?v1, ?v2, ...
    fn get_default_variables(&self, nr: usize) -> Vec<String> {
        (0..nr).map(|i| format!("?v{}", i)).collect()
    }

    /// Generate a new unique symbol name
    /// Matches Python's get_new_symbol behavior
    fn get_new_symbol(&mut self, key_parts: &[String]) -> String {
        let used_names: Vec<String> = self.functions.values().map(|ax| ax.name.clone()).collect();

        // Build addition string from key parts
        let addition = key_parts
            .iter()
            .map(|part| self.prettyprint(part))
            .collect::<Vec<_>>()
            .join("_");

        let mut counter = 1;
        loop {
            let new_func_name = if counter == 1 {
                format!("derived!{}", addition)
            } else {
                format!("derived!{}_{}", addition, counter)
            };

            if !used_names.contains(&new_func_name) {
                return new_func_name;
            }
            counter += 1;
        }
    }

    /// Pretty-print operator names
    fn prettyprint(&self, s: &str) -> String {
        match s {
            "-" => "difference".to_string(),
            "+" => "sum".to_string(),
            "*" => "product".to_string(),
            "/" => "quotient".to_string(),
            _ => s.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant() {
        let mut admin = NormalizationFunctionAdministrator::new();
        let constant = FunctionalExpression::Constant(NumericConstant::new(42));
        let pne = admin.get_derived_function(&constant);

        assert!(pne.symbol.contains("derived!"));
        assert!(pne.symbol.contains("42"));
        assert_eq!(pne.args.len(), 0);
        assert_eq!(admin.get_all_axioms().len(), 1);
    }

    #[test]
    fn test_primitive_passthrough() {
        let mut admin = NormalizationFunctionAdministrator::new();
        let pne = PrimitiveNumericExpression::new("fuel".to_string(), vec!["?v".to_string()], 'S');
        let result = admin.get_derived_function(&FunctionalExpression::Primitive(pne.clone()));

        assert_eq!(result.symbol, "fuel");
        assert_eq!(result.args, vec!["?v"]);
        assert_eq!(admin.get_all_axioms().len(), 0); // No axiom created
    }

    #[test]
    fn test_additive_inverse() {
        let mut admin = NormalizationFunctionAdministrator::new();
        let pne = PrimitiveNumericExpression::new("fuel".to_string(), vec!["?v".to_string()], 'S');
        let inv = AdditiveInverse::new(FunctionalExpression::Primitive(pne));
        let result = admin.get_derived_function(&FunctionalExpression::AdditiveInverse(inv));

        assert!(result.symbol.contains("derived!"));
        assert!(result.symbol.contains("difference"));
        assert_eq!(result.args, vec!["?v"]);
        assert_eq!(admin.get_all_axioms().len(), 1);
    }

    #[test]
    fn test_arithmetic_sum() {
        let mut admin = NormalizationFunctionAdministrator::new();
        let pne1 = PrimitiveNumericExpression::new("fuel".to_string(), vec![], 'S');
        let pne2 = PrimitiveNumericExpression::new("distance".to_string(), vec![], 'S');

        let sum = ArithmeticExpression::new(
            "+".to_string(),
            vec![
                FunctionalExpression::Primitive(pne1),
                FunctionalExpression::Primitive(pne2),
            ],
        );

        let result = admin.get_derived_function(&FunctionalExpression::Arithmetic(sum));

        assert!(result.symbol.contains("derived!"));
        assert!(result.symbol.contains("sum"));
        assert_eq!(admin.get_all_axioms().len(), 1);
    }

    #[test]
    fn test_caching() {
        let mut admin = NormalizationFunctionAdministrator::new();
        let constant = FunctionalExpression::Constant(NumericConstant::new(42));

        let pne1 = admin.get_derived_function(&constant);
        let pne2 = admin.get_derived_function(&constant);

        // Should return the same symbol (caching works)
        assert_eq!(pne1.symbol, pne2.symbol);
        assert_eq!(admin.get_all_axioms().len(), 1); // Only one axiom created
    }

    #[test]
    fn test_commutative_canonicalization() {
        let mut admin = NormalizationFunctionAdministrator::new();
        let pne1 = PrimitiveNumericExpression::new("a".to_string(), vec![], 'S');
        let pne2 = PrimitiveNumericExpression::new("b".to_string(), vec![], 'S');

        // Create a + b
        let sum1 = ArithmeticExpression::new(
            "+".to_string(),
            vec![
                FunctionalExpression::Primitive(pne1.clone()),
                FunctionalExpression::Primitive(pne2.clone()),
            ],
        );

        // Create b + a (should be same due to sorting)
        let sum2 = ArithmeticExpression::new(
            "+".to_string(),
            vec![
                FunctionalExpression::Primitive(pne2),
                FunctionalExpression::Primitive(pne1),
            ],
        );

        let result1 = admin.get_derived_function(&FunctionalExpression::Arithmetic(sum1));
        let result2 = admin.get_derived_function(&FunctionalExpression::Arithmetic(sum2));

        // Should be the same symbol due to canonicalization
        assert_eq!(result1.symbol, result2.symbol);
        assert_eq!(admin.get_all_axioms().len(), 1);
    }
}
