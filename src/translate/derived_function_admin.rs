use crate::translate::pddl_parser::SExpr;
use crate::translate::numeric_axiom_rules::PrimitiveNumericExpression;
use std::collections::HashMap;

/// Minimal DerivedFunctionAdministrator: canonicalize simple arithmetic
/// expressions and generate derived PNE names. This is a light-weight
/// helper to start matching Python behavior.

pub struct DerivedFunctionAdministrator {
    // map from canonical key -> (symbol, args)
    pub functions: HashMap<String, (String, Vec<String>)>,
    counter: usize,
}

impl DerivedFunctionAdministrator {
    pub fn new() -> Self {
        DerivedFunctionAdministrator { functions: HashMap::new(), counter: 0 }
    }

    /// Given an SExpr that represents either a primitive PNE, a numeric constant,
    /// or an arithmetic expression, return a PrimitiveNumericExpression-like
    /// object describing the PNE symbol and its args. For derived expressions
    /// this will register a derived symbol.
    pub fn get_derived_function(&mut self, exp: &SExpr) -> PrimitiveNumericExpression {
        match exp {
            SExpr::Atom(a) => {
                if let Ok(_n) = a.parse::<i64>() {
                    // numeric constant represented as const:val
                    PrimitiveNumericExpression { name: format!("const:{}", a), args: vec![] }
                } else {
                    // primitive PNE like (f a b) represented by key "f(a, b)"
                    PrimitiveNumericExpression { name: a.clone(), args: vec![] }
                }
            }
            SExpr::List(list) => {
                if list.is_empty() { return PrimitiveNumericExpression { name: "".to_string(), args: vec![] }; }
                if let SExpr::Atom(op) = &list[0] {
                    if op == "+" || op == "*" {
                        // commutative: sort parts by string form
                        let mut parts: Vec<String> = list[1..].iter().map(|p| match p { SExpr::Atom(a)=>a.clone(), SExpr::List(_)=>format!("{}", Self::sexpr_to_string(p)), }).collect();
                        parts.sort();
                        let key = format!("({} {})", op, parts.join(" "));
                        if let Some((sym, args)) = self.functions.get(&key) {
                            return PrimitiveNumericExpression { name: sym.clone(), args: args.clone() };
                        }
                        let symbol = format!("derived!{}", self.counter);
                        self.counter += 1;
                        let args: Vec<String> = vec![]; // placeholder
                        self.functions.insert(key.clone(), (symbol.clone(), args.clone()));
                        PrimitiveNumericExpression { name: symbol, args }
                    } else if op == "-" || op == "/" {
                        let parts: Vec<String> = list[1..].iter().map(|p| Self::sexpr_to_string(p)).collect();
                        let key = format!("({} {})", op, parts.join(" "));
                        if let Some((sym, args)) = self.functions.get(&key) {
                            return PrimitiveNumericExpression { name: sym.clone(), args: args.clone() };
                        }
                        let symbol = format!("derived!{}", self.counter);
                        self.counter += 1;
                        let args: Vec<String> = vec![];
                        self.functions.insert(key.clone(), (symbol.clone(), args.clone()));
                        PrimitiveNumericExpression { name: symbol, args }
                    } else {
                        // treat as primitive PNE, name(args...)
                        let args = list[1..].iter().filter_map(|x| match x { SExpr::Atom(a)=>Some(a.clone()), _=>None }).collect::<Vec<_>>();
                        let key = format!("{}({})", op, args.join(", "));
                        PrimitiveNumericExpression { name: key.clone(), args }
                    }
                } else {
                    PrimitiveNumericExpression { name: Self::sexpr_to_string(exp), args: vec![] }
                }
            }
        }
    }

    fn sexpr_to_string(s: &SExpr) -> String {
        match s {
            SExpr::Atom(a) => a.clone(),
            SExpr::List(list) => {
                let parts: Vec<String> = list.iter().map(|p| Self::sexpr_to_string(p)).collect();
                format!("({})", parts.join(" "))
            }
        }
    }
}
