use crate::translate::pddl_parser::SExpr;
use crate::translate::numeric_axiom_rules::PrimitiveNumericExpression;
use std::collections::HashMap;

/// Minimal DerivedFunctionAdministrator: canonicalize simple arithmetic
/// expressions and generate derived PNE names. This is a light-weight
/// helper to start matching Python behavior.

pub struct DerivedFunctionAdministrator {
    // map from canonical key -> (symbol, args)
    pub functions: HashMap<String, (String, Vec<String>)>,
}

impl DerivedFunctionAdministrator {
    pub fn new() -> Self {
    DerivedFunctionAdministrator { functions: HashMap::new() }
    }

    /// Given an SExpr that represents either a primitive PNE, a numeric constant,
    /// or an arithmetic expression, return a PrimitiveNumericExpression-like
    /// object describing the PNE symbol and its args. For derived expressions
    /// this will register a derived symbol.
    pub fn get_derived_function(&mut self, exp: &SExpr) -> PrimitiveNumericExpression {
        match exp {
            SExpr::Atom(a) => {
                // numeric constant -> canonical derived constant name like derived!4.0()
                if let Ok(nv) = a.parse::<i64>() {
                    PrimitiveNumericExpression { name: format!("derived!{}.0()", nv), args: vec![] }
                } else {
                    // plain atom treated as primitive PNE name (no args)
                    PrimitiveNumericExpression { name: a.clone(), args: vec![] }
                }
            }
            SExpr::List(list) => {
                if list.is_empty() { return PrimitiveNumericExpression { name: "".to_string(), args: vec![] }; }
                if let SExpr::Atom(op) = &list[0] {
                    // arithmetic operators: build child primitives and create canonical symbol with placeholders
                    if op == "+" || op == "-" || op == "*" || op == "/" {
                        // build keylist like Python: [op, child1_symbol, child2_symbol, ...]
                        let mut keylist: Vec<String> = Vec::new();
                        keylist.push(op.clone());
                        let mut child_pnes: Vec<PrimitiveNumericExpression> = Vec::new();
                        for p in &list[1..] {
                            let pne = self.get_derived_function(p);
                            keylist.push(pne.name.clone());
                            child_pnes.push(pne);
                        }
                        // for commutative ops, sort child symbols to canonicalize
                        if op == "+" || op == "*" {
                            keylist[1..].sort();
                        }
                        let key = keylist.join("|");
                        // compute args (placeholders) as ?v0..?vN where N is total args from children
                        let mut total_args: Vec<String> = Vec::new();
                        for child in &child_pnes {
                            for _ in 0..child.args.len() { total_args.push(String::new()); }
                        }
                        // compute default placeholder names for number of args
                        let arg_count = child_pnes.iter().map(|c| c.args.len()).sum();
                        let mut placeholders: Vec<String> = Vec::new();
                        for i in 0..arg_count { placeholders.push(format!("?v{}", i)); }

                        if !self.functions.contains_key(&key) {
                            // generate new function name similar to Python's prettyprint concatenation
                            let pretty = match op.as_str() {
                                "+" => "sum",
                                "*" => "product",
                                "-" => "difference",
                                "/" => "division",
                                _ => "op",
                            };
                            let new_name = format!("derived!{}_PNE", pretty);
                            self.functions.insert(key.clone(), (new_name.clone(), placeholders.clone()));
                        }
                        let (sym, args) = self.functions.get(&key).unwrap().clone();
                        PrimitiveNumericExpression { name: sym, args }
                    } else {
                        // treat as primitive PNE, name(args...)
                        let args = list[1..].iter().filter_map(|x| match x { SExpr::Atom(a)=>Some(a.clone()), _=>None }).collect::<Vec<_>>();
                        let key = format!("{}({})", op, args.join(","));
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
