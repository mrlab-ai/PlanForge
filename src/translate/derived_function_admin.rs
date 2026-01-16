use crate::translate::numeric_axiom_rules::PrimitiveNumericExpression;
use crate::translate::pddl_parser::SExpr;
use std::collections::HashMap;

/// Minimal DerivedFunctionAdministrator: canonicalize simple arithmetic
/// expressions and generate derived PNE names. This is a light-weight
/// helper to start matching Python behavior.

pub struct DerivedFunctionAdministrator {
    // map from canonical key -> (symbol, args)
    pub functions: HashMap<String, (String, Vec<String>)>,
    #[allow(dead_code)]
    counter: usize,
}

impl DerivedFunctionAdministrator {
    pub fn new() -> Self {
        DerivedFunctionAdministrator {
            functions: HashMap::new(),
            counter: 0,
        }
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
                    PrimitiveNumericExpression {
                        name: format!("derived!{}.0()", nv),
                        args: vec![],
                    }
                } else {
                    // plain atom treated as primitive PNE name (no args)
                    PrimitiveNumericExpression {
                        name: a.clone(),
                        args: vec![],
                    }
                }
            }
            SExpr::List(list) => {
                if list.is_empty() {
                    return PrimitiveNumericExpression {
                        name: "".to_string(),
                        args: vec![],
                    };
                }
                if let SExpr::Atom(op) = &list[0] {
                    // arithmetic operators: build child PNE tokens and return operator-style name + args
                    if op == "+" || op == "-" || op == "*" || op == "/" {
                        // collect child tokens using recursive calls
                        let mut child_tokens: Vec<String> = Vec::new();
                        let mut child_args: Vec<String> = Vec::new();
                        for p in &list[1..] {
                            let pne = self.get_derived_function(p);
                            // token form: if child is a derived symbol keep it, otherwise append _PNE to its name
                            let token = if pne.name.starts_with("derived!") {
                                pne.name.clone()
                            } else {
                                format!("{}{}_PNE", pne.name, "")
                            };
                            child_tokens.push(token);
                            // keep the underlying primitive name (without _PNE) as arg for effect representation
                            child_args.push(pne.name.clone());
                        }
                        if op == "+" || op == "*" {
                            child_tokens.sort();
                            child_args.sort();
                        }
                        let op_name = match op.as_str() {
                            "+" => "derived!sum_PNE",
                            "*" => "derived!product_PNE",
                            "-" => "derived!difference_PNE",
                            "/" => "derived!division_PNE",
                            _ => "derived!op_PNE",
                        };
                        PrimitiveNumericExpression {
                            name: op_name.to_string(),
                            args: child_tokens,
                        }
                    } else {
                        // treat as primitive PNE, name(args...)
                        let args = list[1..]
                            .iter()
                            .filter_map(|x| match x {
                                SExpr::Atom(a) => Some(a.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        let key = format!("{}({})", op, args.join(", "));
                        PrimitiveNumericExpression {
                            name: key.clone(),
                            args,
                        }
                    }
                } else {
                    PrimitiveNumericExpression {
                        name: Self::sexpr_to_string(exp),
                        args: vec![],
                    }
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
