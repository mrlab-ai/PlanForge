//! Pretty printing for PDDL structures
//! Port of python/translate/pddl_parser/pretty_print.py

use super::SExpr;

pub fn format_sexpr(sexpr: &SExpr, indent: usize) -> String {
    match sexpr {
        SExpr::Atom(atom) => atom.clone(),
        SExpr::List(items) => {
            if items.is_empty() {
                "()".to_string()
            } else if items.len() == 1 {
                format!("({})", format_sexpr(&items[0], indent))
            } else {
                let indent_str = " ".repeat(indent);
                let mut result = "(".to_string();
                
                for (i, item) in items.iter().enumerate() {
                    if i == 0 {
                        result.push_str(&format_sexpr(item, indent + 2));
                    } else {
                        result.push('\n');
                        result.push_str(&indent_str);
                        result.push_str("  ");
                        result.push_str(&format_sexpr(item, indent + 2));
                    }
                }
                
                result.push(')');
                result
            }
        }
    }
}

pub fn pretty_print(sexpr: &SExpr) -> String {
    format_sexpr(sexpr, 0)
}
