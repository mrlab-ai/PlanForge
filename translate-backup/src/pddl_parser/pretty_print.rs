#[cfg(test)]
mod tests;

use crate::translate::pddl_parser::SExpr;

pub fn tokenize_list(obj: &SExpr) -> Vec<String> {
    let mut out = Vec::new();
    tokenize_list_into(obj, &mut out);
    out
}

fn tokenize_list_into(obj: &SExpr, out: &mut Vec<String>) {
    match obj {
        SExpr::Atom(atom) => out.push(atom.clone()),
        SExpr::List(items) => {
            out.push("(".to_string());
            for item in items {
                tokenize_list_into(item, out);
            }
            out.push(")".to_string());
        }
    }
}

pub fn wrap_lines(lines: &[String]) -> Vec<String> {
    lines.to_vec()
}

pub fn print_nested_list(nested_list: &SExpr) -> String {
    let mut stream = String::new();
    let mut indent = 0usize;
    let mut start_of_line = true;
    let mut pending_space = false;

    for token in tokenize_list(nested_list) {
        if token == "(" {
            if !start_of_line {
                stream.push('\n');
            }
            stream.push_str(&" ".repeat(indent));
            stream.push('(');
            indent += 2;
            start_of_line = false;
            pending_space = false;
        } else if token == ")" {
            indent = indent.saturating_sub(2);
            stream.push(')');
            start_of_line = false;
            pending_space = false;
        } else {
            if start_of_line {
                stream.push_str(&" ".repeat(indent));
            }
            if pending_space {
                stream.push(' ');
            }
            stream.push_str(&token);
            start_of_line = false;
            pending_space = true;
        }
    }

    wrap_lines(&stream.lines().map(str::to_string).collect::<Vec<_>>()).join("\n")
}
