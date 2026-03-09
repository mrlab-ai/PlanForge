/// Port of pddl_parser/lisp_parser.py
/// Simple S-expression parser that tokenizes and builds nested lists.
use std::fs;
use std::path::Path;

/// A parsed S-expression: either an atom (string) or a list of S-expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum SExpr {
    Atom(String),
    List(Vec<SExpr>),
}

impl SExpr {
    /// Get as a string atom, panicking if it's a list.
    pub fn as_atom(&self) -> &str {
        match self {
            SExpr::Atom(s) => s,
            SExpr::List(_) => panic!("Expected atom, got list: {:?}", self),
        }
    }

    /// Get as a list, panicking if it's an atom.
    pub fn as_list(&self) -> &[SExpr] {
        match self {
            SExpr::List(l) => l,
            SExpr::Atom(s) => panic!("Expected list, got atom: {}", s),
        }
    }

    /// Check if this is a list.
    pub fn is_list(&self) -> bool {
        matches!(self, SExpr::List(_))
    }

    /// Check if this is an atom.
    pub fn is_atom(&self) -> bool {
        matches!(self, SExpr::Atom(_))
    }

    /// If this is a list, get its length; otherwise 0.
    pub fn len(&self) -> usize {
        match self {
            SExpr::List(l) => l.len(),
            SExpr::Atom(_) => 0,
        }
    }
}

impl std::fmt::Display for SExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SExpr::Atom(s) => write!(f, "{}", s),
            SExpr::List(l) => {
                write!(f, "(")?;
                for (i, item) in l.iter().enumerate() {
                    if i > 0 { write!(f, " ")?; }
                    write!(f, "{}", item)?;
                }
                write!(f, ")")
            }
        }
    }
}

/// Python: def parse_nested_list(input_file)
/// Reads a PDDL file and parses it into an S-expression.
pub fn parse_nested_list(filepath: &Path) -> Result<SExpr, String> {
    let content = fs::read_to_string(filepath)
        .map_err(|e| format!("Could not read file {}: {}", filepath.display(), e))?;
    parse_nested_list_string(&content)
}

/// Parse a string into an S-expression.
pub fn parse_nested_list_string(input: &str) -> Result<SExpr, String> {
    let tokens = tokenize(input);
    let mut iter = tokens.into_iter().peekable();
    let result = parse_list_aux(&mut iter)?;
    if iter.peek().is_some() {
        return Err("Unexpected tokens after end of list".to_string());
    }
    Ok(result)
}

/// Python: def tokenize(input)
/// Splits input into tokens: "(", ")", or atoms. Strips comments (lines starting with ;).
fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = vec![];
    for line in input.lines() {
        // Strip comments
        let line = if let Some(pos) = line.find(';') {
            &line[..pos]
        } else {
            line
        };
        // Add whitespace around parens for splitting
        let line = line.replace('(', " ( ").replace(')', " ) ");
        for token in line.split_whitespace() {
            tokens.push(token.to_lowercase());
        }
    }
    tokens
}

/// Python: def parse_list_aux(tokenstream)
/// Recursively parses a token stream into an S-expression.
fn parse_list_aux(tokens: &mut std::iter::Peekable<std::vec::IntoIter<String>>) -> Result<SExpr, String> {
    let token = tokens.next().ok_or("Unexpected end of input")?;
    if token == "(" {
        let mut result = vec![];
        loop {
            match tokens.peek() {
                Some(t) if t == ")" => {
                    tokens.next();
                    break;
                }
                Some(_) => {
                    result.push(parse_list_aux(tokens)?);
                }
                None => return Err("Missing closing parenthesis".to_string()),
            }
        }
        Ok(SExpr::List(result))
    } else if token == ")" {
        Err("Unexpected closing parenthesis".to_string())
    } else {
        Ok(SExpr::Atom(token))
    }
}
