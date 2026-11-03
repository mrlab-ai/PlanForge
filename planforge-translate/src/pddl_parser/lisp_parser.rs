//! S-expressions with the line they start on, and the errors parsing PDDL
//! reports.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

/// A parsed S-expression and the line (1-based) it starts on.
#[derive(Debug, Clone, PartialEq)]
pub enum SExpr {
    Atom(String, u32),
    List(Vec<SExpr>, u32),
}

/// Why PDDL input could not be translated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseErrorKind {
    /// The input is not well-formed PDDL.
    Malformed,
    /// The input is valid PDDL this translator does not implement.
    Unsupported,
}

/// An error in PDDL input: what is wrong, and where.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub file: Option<PathBuf>,
    pub line: Option<u32>,
    pub message: String,
}

impl ParseError {
    pub fn malformed(line: u32, message: impl Into<String>) -> Self {
        Self {
            kind: ParseErrorKind::Malformed,
            file: None,
            line: Some(line),
            message: message.into(),
        }
    }

    pub fn unsupported(line: u32, message: impl Into<String>) -> Self {
        Self {
            kind: ParseErrorKind::Unsupported,
            ..Self::malformed(line, message)
        }
    }

    /// The error, located in `file`.
    pub fn in_file(self, file: &Path) -> Self {
        Self {
            file: Some(file.to_path_buf()),
            ..self
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(file) = &self.file {
            write!(f, "{}:", file.display())?;
        }
        if let Some(line) = self.line {
            write!(f, "{line}: ")?;
        } else if self.file.is_some() {
            write!(f, " ")?;
        }
        if self.kind == ParseErrorKind::Unsupported {
            write!(f, "unsupported: ")?;
        }
        f.write_str(&self.message)
    }
}

impl std::error::Error for ParseError {}

pub type ParseResult<T> = Result<T, ParseError>;

impl SExpr {
    /// The line the expression starts on.
    pub fn line(&self) -> u32 {
        match self {
            SExpr::Atom(_, line) | SExpr::List(_, line) => *line,
        }
    }

    pub fn atom(&self) -> Option<&str> {
        match self {
            SExpr::Atom(text, _) => Some(text),
            SExpr::List(..) => None,
        }
    }

    /// The atom this is, or an error naming `what` was expected.
    pub fn expect_atom(&self, what: &str) -> ParseResult<&str> {
        match self {
            SExpr::Atom(text, _) => Ok(text),
            SExpr::List(..) => Err(ParseError::malformed(
                self.line(),
                format!("expected {what}, got the list {self}"),
            )),
        }
    }

    /// The list this is, or an error naming `what` was expected.
    pub fn expect_list(&self, what: &str) -> ParseResult<Form<'_>> {
        match self {
            SExpr::List(items, line) => Ok(Form { items, line: *line }),
            SExpr::Atom(text, line) => Err(ParseError::malformed(
                *line,
                format!("expected {what}, got the atom {text}"),
            )),
        }
    }
}

/// The items of a list, with the line the list starts on for errors about
/// items it lacks.
#[derive(Debug, Clone, Copy)]
pub struct Form<'a> {
    pub items: &'a [SExpr],
    pub line: u32,
}

impl<'a> Form<'a> {
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Item `index`, or an error naming `what` is missing.
    pub fn get(&self, index: usize, what: &str) -> ParseResult<&'a SExpr> {
        self.items.get(index).ok_or_else(|| {
            ParseError::malformed(self.line, format!("expected {what}, but the list ends"))
        })
    }

    /// The items from `index` on, which are none past the end.
    pub fn rest(&self, index: usize) -> Form<'a> {
        Form {
            items: self.items.get(index..).unwrap_or(&[]),
            line: self.line,
        }
    }

    /// The first item as an atom: the keyword the list starts with.
    pub fn head(&self, what: &str) -> ParseResult<&'a str> {
        self.get(0, what)?.expect_atom(what)
    }

    /// An error unless the list has exactly `count` items.
    pub fn expect_len(&self, count: usize, what: &str) -> ParseResult<()> {
        if self.items.len() != count {
            let form: Vec<String> = self.items.iter().map(SExpr::to_string).collect();
            return Err(ParseError::malformed(
                self.line,
                format!("{what}, got ({})", form.join(" ")),
            ));
        }
        Ok(())
    }
}

impl fmt::Display for SExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SExpr::Atom(text, _) => f.write_str(text),
            SExpr::List(items, _) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// Reads a PDDL file and parses it into an S-expression.
pub fn parse_nested_list(path: &Path) -> Result<SExpr, crate::TranslateError> {
    let content = fs::read_to_string(path).map_err(|source| crate::TranslateError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(parse_nested_list_string(&content).map_err(|error| error.in_file(path))?)
}

/// Parses one S-expression, the whole of `input`.
pub fn parse_nested_list_string(input: &str) -> ParseResult<SExpr> {
    let mut tokens = tokenize(input).into_iter().peekable();
    let result = parse_list_aux(&mut tokens, 1)?;
    if let Some((_, line)) = tokens.peek() {
        return Err(ParseError::malformed(
            *line,
            "unexpected tokens after the end of the outermost list",
        ));
    }
    Ok(result)
}

/// The tokens of `input` -- `(`, `)` and lowercased atoms -- with their
/// lines, comments (from `;` to the end of the line) stripped.
fn tokenize(input: &str) -> Vec<(String, u32)> {
    let mut tokens = vec![];
    for (line_index, line) in input.lines().enumerate() {
        let line_number =
            u32::try_from(line_index + 1).expect("a PDDL file has fewer than 2^32 lines");
        let line = line.split(';').next().unwrap_or_default();
        let line = line.replace('(', " ( ").replace(')', " ) ");
        for token in line.split_whitespace() {
            tokens.push((token.to_lowercase(), line_number));
        }
    }
    tokens
}

fn parse_list_aux(
    tokens: &mut std::iter::Peekable<std::vec::IntoIter<(String, u32)>>,
    last_line: u32,
) -> ParseResult<SExpr> {
    let (token, line) = tokens
        .next()
        .ok_or_else(|| ParseError::malformed(last_line, "unexpected end of input"))?;
    match token.as_str() {
        "(" => {
            let mut items = vec![];
            loop {
                match tokens.peek() {
                    Some((next, _)) if next == ")" => {
                        tokens.next();
                        break;
                    }
                    Some(_) => items.push(parse_list_aux(tokens, line)?),
                    None => {
                        return Err(ParseError::malformed(
                            line,
                            "this list has no closing parenthesis",
                        ));
                    }
                }
            }
            Ok(SExpr::List(items, line))
        }
        ")" => Err(ParseError::malformed(
            line,
            "unexpected closing parenthesis",
        )),
        _ => Ok(SExpr::Atom(token, line)),
    }
}
