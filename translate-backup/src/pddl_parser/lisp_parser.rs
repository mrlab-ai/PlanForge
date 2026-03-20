#[cfg(test)]
mod tests;

use nom::branch::alt;
use nom::bytes::complete::{take_while, take_while1};
use nom::character::complete::{char, multispace1};
use nom::combinator::{map, recognize};
use nom::error::{convert_error, VerboseError};
use nom::multi::many0;
use nom::sequence::{delimited, preceded};
use nom::IResult;

const COMMENT_CHAR: char = ';';
const ERROR_SNIPPET_LEN: usize = 80;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    value: String,
}

impl ParseError {
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl std::error::Error for ParseError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SExpr {
    Atom(String),
    List(Vec<SExpr>),
}

fn is_atom_char(c: char) -> bool {
    !c.is_whitespace() && c != '(' && c != ')' && c != ';'
}

fn comment(input: &str) -> IResult<&str, &str, VerboseError<&str>> {
    preceded(char(COMMENT_CHAR), take_while(|c| c != '\n'))(input)
}

fn ws(input: &str) -> IResult<&str, (), VerboseError<&str>> {
    let (i, _) = many0(alt((multispace1, map(comment, |_| ""))))(input)?;
    Ok((i, ()))
}

fn atom(input: &str) -> IResult<&str, SExpr, VerboseError<&str>> {
    let (i, s) = recognize(take_while1(is_atom_char))(input)?;
    Ok((i, SExpr::Atom(s.to_ascii_lowercase())))
}

fn list(input: &str) -> IResult<&str, SExpr, VerboseError<&str>> {
    let (i, vec) = delimited(
        preceded(ws, char('(')),
        many0(preceded(ws, sexpr)),
        preceded(ws, char(')')),
    )(input)?;
    Ok((i, SExpr::List(vec)))
}

fn sexpr(input: &str) -> IResult<&str, SExpr, VerboseError<&str>> {
    alt((list, preceded(ws, atom)))(input)
}

fn consume_ws_comments(input: &str) -> Result<&str, String> {
    ws(input)
        .map(|(i, _)| i)
        .map_err(|e| format_parse_error(input, e))
}

fn format_parse_error(input: &str, err: nom::Err<VerboseError<&str>>) -> String {
    match err {
        nom::Err::Error(e) | nom::Err::Failure(e) => {
            let snippet = input.trim_start();
            let preview = &snippet[..snippet.len().min(ERROR_SNIPPET_LEN)];
            format!(
                "parse error:\n{}\nnear: {}",
                convert_error(input, e),
                preview
            )
        }
        nom::Err::Incomplete(_) => "parse error: incomplete input".to_string(),
    }
}

pub fn tokenize(input: &str) -> Result<Vec<String>, ParseError> {
    let mut tokens = Vec::new();
    for raw_line in input.lines() {
        let line = raw_line.split(COMMENT_CHAR).next().unwrap_or("");
        if !line.is_ascii() {
            return Err(ParseError::new(format!(
                "Non-ASCII character outside comment: {}",
                raw_line
            )));
        }
        let normalized = line
            .replace('(', " ( ")
            .replace(')', " ) ")
            .replace('?', " ?");
        tokens.extend(
            normalized
                .split_whitespace()
                .map(|token| token.to_ascii_lowercase()),
        );
    }
    Ok(tokens)
}

pub fn parse_list_aux(
    tokens: &[String],
    start_index: usize,
) -> Result<(Vec<SExpr>, usize), ParseError> {
    let mut result = Vec::new();
    let mut index = start_index;
    while index < tokens.len() {
        match tokens[index].as_str() {
            ")" => return Ok((result, index + 1)),
            "(" => {
                let (items, next_index) = parse_list_aux(tokens, index + 1)?;
                result.push(SExpr::List(items));
                index = next_index;
            }
            token => {
                result.push(SExpr::Atom(token.to_string()));
                index += 1;
            }
        }
    }
    Err(ParseError::new("Unexpected end of token stream."))
}

pub fn parse_sexprs(input: &str) -> Result<Vec<SExpr>, String> {
    let mut rest = input;
    let mut out = Vec::new();
    loop {
        rest = consume_ws_comments(rest)?;
        if rest.trim().is_empty() {
            break;
        }
        match sexpr(rest) {
            Ok((i, s)) => {
                out.push(s);
                rest = i;
            }
            Err(e) => {
                return Err(format_parse_error(rest, e));
            }
        }
    }
    Ok(out)
}

pub fn parse_nested_list(input: &str) -> Result<Vec<SExpr>, ParseError> {
    let tokens = tokenize(input)?;
    if tokens.first().map(String::as_str) != Some("(") {
        return Err(ParseError::new(format!(
            "Expected '(', got {}.",
            tokens
                .first()
                .cloned()
                .unwrap_or_else(|| "<eof>".to_string())
        )));
    }
    let (result, next_index) = parse_list_aux(&tokens, 1)?;
    if next_index != tokens.len() {
        return Err(ParseError::new(format!(
            "Unexpected token: {}.",
            tokens[next_index]
        )));
    }
    Ok(result)
}
