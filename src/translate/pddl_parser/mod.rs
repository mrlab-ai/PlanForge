use nom::branch::alt;
use nom::bytes::complete::{take_while1, take_while};
use nom::character::complete::{char, multispace1};
use nom::combinator::{map, recognize};
use nom::error::VerboseError;
use nom::multi::{many0, many1};
use nom::sequence::{delimited, preceded};
use nom::IResult;

// Re-export all modules
pub mod lisp_parser;
pub mod parsing_functions;
pub mod pddl_file;
pub mod pretty_print;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SExpr {
    Atom(String),
    List(Vec<SExpr>),
}

impl SExpr {
    pub fn is_atom(&self) -> bool {
        matches!(self, SExpr::Atom(_))
    }
    
    pub fn is_list(&self) -> bool {
        matches!(self, SExpr::List(_))
    }
    
    pub fn as_atom(&self) -> Option<&String> {
        match self {
            SExpr::Atom(s) => Some(s),
            _ => None,
        }
    }
    
    pub fn as_list(&self) -> Option<&Vec<SExpr>> {
        match self {
            SExpr::List(v) => Some(v),
            _ => None,
        }
    }
}

fn is_atom_char(c: char) -> bool {
    !c.is_whitespace() && c != '(' && c != ')' && c != ';'
}

fn ws<'a>(input: &'a str) -> IResult<&'a str, (), VerboseError<&'a str>> {
    let (i, _) = many0(alt((multispace1, map(preceded(char(';'), take_while(|c| c != '\n')), |_| ""))))(input)?;
    Ok((i, ()))
}

fn atom(input: &str) -> IResult<&str, SExpr, VerboseError<&str>> {
    let (i, s) = recognize(many1(take_while1(is_atom_char)))(input)?;
    Ok((i, SExpr::Atom(s.to_lowercase())))  // Convert to lowercase to match Python behavior
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

pub fn parse_sexprs(input: &str) -> Result<Vec<SExpr>, String> {
    let mut rest = input;
    let mut out = Vec::new();
    loop {
        // consume whitespace/comments
        match ws(rest) {
            Ok((i, _)) => rest = i,
            Err(_) => {}
        }
        if rest.trim().is_empty() {
            break;
        }
        match sexpr(rest) {
            Ok((i, s)) => {
                out.push(s);
                rest = i;
            }
            Err(e) => return Err(format!("parse error: {:?}\nnear: {}", e, &rest[..rest.len().min(80)])),
        }
    }
    Ok(out)
}

// Re-export from submodules - these will be empty for now
pub use lisp_parser::{parse_lisp, LispParser};
pub use parsing_functions::*;
pub use pddl_file::*;
pub use pretty_print::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn parse_pfile1_smoke() {
        let s = fs::read_to_string("pddl/pfile1.pddl").expect("read pddl file");
        let sexprs = parse_sexprs(&s).expect("parse should succeed");
        assert!(!sexprs.is_empty());
        // first form should be (define ...)
        match &sexprs[0] {
            SExpr::List(items) => {
                match &items[0] {
                    SExpr::Atom(a) => assert_eq!(a.to_lowercase(), "define"),
                    _ => panic!("expected atom define"),
                }
            }
            _ => panic!("expected list"),
        }
    }
}